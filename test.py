from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from model.model import QwenGazeIntegratedModel
from model.utils.data_utils import (
    build_prompt,
    build_vocab_embedding_matrix,
    load_test_groups,
    load_test_label_map,
    sanitize_bbox_pixels,
)
from model.trainer import (
    QwenBackboneAdapter,
    infer_hidden_dim,
    load_yaml_config,
    parse_dtype,
    resolve_path,
)


def _default(defaults: dict[str, Any], key: str, fallback: Any) -> Any:
    return defaults.get(key, fallback)


def _load_vocab(vocab2id_path: Path | None) -> tuple[dict[str, int], dict[int, str]]:
    if vocab2id_path is None or (not vocab2id_path.exists()):
        return {}, {}
    try:
        obj = json.loads(vocab2id_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    if not isinstance(obj, dict):
        return {}, {}
    vocab2id: dict[str, int] = {}
    id2label: dict[int, str] = {}
    for k, v in obj.items():
        try:
            idx = int(v)
        except Exception:
            continue
        vocab2id[str(k)] = idx
        id2label[idx] = str(k)
    return vocab2id, id2label


def _heatmap_overlay(
    scene: Image.Image,
    heatmap: torch.Tensor,
    alpha: float = 0.45,
) -> Image.Image:
    scene_np = np.array(scene.convert("RGB"), dtype=np.float32)
    hm = heatmap.detach().float().cpu().numpy()
    hm = np.clip(hm, 0.0, 1.0)
    hm_img = Image.fromarray((hm * 255.0).astype(np.uint8), mode="L")
    hm_img = hm_img.resize(scene.size, resample=Image.Resampling.BILINEAR)
    hm_np = np.array(hm_img, dtype=np.float32) / 255.0

    color = np.zeros_like(scene_np)
    color[..., 0] = 255.0  # red
    a = np.clip(float(alpha), 0.0, 1.0) * hm_np[..., None]
    out = scene_np * (1.0 - a) + color * a
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _draw_annotations(
    image: Image.Image,
    bbox_px: tuple[float, float, float, float],
    pred_point: tuple[float, float],
    gt_points: list[tuple[float, float]],
    gt_label_text: str | None = None,
    pred_label_text: str | None = None,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    x1, y1, x2, y2 = sanitize_bbox_pixels(bbox_px, width=w, height=h)
    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 255), width=2)

    px = int(round(np.clip(pred_point[0], 0.0, 1.0) * (w - 1)))
    py = int(round(np.clip(pred_point[1], 0.0, 1.0) * (h - 1)))
    r = 5
    draw.ellipse((px - r, py - r, px + r, py + r), outline=(255, 255, 0), width=2)

    for gx, gy in gt_points:
        tx = int(round(np.clip(gx, 0.0, 1.0) * (w - 1)))
        ty = int(round(np.clip(gy, 0.0, 1.0) * (h - 1)))
        rr = 4
        draw.ellipse((tx - rr, ty - rr, tx + rr, ty + rr), outline=(0, 255, 0), width=2)

    overlay_text = f"GT : {gt_label_text or 'N/A'}\npred : {pred_label_text or 'N/A'}"
    margin = 10
    pad = 8
    try:
        tb = draw.multiline_textbbox((0, 0), overlay_text, spacing=4)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
    except Exception:
        tw = int(max(len(line) for line in overlay_text.splitlines()) * 8)
        th = int(len(overlay_text.splitlines()) * 16)
    x0 = max(0, w - margin - tw - (2 * pad))
    y0 = margin
    x1 = min(w - 1, x0 + tw + (2 * pad))
    y1 = min(h - 1, y0 + th + (2 * pad))
    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0), outline=(255, 255, 255), width=1)
    draw.multiline_text((x0 + pad, y0 + pad), overlay_text, fill=(255, 255, 255), spacing=4)
    return out


def _extract_gt_label_text(group: Any, id2label: dict[int, str]) -> str | None:
    for attr in ("label", "label_text", "class_name", "object_label", "target_label"):
        val = getattr(group, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for attr in ("label_id", "class_id", "object_label_id", "target_label_id"):
        val = getattr(group, attr, None)
        if isinstance(val, (int, np.integer)):
            idx = int(val)
            return id2label.get(idx, str(idx))
    return None


def _l2(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return float((dx * dx + dy * dy) ** 0.5)


def build_arg_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("QWEN_GazeEstimation one-shot tester")
    p.add_argument("--config", type=str, default=str(_default(defaults, "config", "config.yaml")))
    p.add_argument("--checkpoint_dir", type=str, default=str(_default(defaults, "checkpoint_dir", "")))
    p.add_argument("--model_path", type=str, default=str(_default(defaults, "model_path", "model/Qwen3-VL-4B-Instruct")))
    p.add_argument("--test_ann", type=str, default=str(_default(defaults, "test_ann", "data/gazefollow_extended/test_annotations_release.txt")))
    p.add_argument("--test_image_root", type=str, default=str(_default(defaults, "test_image_root", "data/gazefollow_extended/test2")))
    p.add_argument("--test_labels", type=str, default=str(_default(defaults, "test_labels", "data/gazefollow/gaze-labels-test.csv")))
    p.add_argument("--vocab2id", type=str, default=str(_default(defaults, "vocab2id", "data/gazefollow/vocab2id.json")))
    p.add_argument("--output_dir", type=str, default="outputs/test_vis")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--test_split_prefix", type=str, default=str(_default(defaults, "test_split_prefix", "test2/")))
    p.add_argument("--test_strip_split_prefix", dest="test_strip_split_prefix", action="store_true")
    p.add_argument("--no_test_strip_split_prefix", dest="test_strip_split_prefix", action="store_false")
    p.set_defaults(test_strip_split_prefix=bool(_default(defaults, "test_strip_split_prefix", True)))
    p.add_argument("--test_bbox_round_decimals", type=int, default=int(_default(defaults, "test_bbox_round_decimals", 3)))

    p.add_argument("--scene_h", type=int, default=int(_default(defaults, "scene_h", 512)))
    p.add_argument("--scene_w", type=int, default=int(_default(defaults, "scene_w", 512)))
    p.add_argument("--head_h", type=int, default=int(_default(defaults, "head_h", 224)))
    p.add_argument("--head_w", type=int, default=int(_default(defaults, "head_w", 224)))
    p.add_argument("--heatmap_h", type=int, default=int(_default(defaults, "heatmap_h", 64)))
    p.add_argument("--heatmap_w", type=int, default=int(_default(defaults, "heatmap_w", 64)))
    p.add_argument("--head_tokens", type=int, default=int(_default(defaults, "head_tokens", 64)))
    p.add_argument("--text_tokens", type=int, default=int(_default(defaults, "text_tokens", 64)))
    p.add_argument("--max_text_length", type=int, default=int(_default(defaults, "max_text_length", 128)))
    p.add_argument("--conditioning_mode", type=str, default=str(_default(defaults, "conditioning_mode", "film")))
    p.add_argument("--pool_mode", type=str, default=str(_default(defaults, "pool_mode", "mean")))
    p.add_argument("--num_conditioning_heads", type=int, default=int(_default(defaults, "num_conditioning_heads", 8)))
    p.add_argument("--num_conditioning_layers", type=int, default=int(_default(defaults, "num_conditioning_layers", 1)))
    p.add_argument("--dropout", type=float, default=float(_default(defaults, "dropout", 0.1)))
    p.add_argument("--recognition_objective", type=str, default=str(_default(defaults, "recognition_objective", "infonce")))
    p.add_argument("--label_emb_dim", type=int, default=int(_default(defaults, "label_emb_dim", 512)))
    p.add_argument("--logit_scale_init", type=float, default=float(_default(defaults, "logit_scale_init", 0.07)))
    p.add_argument("--label_embed_dir", type=str, default=str(_default(defaults, "label_embed_dir", "data/gazefollow/label-embeds")))
    p.add_argument("--normalize_label_emb", dest="normalize_label_emb", action="store_true")
    p.add_argument("--no_normalize_label_emb", dest="normalize_label_emb", action="store_false")
    p.set_defaults(normalize_label_emb=bool(_default(defaults, "normalize_label_emb", True)))
    p.add_argument("--lambda_cls", type=float, default=float(_default(defaults, "lambda_cls", 1.0)))
    p.add_argument("--label_smoothing", type=float, default=float(_default(defaults, "label_smoothing", 0.0)))
    p.add_argument("--cls_ignore_index", type=int, default=int(_default(defaults, "cls_ignore_index", -100)))
    p.add_argument("--prompt_template", type=str, default=str(_default(defaults, "prompt_template", "")))
    p.add_argument("--prompt_text", type=str, default=str(_default(defaults, "prompt_text", "")))
    p.add_argument("--head_text", type=str, default=str(_default(defaults, "head_text", "Target subject head crop.")))

    p.add_argument("--lora_r", type=int, default=int(_default(defaults, "lora_r", 16)))
    p.add_argument("--lora_alpha", type=int, default=int(_default(defaults, "lora_alpha", 32)))
    p.add_argument("--lora_dropout", type=float, default=float(_default(defaults, "lora_dropout", 0.05)))
    p.add_argument("--lora_bias", type=str, default=str(_default(defaults, "lora_bias", "none")))
    lora_target_default = _default(defaults, "lora_target_modules", "q_proj,k_proj,v_proj,o_proj")
    if isinstance(lora_target_default, list):
        lora_target_default = ",".join(str(x) for x in lora_target_default)
    p.add_argument("--lora_target_modules", type=str, default=str(lora_target_default))

    p.add_argument("--device", type=str, default=str(_default(defaults, "device", "cuda")))
    p.add_argument("--dtype", type=str, default=str(_default(defaults, "dtype", "bfloat16")))
    p.add_argument("--attn_implementation", type=str, default=str(_default(defaults, "attn_implementation", "sdpa")))
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--topk", type=int, default=3)
    return p


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="config.yaml")
    cfg_args, _ = config_parser.parse_known_args()
    config_defaults = load_yaml_config(resolve_path(cfg_args.config))
    config_defaults["config"] = str(cfg_args.config)
    args = build_arg_parser(config_defaults).parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = resolve_path(args.model_path)
    ckpt_dir = resolve_path(args.checkpoint_dir) if str(args.checkpoint_dir).strip() else None
    test_ann = resolve_path(args.test_ann)
    test_image_root = resolve_path(args.test_image_root)
    test_labels = resolve_path(args.test_labels)
    vocab2id_path = resolve_path(args.vocab2id)
    label_embed_dir = resolve_path(args.label_embed_dir)
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab2id, id2label = _load_vocab(vocab2id_path)
    vocab2id_lower = {str(k).strip().lower(): int(v) for k, v in vocab2id.items()}
    test_label_map, test_label_text_map, test_label_ids_map, test_label_stats = load_test_label_map(
        test_labels,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
    )
    print(
        "[INFO] test label map: "
        f"rows={test_label_stats.get('rows', 0)} mapped={test_label_stats.get('mapped', 0)} "
        f"missing_text={test_label_stats.get('missing_text', 0)} "
        f"unknown_text={test_label_stats.get('unknown_text', 0)} "
        f"conflicts={test_label_stats.get('conflicts', 0)}"
    )
    groups = load_test_groups(
        annotation_file=test_ann,
        image_root=test_image_root,
        test_label_map=test_label_map,
        test_label_text_map=test_label_text_map,
        test_label_ids_map=test_label_ids_map,
        split_prefix=args.test_split_prefix,
        strip_split_prefix=bool(args.test_strip_split_prefix),
        bbox_round_decimals=int(args.test_bbox_round_decimals),
        max_groups=0,
    )
    if not groups:
        raise RuntimeError("No test groups found.")
    st = max(0, int(args.start_index))
    ed = min(len(groups), st + max(1, int(args.num_samples)))
    groups = groups[st:ed]
    print(f"[INFO] selected groups: {len(groups)} (start={st})")

    num_classes = len(vocab2id)

    aux_state = None
    ckpt_num_classes = -1
    if ckpt_dir is not None and (ckpt_dir / "heads.pt").exists():
        aux_state = torch.load(ckpt_dir / "heads.pt", map_location="cpu")
        if isinstance(aux_state, dict):
            cls_state = aux_state.get("classifier")
            if isinstance(cls_state, dict) and ("classifier.weight" in cls_state):
                try:
                    ckpt_num_classes = int(cls_state["classifier.weight"].shape[0])
                except Exception:
                    pass
        if ckpt_num_classes > 0:
            if num_classes > 0 and num_classes != ckpt_num_classes:
                print(
                    "[WARN] vocab class count and checkpoint class count mismatch: "
                    f"vocab={num_classes}, ckpt={ckpt_num_classes}. "
                    "Using checkpoint class count."
                )
            num_classes = ckpt_num_classes if num_classes != ckpt_num_classes else num_classes

    device = torch.device(args.device)
    load_dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and load_dtype in {torch.bfloat16, torch.float16}:
        load_dtype = torch.float32
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if load_dtype != "auto":
        model_kwargs["dtype"] = load_dtype

    processor_path = ckpt_dir / "processor" if (ckpt_dir is not None and (ckpt_dir / "processor").exists()) else model_path
    processor = AutoProcessor.from_pretrained(str(processor_path), trust_remote_code=True)
    base_qwen = AutoModelForImageTextToText.from_pretrained(str(model_path), **model_kwargs)

    adapter_dir = ckpt_dir / "lora_adapter" if ckpt_dir is not None else None
    if adapter_dir is not None and adapter_dir.exists():
        qwen_model = PeftModel.from_pretrained(base_qwen, model_id=str(adapter_dir), is_trainable=False)
        print(f"[INFO] loaded checkpoint adapter: {adapter_dir}")
    else:
        qwen_model = base_qwen
        print("[INFO] checkpoint not set (or missing adapter). Running in zero-shot mode.")
    qwen_model.to(device)
    qwen_model.eval()

    hidden_dim = infer_hidden_dim(qwen_model, model_path=model_path)
    if hidden_dim <= 0:
        raise RuntimeError("Failed to infer hidden_dim.")

    backbone = QwenBackboneAdapter(
        qwen_model=qwen_model,
        processor=processor,
        scene_tokens=None,
        head_tokens=args.head_tokens,
        text_tokens=args.text_tokens,
        max_text_length=args.max_text_length,
        head_text=args.head_text,
    )
    model = QwenGazeIntegratedModel(
        backbone=backbone,
        hidden_dim=hidden_dim,
        scene_grid_size=None,
        num_classes=num_classes if num_classes > 0 else None,
        conditioning_mode=args.conditioning_mode,
        pool_mode=args.pool_mode,
        scene_input_size=(args.scene_h, args.scene_w),
        head_input_size=(args.head_h, args.head_w),
        heatmap_size=(args.heatmap_h, args.heatmap_w),
        num_conditioning_heads=args.num_conditioning_heads,
        num_conditioning_layers=args.num_conditioning_layers,
        dropout=args.dropout,
        recognition_objective=str(args.recognition_objective),
        label_emb_dim=int(args.label_emb_dim),
        logit_scale_init=float(args.logit_scale_init),
        lambda_cls=args.lambda_cls,
        label_smoothing=args.label_smoothing,
        cls_ignore_index=args.cls_ignore_index,
    ).to(device)
    if num_classes > 0:
        vocab_emb = build_vocab_embedding_matrix(
            vocab2id=vocab2id,
            label_embed_dir=label_embed_dir,
            label_emb_dim=int(args.label_emb_dim),
            normalize=bool(args.normalize_label_emb),
        )
        if vocab_emb is not None:
            model.set_vocab_embeddings(vocab_emb.to(device))
    model.eval()

    if isinstance(aux_state, dict):
        if "summary" in aux_state:
            model.summary.load_state_dict(aux_state["summary"], strict=True)
        if "conditioner" in aux_state:
            model.conditioner.load_state_dict(aux_state["conditioner"], strict=True)
        if "localizer" in aux_state:
            model.localizer.load_state_dict(aux_state["localizer"], strict=True)
        if model.classifier is not None and aux_state.get("classifier") is not None:
            try:
                model.classifier.load_state_dict(aux_state["classifier"], strict=True)
            except Exception as e:
                print(f"[WARN] classifier strict load failed: {e}; retrying strict=False")
                model.classifier.load_state_dict(aux_state["classifier"], strict=False)
        if ckpt_dir is not None:
            print(f"[INFO] loaded checkpoint heads: {ckpt_dir / 'heads.pt'}")

    results: list[dict[str, Any]] = []
    avg_l2s: list[float] = []
    min_l2s: list[float] = []
    with torch.no_grad():
        for i, g in enumerate(groups):
            with Image.open(g.image_path) as img:
                scene = img.convert("RGB")
            w, h = scene.size
            x1, y1, x2, y2 = sanitize_bbox_pixels(g.bbox_px, width=w, height=h)
            head = scene.crop((x1, y1, x2, y2))
            bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)
            prompt = build_prompt(bbox_norm, args.prompt_template, args.prompt_text)

            with torch.autocast(
                device_type=device.type,
                dtype=(torch.bfloat16 if load_dtype == "auto" else load_dtype),
                enabled=(device.type == "cuda"),
            ):
                out = model(
                    scene_image=[scene],
                    head_image=[head],
                    text_inputs=[prompt],
                    use_softargmax=False,
                )

            heatmap = out["heatmap"][0, 0].detach().cpu()
            pred_xy = out["point_hard"][0].detach().cpu()
            pred_point = (float(pred_xy[0].item()), float(pred_xy[1].item()))

            pred_label_id = None
            pred_label_text = None
            topk_ids: list[int] = []
            topk_probs: list[float] = []
            if ("logits" in out) and (out["logits"] is not None):
                probs = torch.softmax(out["logits"][0].detach().float().cpu(), dim=-1)
                kk = int(min(max(int(args.topk), 1), probs.numel()))
                topv, topi = torch.topk(probs, k=kk, dim=-1)
                topk_ids = [int(x) for x in topi.tolist()]
                topk_probs = [float(x) for x in topv.tolist()]
                pred_label_id = int(topk_ids[0])
                pred_label_text = id2label.get(pred_label_id, str(pred_label_id))

            gt_label_text = _extract_gt_label_text(g, id2label)

            vis = _heatmap_overlay(scene, heatmap, alpha=float(args.alpha))
            vis = _draw_annotations(
                vis,
                g.bbox_px,
                pred_point=pred_point,
                gt_points=g.gt_points,
                gt_label_text=gt_label_text,
                pred_label_text=pred_label_text,
            )

            stem = Path(g.image_rel).stem
            out_img = out_dir / f"{i:03d}_{stem}_overlay.png"
            vis.save(out_img)

            result = {
                "index": i,
                "image_rel": g.image_rel,
                "image_path": str(g.image_path),
                "output_image": str(out_img),
                "pred_point": {"x": pred_point[0], "y": pred_point[1]},
                "gt_points": [{"x": float(x), "y": float(y)} for x, y in g.gt_points],
                "pred_label_id": pred_label_id,
                "pred_label_text": id2label.get(pred_label_id, None) if pred_label_id is not None else None,
                "gt_label_text": gt_label_text,
                "topk": [
                    {
                        "label_id": lid,
                        "label_text": id2label.get(lid, None),
                        "prob": prob,
                    }
                    for lid, prob in zip(topk_ids, topk_probs)
                ],
            }
            results.append(result)
            if g.gt_points:
                mean_gt = (
                    sum(float(x) for x, _ in g.gt_points) / float(len(g.gt_points)),
                    sum(float(y) for _, y in g.gt_points) / float(len(g.gt_points)),
                )
                avg_l2s.append(_l2(pred_point, mean_gt))
                min_l2s.append(min(_l2(pred_point, (float(x), float(y))) for x, y in g.gt_points))

            topk_str = ", ".join(f"{lid}:{p:.3f}" for lid, p in zip(topk_ids, topk_probs))
            print(f"[{i}] {g.image_rel}")
            print(f"  pred_point=({pred_point[0]:.4f}, {pred_point[1]:.4f})")
            if pred_label_id is not None:
                lbl = id2label.get(pred_label_id, "")
                print(f"  pred_label={pred_label_id}" + (f" ({lbl})" if lbl else ""))
                print(f"  topk={topk_str}")
            print(f"  vis={out_img}")

    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if avg_l2s and min_l2s:
        print(f"[METRICS] Avg L2={sum(avg_l2s) / len(avg_l2s):.6f} Min L2={sum(min_l2s) / len(min_l2s):.6f}")
    print(f"[DONE] saved {len(results)} results to: {out_dir}")


if __name__ == "__main__":
    main()
