from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.datasets import GazeTestDataset, draw_head_bbox_prompt
from model.model import QwenTextGenerationModel
from model.modules.preprocess import resize_pil_image
from model.trainer import (
    build_id2label,
    infer_num_classes,
    init_base_model,
    init_processor,
    parse_dtype,
    resolve_model_source,
    resolve_path,
    set_seed,
    to_autocast_dtype,
)
from model.utils.config_parser import build_parser, load_yaml_config
from model.utils.data_utils import (
    load_test_groups,
    load_test_label_map,
    load_vocab2id,
)
from model.utils.eval_utils import collect_generation_samples
from model.utils.special_tokens import register_gaze_special_tokens
from model.utils.processor_collate import QwenTestCollator


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Visualize test predictions on images")
    p.add_argument("--config", type=str, default="sft.yaml")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/qwen_ge/best")
    p.add_argument("--output_dir", type=str, default="outputs/test_visualization")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--thumb_width", type=int, default=640)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--num_workers", type=int, default=-1)
    p.add_argument("--generation_num_beams", type=int, default=-1)
    p.add_argument("--generation_max_new_tokens", type=int, default=-1)
    p.add_argument("--test_batch_size", type=int, default=-1)
    return p


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _draw_point(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], r: int) -> None:
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=max(2, r // 2))
    draw.line((x - r, y, x + r, y), fill=color, width=max(2, r // 2))
    draw.line((x, y - r, x, y + r), fill=color, width=max(2, r // 2))


def _fit_multiline(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    words = str(text or "").split()
    if not words:
        return ""
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def annotate_sample(
    *,
    group: Any,
    sample: dict[str, Any],
    index: int,
    thumb_width: int,
) -> Image.Image:
    with Image.open(group.image_path) as src:
        image = src.convert("RGB")
    image = draw_head_bbox_prompt(
        image,
        x1=int(group.bbox_px[0]),
        y1=int(group.bbox_px[1]),
        x2=int(group.bbox_px[2]),
        y2=int(group.bbox_px[3]),
    )

    w, h = image.size
    draw = ImageDraw.Draw(image)
    side = max(1, min(w, h))
    gt_r = max(5, side // 110)
    pred_r = max(6, side // 95)

    for gx, gy in list(getattr(group, "gt_points", []) or []):
        _draw_point(draw, _clamp01(gx) * w, _clamp01(gy) * h, (32, 196, 96), gt_r)

    parsed = sample.get("parsed", {})
    point_xy = parsed.get("point_xy")
    if isinstance(point_xy, (list, tuple)) and len(point_xy) == 2:
        _draw_point(
            draw,
            _clamp01(float(point_xy[0])) * w,
            _clamp01(float(point_xy[1])) * h,
            (230, 57, 70),
            pred_r,
        )

    panel_w = 360
    canvas = Image.new("RGB", (w + panel_w, h), (250, 250, 248))
    canvas.paste(image, (0, 0))
    panel = ImageDraw.Draw(canvas)

    title_font = _load_font(22)
    body_font = _load_font(16)
    small_font = _load_font(14)
    x0 = w + 18
    y = 18

    panel.text((x0, y), f"Sample {index:02d}", fill=(18, 18, 18), font=title_font)
    y += 34

    object_id = parsed.get("object_id")
    gt_obj_id = sample.get("gt_object_id", "")
    exact_match = (
        sample.get("generated_text", "") == sample.get("target_text", "")
        if sample.get("target_text_valid") else None
    )
    items = [
        f"image: {sample.get('image_rel', '')}",
        f"gt label: {sample.get('target_label_text', '')}  (id={gt_obj_id})",
        f"pred object id: {object_id}",
        f"pred point xy: {[round(float(v), 4) for v in point_xy] if point_xy else None}",
        f"pred point bins: {parsed.get('point_bins')}",
        f"avg_l2: {sample.get('avg_l2')}",
        f"min_l2: {sample.get('min_l2')}",
        f"fmt_valid: {parsed.get('valid_format', False)}  exact: {exact_match}",
        f"generated: {sample.get('generated_text', '')}",
        "legend: green=GT gaze, red=pred gaze",
    ]
    max_text_w = panel_w - 30
    for item in items:
        text = _fit_multiline(panel, item, body_font, max_text_w)
        panel.multiline_text((x0, y), text, fill=(35, 35, 35), font=body_font, spacing=4)
        bbox = panel.multiline_textbbox((x0, y), text, font=body_font, spacing=4)
        y = bbox[3] + 14

    prompt = _fit_multiline(panel, f"prompt: {sample.get('prompt_text', '')}", small_font, max_text_w)
    panel.multiline_text((x0, y), prompt, fill=(85, 85, 85), font=small_font, spacing=3)

    if thumb_width > 0 and canvas.width > thumb_width:
        scale = float(thumb_width) / float(canvas.width)
        thumb_h = max(1, int(round(canvas.height * scale)))
        canvas = resize_pil_image(canvas, (thumb_h, thumb_width))
    return canvas


def build_contact_sheet(images: list[Image.Image], cols: int = 2, bg: tuple[int, int, int] = (245, 245, 242)) -> Image.Image | None:
    if not images:
        return None
    cols = max(1, int(cols))
    rows = (len(images) + cols - 1) // cols
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    gap = 14
    sheet = Image.new(
        "RGB",
        (cols * cell_w + (cols + 1) * gap, rows * cell_h + (rows + 1) * gap),
        bg,
    )
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = gap + col * (cell_w + gap)
        y = gap + row * (cell_h + gap)
        sheet.paste(img, (x, y))
    return sheet


def main() -> None:
    script_args = make_parser().parse_args()

    config_path = resolve_path(script_args.config)
    config_defaults = load_yaml_config(config_path)
    config_defaults["config"] = str(script_args.config)
    args = build_parser(defaults=config_defaults).parse_args([])

    # Script overrides
    args.config = str(script_args.config)
    args.checkpoint_dir = str(script_args.checkpoint_dir)
    args.output_dir = str(script_args.output_dir)
    if str(script_args.device).strip():
        args.device = str(script_args.device).strip()
    if int(script_args.num_workers) >= 0:
        args.num_workers = int(script_args.num_workers)
    if int(script_args.generation_num_beams) > 0:
        args.generation_num_beams = int(script_args.generation_num_beams)
    if int(script_args.generation_max_new_tokens) > 0:
        args.generation_max_new_tokens = int(script_args.generation_max_new_tokens)
    if int(script_args.test_batch_size) > 0:
        args.test_batch_size = int(script_args.test_batch_size)

    set_seed(int(args.seed))

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_args.json").write_text(
        json.dumps(
            {
                "config": str(config_path),
                "checkpoint_dir": str(resolve_path(args.checkpoint_dir)),
                "output_dir": str(out_dir),
                "num_samples": int(script_args.num_samples),
                "start_index": int(script_args.start_index),
                "thumb_width": int(script_args.thumb_width),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.device == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but no GPU is available.")
    device = torch.device(args.device)

    model_path = resolve_model_source(args.model_path)
    checkpoint_dir = resolve_path(args.checkpoint_dir) if str(args.checkpoint_dir).strip() else None
    test_ann = resolve_path(args.test_ann)
    test_image_root = resolve_path(args.test_image_root)
    test_labels = resolve_path(args.test_labels)
    vocab2id_path = resolve_path(args.vocab2id)

    vocab2id, vocab2id_lower = load_vocab2id(vocab2id_path)
    num_classes = infer_num_classes(vocab2id, vocab2id_path)
    id2label = build_id2label(vocab2id)
    coord_bins = int(getattr(args, "coord_bins", 1000))

    _resize_mode = str(getattr(args, "image_resize_mode", "native")).strip().lower()
    _fixed_resize = (_resize_mode == "fixed")
    _scene_size: tuple[int, int] | None = (
        (int(args.scene_h), int(args.scene_w)) if _fixed_resize else None
    )
    processor = init_processor(
        model_path=model_path,
        checkpoint_dir=checkpoint_dir,
        min_pixels=None if _fixed_resize else int(getattr(args, "min_pixels", 12544)),
        max_pixels=None if _fixed_resize else int(getattr(args, "max_pixels", 2007040)),
    )
    register_gaze_special_tokens(
        processor.tokenizer,
        num_classes=num_classes,
        coord_bins=coord_bins,
    )

    test_label_map, test_label_text_map, test_label_ids_map, _ = load_test_label_map(
        test_labels,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
    )
    test_groups = load_test_groups(
        annotation_file=test_ann,
        image_root=test_image_root,
        test_label_map=test_label_map,
        test_label_text_map=test_label_text_map,
        test_label_ids_map=test_label_ids_map,
        split_prefix=args.test_split_prefix,
        strip_split_prefix=bool(args.test_strip_split_prefix),
        bbox_round_decimals=int(args.test_bbox_round_decimals),
    )
    start = max(0, int(script_args.start_index))
    limit = max(1, int(script_args.num_samples))
    groups = test_groups[start : start + limit]
    if not groups:
        raise RuntimeError(f"No test groups available in range start={start} num_samples={limit}.")

    dataset = GazeTestDataset(
        groups=groups,
        prompt_template=args.prompt_template,
        prompt_text=str(args.prompt_text or ""),
        id2label=id2label,
        vocab2id=vocab2id,
        vocab2id_lower=vocab2id_lower,
        num_classes=int(num_classes),
        visual_prompting=bool(args.visual_prompting),
        image_cache_size=max(0, int(getattr(args, "image_cache_size", 0))),
        coord_bins=coord_bins,
    )
    collator = QwenTestCollator(
        processor=processor,
        max_text_length=int(args.max_text_length),
        scene_size=_scene_size,
    )

    load_dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and load_dtype in {torch.bfloat16, torch.float16}:
        load_dtype = torch.float32
    amp_dtype = to_autocast_dtype(load_dtype)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if load_dtype != "auto":
        model_kwargs["dtype"] = load_dtype

    base_qwen = init_base_model(model_path=model_path, model_kwargs=model_kwargs)
    base_qwen.resize_token_embeddings(len(processor.tokenizer))
    adapter_dir = (checkpoint_dir / "lora_adapter") if checkpoint_dir is not None else None
    if adapter_dir is not None and adapter_dir.exists():
        qwen_model = PeftModel.from_pretrained(base_qwen, model_id=str(adapter_dir), is_trainable=False)
    else:
        qwen_model = base_qwen
    model = QwenTextGenerationModel(qwen_model=qwen_model).to(device)
    model.eval()

    batch_size = max(1, min(int(args.test_batch_size), len(dataset)))
    num_workers = max(0, int(args.num_workers))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )

    samples = collect_generation_samples(
        model=model,
        loader=loader,
        device=device,
        amp_dtype=amp_dtype,
        processor=collator.processor,
        num_classes=int(num_classes),
        coord_bins=coord_bins,
        show_tqdm=bool(args.show_tqdm),
        desc="Visualize",
        max_new_tokens=int(args.generation_max_new_tokens),
        num_beams=int(args.generation_num_beams),
        repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
        no_repeat_ngram_size=int(getattr(args, "no_repeat_ngram_size", 0)),
        stop_at_object_end=bool(getattr(args, "generation_stop_at_object_end", True)),
        max_samples=len(dataset),
        constrained_decoding=bool(getattr(args, "constrained_decoding", False)),
        constrained_target_order=str(getattr(args, "constrained_target_order", "point_object")),
        constrained_temperature=float(getattr(args, "constrained_temperature", 1.0)),
    )

    if len(samples) != len(groups):
        raise RuntimeError(f"Expected {len(groups)} preview samples, got {len(samples)}.")

    annotated_images: list[Image.Image] = []
    summary_rows: list[dict[str, Any]] = []
    for idx, (group, sample) in enumerate(zip(groups, samples, strict=True)):
        panel = annotate_sample(
            group=group,
            sample=sample,
            index=start + idx,
            thumb_width=int(script_args.thumb_width),
        )
        out_path = out_dir / f"sample_{start + idx:03d}.png"
        panel.save(out_path)
        annotated_images.append(panel)
        summary_rows.append(
            {
                "sample_index": start + idx,
                "image_path": str(group.image_path),
                "bbox_px": [float(v) for v in group.bbox_px],
                "gt_points": [[float(x), float(y)] for x, y in group.gt_points],
                **sample,
                "output_image": str(out_path),
            }
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    sheet = build_contact_sheet(annotated_images, cols=2)
    if sheet is not None:
        sheet.save(out_dir / "contact_sheet.jpg", quality=92)

    print(f"[DONE] saved {len(annotated_images)} annotated samples to: {out_dir}")
    print(f"[DONE] summary: {out_dir / 'summary.json'}")
    if sheet is not None:
        print(f"[DONE] contact sheet: {out_dir / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
