"""Visualise student vs. teacher outputs on train samples.

Usage (from QWEN_GazeEstimation root):
    python scripts/sdft_sample_inference.py --config sdft.yaml \
        --checkpoint_dir checkpoints/SFT_direct_expectation_round/best

The script loads the checkpoint, builds train prompts exactly as the SDFT
trainer does, then runs constrained decoding for both the student prompt
(prompt_text_direct) and the teacher prompt (prompt_text_teacher + suffix
when reasoning text is available) and prints a side-by-side table.
"""
from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel

from model.model import QwenTextGenerationModel
from model.utils.checkpoint import load_added_token_rows, load_token_rows
from model.utils.common import parse_dtype, resolve_path, to_autocast_dtype, to_device
from model.utils.config_parser import build_parser, load_yaml_config
from model.utils.data_utils import (
    build_prompt,
    build_reasoning_index,
    load_label_map,
    load_label_text_map,
    load_records,
    load_vocab2id,
)
from model.utils.eval_utils import constrained_generate_structured
from model.utils.gaze_tokens import parse_structured_output_text
from model.utils.model_init import init_base_model, init_processor, resolve_model_source
from model.utils.processor_collate import build_infer_inputs
from model.utils.special_tokens import (
    format_loc_token,
    format_obj_token,
    register_gaze_special_tokens,
)
from model.modules.preprocess import resize_scene


@contextmanager
def _temporary_model_params(model: torch.nn.Module, named_tensors: dict[str, torch.Tensor]):
    if not named_tensors:
        yield
        return

    params = dict(model.named_parameters())
    missing = [name for name in named_tensors if name not in params]
    if missing:
        raise RuntimeError(f"EMA teacher parameter not found in model: {missing[0]}")

    originals: dict[str, torch.Tensor] = {}
    for name, value in named_tensors.items():
        param = params[name]
        originals[name] = param.data.detach().clone()
        param.data.copy_(value.to(device=param.device, dtype=param.dtype))
    try:
        yield
    finally:
        for name, value in originals.items():
            params[name].data.copy_(value)


def _load_ema_teacher_tensors(checkpoint_dir: Path, device: torch.device) -> dict[str, torch.Tensor]:
    path = checkpoint_dir / "ema_teacher.pt"
    if not path.exists():
        return {}
    payload = torch.load(path, map_location=device)
    ema_params = payload.get("ema_params", payload)
    if not isinstance(ema_params, dict):
        raise RuntimeError(f"Invalid EMA teacher checkpoint: {path}")
    return {str(k): v for k, v in ema_params.items()}


def _run(args: argparse.Namespace, num_samples: int = 4) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and load_dtype in {torch.bfloat16, torch.float16}:
        load_dtype = torch.float32
    amp_dtype = to_autocast_dtype(load_dtype)

    model_path = resolve_model_source(args.model_path)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    image_root = resolve_path(args.image_root)
    vocab2id_path = resolve_path(args.vocab2id)

    vocab2id, vocab2id_lower = load_vocab2id(vocab2id_path)
    num_classes = len(vocab2id)
    id2label = {int(v): k for k, v in vocab2id.items()}
    coord_bins = int(getattr(args, "coord_bins", 128))

    _resize_mode = str(getattr(args, "image_resize_mode", "native")).strip().lower()
    _fixed_resize = _resize_mode == "fixed"
    _scene_size: tuple[int, int] | None = (
        (int(args.scene_h), int(args.scene_w)) if _fixed_resize else None
    )
    _proc_min_pixels = None if _fixed_resize else int(getattr(args, "min_pixels", 12544))
    _proc_max_pixels = None if _fixed_resize else int(getattr(args, "max_pixels", 2007040))

    # Processor ------------------------------------------------------------------
    print(f"[INFO] loading processor from checkpoint: {checkpoint_dir}")
    processor = init_processor(
        model_path=model_path,
        checkpoint_dir=checkpoint_dir,
        min_pixels=_proc_min_pixels,
        max_pixels=_proc_max_pixels,
    )
    token_id_map = register_gaze_special_tokens(
        tokenizer=processor.tokenizer,
        num_classes=num_classes,
        coord_bins=coord_bins,
    )
    new_vocab_size = len(processor.tokenizer)
    print(f"[INFO] vocab size: {new_vocab_size}")

    # Model ----------------------------------------------------------------------
    model_kwargs: dict = {
        "trust_remote_code": True,
        "attn_implementation": getattr(args, "attn_implementation", "eager"),
    }
    if load_dtype != "auto":
        model_kwargs["dtype"] = load_dtype

    print(f"[INFO] loading base model: {model_path}")
    base_qwen = init_base_model(model_path=model_path, model_kwargs=model_kwargs)
    base_qwen.resize_token_embeddings(new_vocab_size)

    adapter_dir = checkpoint_dir / "lora_adapter"
    if adapter_dir.exists():
        qwen_model = PeftModel.from_pretrained(base_qwen, str(adapter_dir), is_trainable=False)
        print(f"[INFO] loaded LoRA adapter: {adapter_dir}")
    else:
        qwen_model = base_qwen
        print(f"[WARN] no LoRA adapter at {adapter_dir}; using base model")

    model = QwenTextGenerationModel(qwen_model=qwen_model).to(device)
    _tmp = model
    rows_ok = load_token_rows(ckpt_dir=checkpoint_dir, model=_tmp, device=device)
    if not rows_ok:
        rows_ok = load_added_token_rows(ckpt_dir=checkpoint_dir, model=_tmp, device=device)
    if rows_ok:
        print(f"[INFO] gaze token embeddings restored from checkpoint")
    model.eval()

    teacher_update = str(getattr(args, "teacher_update", "fixed")).strip().lower()
    ema_teacher_tensors: dict[str, torch.Tensor] = {}
    if teacher_update == "ema":
        ema_teacher_tensors = _load_ema_teacher_tensors(checkpoint_dir, device=device)
        if ema_teacher_tensors:
            print(f"[INFO] loaded EMA teacher params: {len(ema_teacher_tensors)} tensors")
        else:
            print(
                f"[WARN] teacher_update=ema but no {checkpoint_dir / 'ema_teacher.pt'} found; "
                "teacher generation will use student weights with the teacher prompt."
            )

    # Train data -----------------------------------------------------------------
    train_ann = resolve_path(args.train_ann)
    train_labels = resolve_path(args.train_labels)

    train_label_map, _ = load_label_map(
        train_labels, vocab2id=vocab2id, vocab2id_lower=vocab2id_lower,
        text_key="gaze_pseudo_label", use_embed_fallback=False,
    )
    train_label_text_map, _ = load_label_text_map(train_labels, text_key="gaze_pseudo_label")
    train_records = load_records(
        annotation_file=train_ann,
        image_root=image_root,
        label_map=train_label_map,
        label_text_map=train_label_text_map,
        split_prefix=str(getattr(args, "split_prefix", "train/")),
        strip_split_prefix=bool(getattr(args, "strip_split_prefix", True)),
        max_samples=num_samples * 10,  # load extra in case of filter
    )
    print(f"[INFO] loaded {len(train_records)} train records")

    # Reasoning index for teacher suffix -----------------------------------------
    reasoning_dir_raw = str(getattr(args, "train_reasoning_dir", "") or "").strip()
    reasoning_index = None
    if reasoning_dir_raw:
        rdir = resolve_path(reasoning_dir_raw)
        if rdir.exists():
            reasoning_index = build_reasoning_index(rdir)
            print(f"[INFO] reasoning index: {len(reasoning_index)} entries")

    # Resolve teacher suffix struct-token variables ------------------------------
    _coord_n = int(coord_bins)
    _loc_w = max(3, len(str(_coord_n - 1)))
    _obj_max = max(0, num_classes - 1)
    _obj_w = max(3, len(str(_obj_max)))
    raw_teacher_suffix = str(getattr(args, "teacher_suffix", "")).strip()
    if raw_teacher_suffix:
        teacher_suffix = raw_teacher_suffix.format(
            loc_tok_min=format_loc_token(0, _loc_w),
            loc_tok_max=format_loc_token(_coord_n - 1, _loc_w),
            obj_tok_min=format_obj_token(0, _obj_w),
            obj_tok_max=format_obj_token(_obj_max, _obj_w),
            reasoning_text="{reasoning_text}",
            object_text="{object_text}",
        )
    else:
        teacher_suffix = ""

    # Prompt templates -----------------------------------------------------------
    _prompt_direct = str(getattr(args, "prompt_text_direct", "") or "")
    _prompt_teacher = str(getattr(args, "prompt_text_teacher", "") or "")
    _prompt_template = str(getattr(args, "prompt_template", "") or "")

    # Collect samples (filter invalid if needed) ---------------------------------
    from model.datasets import _reasoning_key
    from model.utils.data_utils import load_reasoning_record

    samples_collected = []
    for rec in train_records:
        if len(samples_collected) >= num_samples:
            break

        # Resize / load image
        from PIL import Image
        with Image.open(rec.image_path) as img:
            scene = img.convert("RGB")

        if _fixed_resize and _scene_size is not None:
            scene = resize_scene(scene_image=[scene], scene_size=_scene_size)[0]

        w, h = scene.size
        from model.utils.data_utils import sanitize_bbox_pixels
        x1, y1, x2, y2 = sanitize_bbox_pixels(rec.bbox_px, width=w, height=h)
        if bool(args.visual_prompting):
            from model.datasets import draw_head_bbox_prompt
            scene = draw_head_bbox_prompt(scene, x1=x1, y1=y1, x2=x2, y2=y2)
        bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)

        prompt_student = build_prompt(
            bbox_norm, _prompt_template, _prompt_direct,
            num_classes=num_classes, coord_bins=coord_bins,
        )
        if _prompt_teacher:
            prompt_teacher_base = build_prompt(
                bbox_norm, _prompt_template, _prompt_teacher,
                num_classes=num_classes, coord_bins=coord_bins,
            )
        else:
            prompt_teacher_base = prompt_student

        # Reasoning for teacher (only available in train reasoning dir)
        rtext, otext = "", ""
        if reasoning_index is not None:
            key = _reasoning_key(str(rec.image_rel), int(rec.sample_id))
            rpath = reasoning_index.get(key)
            if rpath is not None:
                rdata = load_reasoning_record(rpath)
                rtext = rdata.get("reasoning_text") or ""
                otext = rdata.get("object_text") or ""

        if rtext or otext:
            prompt_teacher_full = prompt_teacher_base + teacher_suffix.format(
                reasoning_text=rtext, object_text=otext
            )
        else:
            prompt_teacher_full = prompt_teacher_base

        # GT target text
        from model.utils.gaze_tokens import build_structured_target_text
        gt_obj_id = int(rec.label_id) if int(rec.label_id) >= 0 else None
        gt_text = build_structured_target_text(
            point_x=float(rec.gaze_x),
            point_y=float(rec.gaze_y),
            obj_id=gt_obj_id,
            num_classes=num_classes,
            coord_bins=coord_bins,
            target_order="point_object",
        )

        samples_collected.append({
            "rec": rec,
            "scene": scene,
            "prompt_student": prompt_student,
            "prompt_teacher": prompt_teacher_full,
            "has_reasoning": bool(rtext or otext),
            "reasoning_text": rtext,
            "object_text": otext,
            "gt_text": gt_text,
            "gt_gaze": (rec.gaze_x, rec.gaze_y),
            "gt_obj_id": gt_obj_id,
            "gt_label": str(id2label.get(int(rec.label_id), "?")) if int(rec.label_id) >= 0 else "?",
        })

    print(f"\n[INFO] running inference on {len(samples_collected)} samples...\n")

    # Inference ------------------------------------------------------------------
    _constrained_loc = str(getattr(args, "constrained_loc_decoding", "round_expectation"))
    student_l2_values: list[float] = []
    teacher_l2_values: list[float] = []

    with torch.no_grad():
        for i, s in enumerate(samples_collected):
            # Student
            student_infer = build_infer_inputs(
                processor=processor,
                scene_images=[s["scene"]],
                text_inputs=[s["prompt_student"]],
                max_text_length=int(getattr(args, "max_text_length", 512)),
            )
            student_infer_dev = to_device(student_infer, device)
            student_texts = constrained_generate_structured(
                model=model,
                joint=student_infer_dev,
                processor=processor,
                num_classes=num_classes,
                coord_bins=coord_bins,
                amp_dtype=amp_dtype,
                target_order="point_object",
                loc_selection_strategy=_constrained_loc,
            )
            student_out = student_texts[0] if student_texts else ""

            # Teacher
            teacher_infer = build_infer_inputs(
                processor=processor,
                scene_images=[s["scene"]],
                text_inputs=[s["prompt_teacher"]],
                max_text_length=int(getattr(args, "max_text_length", 512)),
            )
            teacher_infer_dev = to_device(teacher_infer, device)
            teacher_weight_ctx = (
                _temporary_model_params(model, ema_teacher_tensors)
                if ema_teacher_tensors else nullcontext()
            )
            with teacher_weight_ctx:
                teacher_texts = constrained_generate_structured(
                    model=model,
                    joint=teacher_infer_dev,
                    processor=processor,
                    num_classes=num_classes,
                    coord_bins=coord_bins,
                    amp_dtype=amp_dtype,
                    target_order="point_object",
                    loc_selection_strategy=_constrained_loc,
                )
            teacher_out = teacher_texts[0] if teacher_texts else ""

            # Parse
            s_parsed = parse_structured_output_text(student_out, num_classes, coord_bins=coord_bins)
            t_parsed = parse_structured_output_text(teacher_out, num_classes, coord_bins=coord_bins)

            rec = s["rec"]
            print("=" * 72)
            print(f"[SAMPLE {i}]  image : {rec.image_rel}")
            print(f"             sample_id : {rec.sample_id}")
            print()
            print(f"  GT gaze   : ({s['gt_gaze'][0]:.4f}, {s['gt_gaze'][1]:.4f})")
            print(f"  GT obj    : id={s['gt_obj_id']}  label='{s['gt_label']}'")
            print(f"  GT text   : {s['gt_text']}")
            print()

            s_xy = s_parsed.get("point_xy")
            s_oid = s_parsed.get("object_id")
            t_xy = t_parsed.get("point_xy")
            t_oid = t_parsed.get("object_id")

            s_label = str(id2label.get(int(s_oid), "?")) if s_oid is not None else "?"
            t_label = str(id2label.get(int(t_oid), "?")) if t_oid is not None else "?"

            import math
            def _l2(pred_xy, gt_xy):
                if pred_xy is None:
                    return None
                return math.sqrt((pred_xy[0] - gt_xy[0])**2 + (pred_xy[1] - gt_xy[1])**2)

            s_l2 = _l2(s_xy, s["gt_gaze"])
            t_l2 = _l2(t_xy, s["gt_gaze"])
            if s_l2 is not None:
                student_l2_values.append(float(s_l2))
            if t_l2 is not None:
                teacher_l2_values.append(float(t_l2))

            print(f"  STUDENT output : {student_out}")
            print(f"    pred_xy  : {s_xy}   l2={s_l2:.4f}" if s_l2 is not None else f"    pred_xy  : {s_xy}")
            print(f"    pred_obj : id={s_oid}  label='{s_label}'  fmt_valid={s_parsed.get('valid_format')}")
            print()
            print(f"  TEACHER output : {teacher_out}")
            print(f"    pred_xy  : {t_xy}   l2={t_l2:.4f}" if t_l2 is not None else f"    pred_xy  : {t_xy}")
            print(f"    pred_obj : id={t_oid}  label='{t_label}'  fmt_valid={t_parsed.get('valid_format')}")
            if s["has_reasoning"]:
                print(f"\n  [Reasoning available]")
                print(f"    object_text   : {s['object_text']}")
                print(f"    reasoning_text: {s['reasoning_text'][:120]}...")
            else:
                print(f"\n  [No reasoning data for train sample — teacher uses base prompt only]")
            print()

    def _print_l2_stats(name: str, values: list[float], total: int) -> None:
        if not values:
            print(f"  {name:<7} AvgL2=N/A  valid=0/{total}")
            return
        values_t = torch.tensor(values, dtype=torch.float32)
        std = values_t.std(unbiased=False).item() if len(values) > 1 else 0.0
        print(
            f"  {name:<7} AvgL2={values_t.mean().item():.4f}  "
            f"Std={std:.4f}  Min={values_t.min().item():.4f}  "
            f"Max={values_t.max().item():.4f}  valid={len(values)}/{total}"
        )

    print("=" * 72)
    print("[SUMMARY]")
    _print_l2_stats("Student", student_l2_values, len(samples_collected))
    _print_l2_stats("Teacher", teacher_l2_values, len(samples_collected))


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="sdft.yaml")
    config_parser.add_argument("--num_samples", type=int, default=None)
    cfg_args, remaining = config_parser.parse_known_args()

    config_path = resolve_path(cfg_args.config)
    config_defaults = load_yaml_config(config_path)
    config_defaults["config"] = str(cfg_args.config)

    parser = build_parser(defaults=config_defaults)
    parser.add_argument("--num_samples", type=int, default=cfg_args.num_samples or 4)
    args = parser.parse_args(remaining)

    _run(args, num_samples=int(args.num_samples))


if __name__ == "__main__":
    main()
