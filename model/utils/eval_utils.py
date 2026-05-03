from __future__ import annotations

import math
import re
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from .common import to_device
from .gaze_tokens import (
    ANSWER_END,
    parse_structured_output_text,
)
from .loss_utils import compute_answer_loss


# ---------------------------------------------------------------------------
# Stopping criteria — stop when <gaze_obj_end> is generated
# ---------------------------------------------------------------------------

class _GazeObjEndStoppingCriteria(StoppingCriteria):
    """Stop generation once <gaze_obj_end> token appears in generated portion."""

    def __init__(self, prompt_len: int, obj_end_id: int) -> None:
        self.prompt_len = int(prompt_len)
        self.obj_end_id = int(obj_end_id)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        generated = input_ids[:, self.prompt_len :]
        return bool((generated == self.obj_end_id).any(dim=1).all().item())


def make_gaze_obj_end_stopping_criteria(
    processor: Any,
    prompt_len: int,
    stop_at_object_end: bool = True,
) -> StoppingCriteriaList | None:
    if not stop_at_object_end:
        return None
    try:
        tok = getattr(processor, "tokenizer", None) or processor
        ids = tok.encode(ANSWER_END, add_special_tokens=False)
        if not ids:
            return None
        obj_end_id = int(ids[-1])
        criteria = _GazeObjEndStoppingCriteria(
            prompt_len=int(prompt_len),
            obj_end_id=obj_end_id,
        )
        return StoppingCriteriaList([criteria])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def decode_generated(
    processor: Any,
    generated_ids: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_return_sequences: int = 1,
) -> list[str]:
    out: list[str] = []
    tok = getattr(processor, "tokenizer", None)
    nrs = max(1, int(num_return_sequences))
    bsz = int(input_ids.shape[0]) if torch.is_tensor(input_ids) and input_ids.dim() >= 2 else 0
    for i in range(generated_ids.shape[0]):
        src_i = (i // nrs) if bsz > 0 else i
        src_i = min(max(0, int(src_i)), bsz - 1) if bsz > 0 else src_i
        start = int(input_ids.shape[1])
        new_tokens = generated_ids[i, start:]
        txt = (
            tok.decode(new_tokens, skip_special_tokens=False)
            if tok is not None
            else str(new_tokens.tolist())
        )
        # Keep parser-visible schema markers intact. The parser accepts <|im_end|>
        # as an optional trailing EOS, and gaze markers are part of the output schema.
        # Strip other Qwen chat markers (e.g. <|endoftext|>).
        txt = re.sub(r"<\|(?!im_start\||im_end\||gaze_)[^>]+?\|>", "", str(txt)).strip()
        out.append(str(txt))
    return out


def l2_stats(pred_xy: tuple[float, float], gt_points: torch.Tensor) -> tuple[float, float] | None:
    if (not torch.is_tensor(gt_points)) or gt_points.numel() < 2:
        return None
    if int(gt_points.numel()) % 2 != 0:
        return None
    pts = gt_points.to(dtype=torch.float32).view(-1, 2)
    if int(pts.shape[0]) <= 0:
        return None
    px, py = float(pred_xy[0]), float(pred_xy[1])
    dists = [
        math.sqrt((px - float(pts[j, 0].item())) ** 2 + (py - float(pts[j, 1].item())) ** 2)
        for j in range(int(pts.shape[0]))
    ]
    return float(sum(dists) / len(dists)), float(min(dists))


def valid_label_ids(label_ids: torch.Tensor | list[int] | tuple[int, ...] | None) -> list[int]:
    if label_ids is None:
        return []
    if torch.is_tensor(label_ids):
        vals = label_ids.detach().cpu().flatten().tolist()
    else:
        vals = list(label_ids)
    out: list[int] = []
    for v in vals:
        iv = int(v)
        if iv >= 0 and iv not in out:
            out.append(iv)
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    loss_weights: dict[str, float] | None = None,
    show_tqdm: bool = True,
    desc: str = "Eval",
) -> dict[str, float]:
    model.eval()
    if loss_weights is None:
        loss_weights = {"point": 1.0, "object": 1.0, "format": 0.25}

    loss_sum = 0.0
    pt_sum = 0.0
    obj_sum = 0.0
    fmt_sum = 0.0
    count = 0

    with torch.no_grad():
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not show_tqdm)
        for batch in it:
            joint = to_device(batch["joint_inputs"], device=device)
            labels = batch["labels"].to(device)
            if torch.all(labels.eq(-100)):
                continue
            bsz = int(labels.shape[0])

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                out = model(joint_inputs=joint, use_cache=False)

            losses = compute_answer_loss(
                logits=out.get("logits", None),
                labels=labels,
                loss_mask_point=batch.get("loss_mask_point", None),
                loss_mask_object=batch.get("loss_mask_object", None),
                loss_mask_format=batch.get("loss_mask_format", None),
                weight_point=float(loss_weights.get("point", 1.0)),
                weight_object=float(loss_weights.get("object", 1.0)),
                weight_format=float(loss_weights.get("format", 0.25)),
            )
            loss_sum += float(losses["loss"].detach().item()) * float(bsz)
            pt_sum += float(losses["loss_point"].detach().item()) * float(bsz)
            obj_sum += float(losses["loss_object"].detach().item()) * float(bsz)
            fmt_sum += float(losses["loss_format"].detach().item()) * float(bsz)
            count += bsz
            if show_tqdm:
                it.set_postfix(loss=f"{(loss_sum / max(count, 1)):.4f}")

    if count <= 0:
        return {"loss": 0.0, "loss_point": 0.0, "loss_object": 0.0, "loss_format": 0.0}
    d = float(count)
    return {
        "loss": float(loss_sum / d),
        "loss_point": float(pt_sum / d),
        "loss_object": float(obj_sum / d),
        "loss_format": float(fmt_sum / d),
    }


def run_test_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    processor: Any,
    num_classes: int,
    coord_bins: int = 1000,
    show_tqdm: bool = True,
    desc: str = "Test",
    max_new_tokens: int = 8,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    stop_at_object_end: bool = True,
) -> dict[str, float]:
    model.eval()

    total = 0           # all samples
    valid_total = 0     # samples with target_text_valid > 0
    format_valid_total = 0  # samples with target_format_valid > 0 (denominator for FormatValid/ExtraTextRate)
    point_bin_den = 0   # samples with target_point_valid > 0 (denominator for PointBinExact)
    format_valid_count = 0
    extra_text_count = 0
    avg_l2_sum = 0.0
    min_l2_sum = 0.0
    l2_den = 0
    point_bin_exact = 0
    object_acc = 0
    multi_acc_at_1 = 0
    joint_exact = 0
    obj_den = 0
    multi_obj_den = 0
    beam_k = max(1, int(num_beams))

    with torch.no_grad():
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not show_tqdm)
        for batch in it:
            joint = to_device(batch["joint_inputs"], device=device)
            target_texts = [str(x) for x in batch.get("target_text", [])]
            target_valid = batch.get("target_text_valid", None)
            target_point_valid = batch.get("target_point_valid", None)
            target_object_valid = batch.get("target_object_valid", None)
            target_format_valid = batch.get("target_format_valid", None)
            target_point_bin = batch.get("target_point_bin", None)
            target_object_id = batch.get("target_object_id", None)
            target_label_ids = batch.get("target_label_ids", None)
            gt_points = batch.get("gt_points", None)
            if target_valid is None:
                target_valid = torch.ones((len(target_texts),), dtype=torch.float32)
            target_valid = target_valid.to(dtype=torch.float32)
            if target_point_valid is None:
                target_point_valid = target_valid
            if target_object_valid is None:
                target_object_valid = target_valid
            if target_format_valid is None:
                target_format_valid = target_valid
            target_point_valid = target_point_valid.to(dtype=torch.float32)
            target_object_valid = target_object_valid.to(dtype=torch.float32)
            target_format_valid = target_format_valid.to(dtype=torch.float32)

            prompt_len = int(joint["input_ids"].shape[1])
            stopping = make_gaze_obj_end_stopping_criteria(
                processor, prompt_len, stop_at_object_end=bool(stop_at_object_end)
            )

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                generate_kwargs: dict[str, Any] = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=beam_k,
                    repetition_penalty=max(float(repetition_penalty), 1.0),
                    no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                )
                if stopping is not None:
                    generate_kwargs["stopping_criteria"] = stopping
                generated_ids = model.generate(joint_inputs=joint, **generate_kwargs)

            generated_ids_cpu = generated_ids.detach().cpu()
            input_ids_cpu = joint["input_ids"].detach().cpu()
            attn_cpu = joint.get("attention_mask", None)
            if torch.is_tensor(attn_cpu):
                attn_cpu = attn_cpu.detach().cpu()

            preds = decode_generated(
                processor=processor,
                generated_ids=generated_ids_cpu,
                input_ids=input_ids_cpu,
                attention_mask=attn_cpu,
                num_return_sequences=1,
            )
            bsz = len(target_texts)
            preds = preds[:bsz]

            for i in range(bsz):
                total += 1
                pred = preds[i] if i < len(preds) else ""
                parsed = parse_structured_output_text(pred, int(num_classes), coord_bins=int(coord_bins))
                is_text_valid = i < int(target_valid.numel()) and float(target_valid[i].item()) > 0.0
                is_point_valid = i < int(target_point_valid.numel()) and float(target_point_valid[i].item()) > 0.0
                is_object_valid = i < int(target_object_valid.numel()) and float(target_object_valid[i].item()) > 0.0
                is_format_valid_gt = i < int(target_format_valid.numel()) and float(target_format_valid[i].item()) > 0.0
                if is_text_valid:
                    valid_total += 1
                if is_format_valid_gt:
                    format_valid_total += 1
                if is_point_valid:
                    point_bin_den += 1

                # FormatValid / ExtraTextRate: denominator = format_valid_total
                if is_format_valid_gt:
                    if parsed["has_extra_text"]:
                        extra_text_count += 1
                    if parsed["valid_format"]:
                        format_valid_count += 1

                # point L2: valid-GT samples only (point GT is meaningful)
                if is_point_valid and parsed["point_xy"] is not None and isinstance(gt_points, list) and i < len(gt_points):
                    stats = l2_stats(parsed["point_xy"], gt_points[i])
                    if stats is not None:
                        avg_l2_sum += float(stats[0])
                        min_l2_sum += float(stats[1])
                        l2_den += 1

                # point bin exact match: denominator = point_bin_den (all is_point_valid samples,
                # format failures count as misses — consistent with ObjectAcc gating on obj_den)
                if (
                    is_point_valid
                    and parsed["point_bins"] is not None
                    and torch.is_tensor(target_point_bin)
                    and i < int(target_point_bin.shape[0])
                ):
                    gt_bx = int(target_point_bin[i, 0].item())
                    gt_by = int(target_point_bin[i, 1].item())
                    px, py = parsed["point_bins"]
                    if int(px) == gt_bx and int(py) == gt_by:
                        point_bin_exact += 1

                # object accuracy: valid-GT samples only
                if is_object_valid and torch.is_tensor(target_object_id) and i < int(target_object_id.shape[0]):
                    gt_obj = int(target_object_id[i].item())
                    obj_den += 1
                    if parsed["object_id"] is not None and int(parsed["object_id"]) == gt_obj:
                        object_acc += 1

                if is_object_valid and torch.is_tensor(target_label_ids) and i < int(target_label_ids.shape[0]):
                    gt_obj_ids = valid_label_ids(target_label_ids[i])
                    if gt_obj_ids:
                        multi_obj_den += 1
                        if parsed["object_id"] is not None and int(parsed["object_id"]) in gt_obj_ids:
                            multi_acc_at_1 += 1

                # joint exact: valid-GT samples only
                if (
                    is_point_valid
                    and is_object_valid
                    and parsed["valid_format"]
                    and parsed["point_bins"] is not None
                    and parsed["object_id"] is not None
                    and torch.is_tensor(target_point_bin)
                    and torch.is_tensor(target_object_id)
                    and i < int(target_point_bin.shape[0])
                    and i < int(target_object_id.shape[0])
                ):
                    gt_bx = int(target_point_bin[i, 0].item())
                    gt_by = int(target_point_bin[i, 1].item())
                    gt_obj = int(target_object_id[i].item())
                    px, py = parsed["point_bins"]
                    if int(px) == gt_bx and int(py) == gt_by and int(parsed["object_id"]) == gt_obj:
                        joint_exact += 1

            if show_tqdm and l2_den > 0:
                it.set_postfix(
                    l2=f"{(avg_l2_sum / max(l2_den, 1)):.4f}",
                    fmt=f"{(format_valid_count / max(format_valid_total, 1)):.3f}",
                )

    _L2_SENTINEL = math.sqrt(2.0)  # max possible L2 for normalized coords (0,0)→(1,1)
    if total <= 0:
        return {
            "FormatValid": 0.0,
            "Avg L2": _L2_SENTINEL,
            "Min L2": _L2_SENTINEL,
            "PointBinExact": 0.0,
            "ObjectAcc": 0.0,
            "MultiAcc@1": 0.0,
            "JointExact": 0.0,
            "ExtraTextRate": 0.0,
            "PointL2ValidFrac": 0.0,
            "num_samples": 0.0,
            "num_valid_samples": 0.0,
        }

    return {
        "FormatValid": float(format_valid_count / max(format_valid_total, 1)),
        "Avg L2": float(avg_l2_sum / l2_den) if l2_den > 0 else _L2_SENTINEL,
        "Min L2": float(min_l2_sum / l2_den) if l2_den > 0 else _L2_SENTINEL,
        "PointBinExact": float(point_bin_exact / max(point_bin_den, 1)),
        "ObjectAcc": float(object_acc / max(obj_den, 1)),
        "MultiAcc@1": float(multi_acc_at_1 / max(multi_obj_den, 1)),
        "JointExact": float(joint_exact / max(obj_den, 1)),
        "ExtraTextRate": float(extra_text_count / max(format_valid_total, 1)),
        "PointL2ValidFrac": float(l2_den / max(point_bin_den, 1)),
        "num_samples": float(total),
        "num_valid_samples": float(valid_total),
    }


def collect_generation_samples(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    processor: Any,
    num_classes: int,
    coord_bins: int = 1000,
    show_tqdm: bool = True,
    desc: str = "GenerationPreview",
    max_new_tokens: int = 8,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    max_samples: int = 8,
    stop_at_object_end: bool = True,
) -> list[dict[str, Any]]:
    limit = max(0, int(max_samples))
    if limit <= 0:
        return []

    model.eval()
    previews: list[dict[str, Any]] = []
    beam_k = max(1, int(num_beams))

    with torch.no_grad():
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not show_tqdm)
        for batch in it:
            joint = to_device(batch["joint_inputs"], device=device)
            target_texts = [str(x) for x in batch.get("target_text", [])]
            target_valid = batch.get("target_text_valid", None)
            target_label = batch.get("target_label", None)
            target_object_id = batch.get("target_object_id", None)
            target_point_bin = batch.get("target_point_bin", None)
            gt_points = batch.get("gt_points", None)
            prompt_texts = [str(x) for x in batch.get("text_input", [])]
            target_label_texts = [str(x) for x in batch.get("target_label_text", [])]
            image_rels = [str(x) for x in batch.get("image_rel", [])]
            if target_valid is None:
                target_valid = torch.ones((len(target_texts),), dtype=torch.float32)
            target_valid = target_valid.to(dtype=torch.float32)

            prompt_len = int(joint["input_ids"].shape[1])
            stopping = make_gaze_obj_end_stopping_criteria(
                processor, prompt_len, stop_at_object_end=bool(stop_at_object_end)
            )
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                generate_kwargs: dict[str, Any] = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=beam_k,
                    repetition_penalty=max(float(repetition_penalty), 1.0),
                    no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                )
                if stopping is not None:
                    generate_kwargs["stopping_criteria"] = stopping
                generated_ids = model.generate(joint_inputs=joint, **generate_kwargs)

            generated_ids_cpu = generated_ids.detach().cpu()
            input_ids_cpu = joint["input_ids"].detach().cpu()
            attn_cpu = joint.get("attention_mask", None)
            if torch.is_tensor(attn_cpu):
                attn_cpu = attn_cpu.detach().cpu()

            preds = decode_generated(
                processor=processor,
                generated_ids=generated_ids_cpu,
                input_ids=input_ids_cpu,
                attention_mask=attn_cpu,
                num_return_sequences=1,
            )
            bsz = len(target_texts)
            preds = preds[:bsz]

            for i in range(bsz):
                pred = preds[i] if i < len(preds) else ""
                parsed = parse_structured_output_text(pred, int(num_classes), coord_bins=int(coord_bins))

                serialized_gt_points: list[list[float]] = []
                stats = None
                if isinstance(gt_points, list) and i < len(gt_points):
                    if parsed["point_xy"] is not None:
                        stats = l2_stats(parsed["point_xy"], gt_points[i])
                    if torch.is_tensor(gt_points[i]) and gt_points[i].numel() >= 2:
                        pts = gt_points[i].detach().cpu().to(dtype=torch.float32).view(-1, 2)
                        serialized_gt_points = [
                            [float(pts[j, 0].item()), float(pts[j, 1].item())]
                            for j in range(int(pts.shape[0]))
                        ]

                gt_obj_id = -1
                if torch.is_tensor(target_object_id) and i < int(target_object_id.shape[0]):
                    gt_obj_id = int(target_object_id[i].item())
                gt_label_id = -1
                if torch.is_tensor(target_label) and i < int(target_label.shape[0]):
                    gt_label_id = int(target_label[i].item())
                gt_pt_bins: list[int] = []
                if torch.is_tensor(target_point_bin) and i < int(target_point_bin.shape[0]):
                    gt_pt_bins = [int(target_point_bin[i, 0].item()), int(target_point_bin[i, 1].item())]

                target_is_valid = (
                    bool(float(target_valid[i].item()) > 0.0)
                    if i < int(target_valid.numel())
                    else False
                )
                previews.append(
                    {
                        "sample_index": int(len(previews)),
                        "image_rel": image_rels[i] if i < len(image_rels) else "",
                        "prompt_text": prompt_texts[i] if i < len(prompt_texts) else "",
                        "target_text": target_texts[i] if i < len(target_texts) else "",
                        "target_text_valid": target_is_valid,
                        "generated_text": pred,
                        "parsed": {
                            "valid_format": bool(parsed["valid_format"]),
                            "has_extra_text": bool(parsed["has_extra_text"]),
                            "point_bins": list(parsed["point_bins"]) if parsed["point_bins"] else None,
                            "point_xy": list(parsed["point_xy"]) if parsed["point_xy"] else None,
                            "object_id": int(parsed["object_id"]) if parsed["object_id"] is not None else None,
                        },
                        "avg_l2": float(stats[0]) if stats is not None else None,
                        "min_l2": float(stats[1]) if stats is not None else None,
                        "gt_points": serialized_gt_points,
                        "gt_point_bins": gt_pt_bins,
                        "gt_object_id": gt_obj_id,
                        "gt_label_id": gt_label_id,
                        "target_label_text": (
                            target_label_texts[i] if i < len(target_label_texts) else ""
                        ),
                    }
                )
                if len(previews) >= limit:
                    return previews

    return previews


def print_generation_samples(samples: list[dict[str, Any]]) -> None:
    if not samples:
        print("[TEST] generation preview: no samples collected.")
        return
    print(f"[TEST] generation preview ({len(samples)} samples)")
    for sample in samples:
        idx = int(sample.get("sample_index", -1))
        image_rel = str(sample.get("image_rel", ""))
        parsed = sample.get("parsed", {})
        print(f"[SAMPLE {idx}] image={image_rel}")
        print(f"  target   : {sample.get('target_text', '')}")
        print(f"  generated: {sample.get('generated_text', '')}")
        print(f"  fmt_valid: {parsed.get('valid_format', False)}  "
              f"extra_text: {parsed.get('has_extra_text', False)}")
        print(f"  pred_bins: {parsed.get('point_bins', None)}  "
              f"pred_xy: {parsed.get('point_xy', None)}")
        print(f"  pred_obj : {parsed.get('object_id', None)}  "
              f"gt_obj: {sample.get('gt_object_id', -1)}")
        print(f"  gt_points: {sample.get('gt_points', [])}  "
              f"avg_l2: {sample.get('avg_l2', None)}  "
              f"min_l2: {sample.get('min_l2', None)}")


def print_test_metrics_table(test_metrics: dict[str, float]) -> None:
    rows = [
        ("FormatValid", float(test_metrics.get("FormatValid", 0.0))),
        ("Avg L2", float(test_metrics.get("Avg L2", 0.0))),
        ("Min L2", float(test_metrics.get("Min L2", 0.0))),
        ("PointBinExact", float(test_metrics.get("PointBinExact", 0.0))),
        ("ObjectAcc", float(test_metrics.get("ObjectAcc", 0.0))),
        ("MultiAcc@1", float(test_metrics.get("MultiAcc@1", 0.0))),
        ("JointExact", float(test_metrics.get("JointExact", 0.0))),
        ("ExtraTextRate", float(test_metrics.get("ExtraTextRate", 0.0))),
        ("num_samples", float(test_metrics.get("num_samples", 0.0))),
    ]
    key_w = max(len(k) for k, _ in rows)
    val_w = 12
    line = "+" + "-" * (key_w + 2) + "+" + "-" * (val_w + 2) + "+"
    print("[TEST] metrics")
    print(line)
    print(f"| {'Metric'.ljust(key_w)} | {'Value'.rjust(val_w)} |")
    print(line)
    for k, v in rows:
        if k.startswith("num_"):
            print(f"| {k.ljust(key_w)} | {v:>{val_w}.0f} |")
        else:
            print(f"| {k.ljust(key_w)} | {v:>{val_w}.6f} |")
    print(line)
