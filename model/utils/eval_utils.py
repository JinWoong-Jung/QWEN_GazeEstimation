from __future__ import annotations

import math
import re
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .loss_utils import compute_structured_losses


def _move_joint_inputs_to_device(joint_inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in joint_inputs.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _decode_generated_text(
    processor: Any,
    generated_ids: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_return_sequences: int = 1,
) -> list[str]:
    preds: list[str] = []
    tok = getattr(processor, "tokenizer", None)
    nrs = max(1, int(num_return_sequences))
    bsz = int(input_ids.shape[0]) if torch.is_tensor(input_ids) and input_ids.dim() >= 2 else 0
    for i in range(generated_ids.shape[0]):
        src_i = (i // nrs) if bsz > 0 else i
        if bsz > 0:
            src_i = min(max(0, int(src_i)), bsz - 1)
        if attention_mask is not None and torch.is_tensor(attention_mask) and attention_mask.dim() >= 2 and bsz > 0:
            start = int(attention_mask[src_i].sum().item())
        else:
            start = int(input_ids.shape[1])
        new_tokens = generated_ids[i, start:]
        if tok is not None:
            txt = tok.decode(new_tokens, skip_special_tokens=True)
        else:
            txt = str(new_tokens.tolist())
        preds.append(str(txt).strip())
    return preds


def _parse_object_id(text: str) -> int | None:
    m = re.search(r"(?im)^\s*objectid\s*:\s*(-?\d+)\s*$", str(text or ""))
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_point_xy(text: str) -> tuple[float, float] | None:
    num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    m = re.search(rf"(?im)^\s*point\s*:\s*({num})\s*[,\s]+\s*({num})\b", str(text or ""))
    if m is None:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return None


def _avg_min_l2_to_gt_points(
    pred_xy: tuple[float, float],
    gt_points: torch.Tensor,
) -> tuple[float, float] | None:
    if (not torch.is_tensor(gt_points)) or gt_points.numel() < 2:
        return None
    if int(gt_points.numel()) % 2 != 0:
        return None
    pts = gt_points.to(dtype=torch.float32).view(-1, 2)
    if int(pts.shape[0]) <= 0:
        return None

    px = float(pred_xy[0])
    py = float(pred_xy[1])
    dists: list[float] = []
    for j in range(int(pts.shape[0])):
        dx = px - float(pts[j, 0].item())
        dy = py - float(pts[j, 1].item())
        dists.append(math.sqrt(dx * dx + dy * dy))
    if not dists:
        return None
    return float(sum(dists) / len(dists)), float(min(dists))


def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    loss_answer_weight: float = 0.0,
    loss_localization_weight: float = 1.0,
    loss_recognition_weight: float = 1.0,
    loss_use_lm_fallback: bool = False,
    show_tqdm: bool = True,
    desc: str = "Eval",
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    answer_loss_sum = 0.0
    loc_loss_sum = 0.0
    rec_loss_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        eval_iter = tqdm(
            loader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            disable=not show_tqdm,
        )
        for batch in eval_iter:
            joint_inputs = _move_joint_inputs_to_device(batch["joint_inputs"], device=device)
            labels = batch["labels"].to(device)
            bsz = int(labels.shape[0])
            if torch.all(labels.eq(-100)):
                continue

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(device.type == "cuda"),
            ):
                out = model(
                    joint_inputs=joint_inputs,
                    labels=labels,
                    use_cache=False,
                )
            loss = out.get("loss", None)
            if loss is None:
                raise RuntimeError("Model forward must return loss during evaluation.")
            structured = compute_structured_losses(
                logits=out.get("logits", None),
                labels=labels,
                loss_mask_answer=batch.get("loss_mask_answer", None),
                loss_mask_point=batch.get("loss_mask_point", None),
                loss_mask_objectid=batch.get("loss_mask_objectid", None),
                weight_answer=float(loss_answer_weight),
                weight_point=float(loss_localization_weight),
                weight_objectid=float(loss_recognition_weight),
                fallback_loss=(loss if bool(loss_use_lm_fallback) else None),
            )
            eval_loss = structured["loss"]

            loss_sum += float(eval_loss.detach().item()) * float(bsz)
            answer_loss_sum += float(structured["loss_answer"].detach().item()) * float(bsz)
            loc_loss_sum += float(structured["loss_localization"].detach().item()) * float(bsz)
            rec_loss_sum += float(structured["loss_recognition"].detach().item()) * float(bsz)
            sample_count += bsz
            if show_tqdm:
                eval_iter.set_postfix(loss=f"{(loss_sum / max(sample_count, 1)):.4f}")

    if sample_count <= 0:
        return {"loss": 0.0, "loss_answer": 0.0, "loss_localization": 0.0, "loss_recognition": 0.0}
    denom = float(sample_count)
    return {
        "loss": float(loss_sum / denom),
        "loss_answer": float(answer_loss_sum / denom),
        "loss_localization": float(loc_loss_sum / denom),
        "loss_recognition": float(rec_loss_sum / denom),
    }


def run_test_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    processor: Any,
    show_tqdm: bool = True,
    desc: str = "Test",
    max_new_tokens: int = 16,
    num_beams: int = 3,
) -> dict[str, float]:
    model.eval()

    total = 0
    valid_total = 0
    exact_match = 0
    contains_match = 0
    acc1_num = 0
    acc1_den = 0
    acc3_num = 0
    acc3_den = 0
    multiacc1_num = 0
    multiacc1_den = 0
    object_id_valid_num = 0
    avg_l2_sum = 0.0
    min_l2_sum = 0.0
    l2_den = 0
    beam_k = max(1, int(num_beams))

    with torch.no_grad():
        test_iter = tqdm(
            loader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            disable=not show_tqdm,
        )
        for batch in test_iter:
            joint_inputs = _move_joint_inputs_to_device(batch["joint_inputs"], device=device)
            target_texts = [str(x) for x in batch.get("target_text", [])]
            target_valid = batch.get("target_text_valid", None)
            target_label = batch.get("target_label", None)
            target_label_ids_batch = batch.get("target_label_ids", None)
            gt_points_batch = batch.get("gt_points", None)
            if target_valid is None:
                target_valid = torch.ones((len(target_texts),), dtype=torch.float32)
            target_valid = target_valid.to(dtype=torch.float32)
            if torch.is_tensor(target_label):
                target_label = target_label.to(dtype=torch.long)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(device.type == "cuda"),
            ):
                generated_ids = model.generate(
                    joint_inputs=joint_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=beam_k,
                    num_return_sequences=beam_k,
                )

            preds_flat = _decode_generated_text(
                processor=processor,
                generated_ids=generated_ids.detach().cpu(),
                input_ids=joint_inputs["input_ids"].detach().cpu(),
                attention_mask=joint_inputs.get("attention_mask", None).detach().cpu()
                if torch.is_tensor(joint_inputs.get("attention_mask", None))
                else None,
                num_return_sequences=beam_k,
            )

            bsz = len(target_texts)
            preds_by_sample: list[list[str]] = []
            for i in range(bsz):
                s = i * beam_k
                e = s + beam_k
                cand = preds_flat[s:e]
                if not cand:
                    cand = [""]
                preds_by_sample.append(cand)

            for i, pred_list in enumerate(preds_by_sample):
                total += 1
                if i >= len(target_texts):
                    continue
                pred_top1 = str(pred_list[0]) if pred_list else ""
                top3_obj: list[int] = []
                for pred in pred_list:
                    obj = _parse_object_id(pred)
                    if (obj is None) or (int(obj) < 0):
                        continue
                    obj_i = int(obj)
                    if obj_i in top3_obj:
                        continue
                    top3_obj.append(obj_i)
                    if len(top3_obj) >= 3:
                        break
                pred_obj = int(top3_obj[0]) if top3_obj else None
                pred_point = _parse_point_xy(pred_top1)
                if (
                    pred_point is not None
                    and isinstance(gt_points_batch, list)
                    and i < len(gt_points_batch)
                ):
                    l2_pair = _avg_min_l2_to_gt_points(pred_point, gt_points_batch[i])
                    if l2_pair is not None:
                        avg_l2_sum += float(l2_pair[0])
                        min_l2_sum += float(l2_pair[1])
                        l2_den += 1
                if float(target_valid[i].item()) <= 0.0:
                    continue

                tgt_n = _normalize_text(target_texts[i])
                pred_n = _normalize_text(pred_top1)
                valid_total += 1
                exact_match += int(pred_n == tgt_n)
                contains_match += int((tgt_n != "") and (tgt_n in pred_n))
                if (pred_obj is not None) and (pred_obj >= 0):
                    object_id_valid_num += 1

                gt_obj = -1
                if torch.is_tensor(target_label):
                    gt_obj = int(target_label[i].item())
                    if gt_obj >= 0:
                        acc1_den += 1
                        acc3_den += 1
                        acc1_num += int((pred_obj is not None) and (int(pred_obj) == gt_obj))
                        acc3_num += int(gt_obj in top3_obj)

                gt_multi: list[int] = []
                if isinstance(target_label_ids_batch, list) and i < len(target_label_ids_batch):
                    raw_multi = target_label_ids_batch[i]
                    if isinstance(raw_multi, list):
                        gt_multi = [int(x) for x in raw_multi if int(x) >= 0]
                if (not gt_multi) and (gt_obj >= 0):
                    gt_multi = [int(gt_obj)]
                if gt_multi:
                    multiacc1_den += 1
                    multiacc1_num += int((pred_obj is not None) and (int(pred_obj) in set(gt_multi)))

            if show_tqdm and valid_total > 0:
                test_iter.set_postfix(
                    l2=f"{(avg_l2_sum / max(l2_den, 1)):.4f}",
                    acc1=f"{(acc1_num / max(acc1_den, 1)):.4f}",
                    acc3=f"{(acc3_num / max(acc3_den, 1)):.4f}",
                )

    if total <= 0:
        return {
            "ExactMatch": 0.0,
            "Contains": 0.0,
            "Avg L2": 0.0,
            "Min L2": 0.0,
            "PointL2": 0.0,
            "acc@1": 0.0,
            "acc@3": 0.0,
            "multiacc@1": 0.0,
            "ObjectIDValidRate": 0.0,
            "num_samples": 0.0,
            "num_valid_targets": 0.0,
        }

    return {
        "ExactMatch": float(exact_match / max(valid_total, 1)),
        "Contains": float(contains_match / max(valid_total, 1)),
        "Avg L2": float(avg_l2_sum / max(l2_den, 1)),
        "Min L2": float(min_l2_sum / max(l2_den, 1)),
        "PointL2": float(avg_l2_sum / max(l2_den, 1)),
        "acc@1": float(acc1_num / max(acc1_den, 1)),
        "acc@3": float(acc3_num / max(acc3_den, 1)),
        "multiacc@1": float(multiacc1_num / max(multiacc1_den, 1)),
        "ObjectIDValidRate": float(object_id_valid_num / max(valid_total, 1)),
        "num_samples": float(total),
        "num_valid_targets": float(valid_total),
    }


def print_test_metrics_table(test_metrics: dict[str, float]) -> None:
    rows = [
        ("ExactMatch", float(test_metrics.get("ExactMatch", 0.0))),
        ("Contains", float(test_metrics.get("Contains", 0.0))),
        ("Avg L2", float(test_metrics.get("Avg L2", test_metrics.get("PointL2", 0.0)))),
        ("Min L2", float(test_metrics.get("Min L2", 0.0))),
        ("acc@1", float(test_metrics.get("acc@1", 0.0))),
        ("acc@3", float(test_metrics.get("acc@3", 0.0))),
        ("multiacc@1", float(test_metrics.get("multiacc@1", 0.0))),
        ("ObjectIDValidRate", float(test_metrics.get("ObjectIDValidRate", 0.0))),
        ("num_samples", float(test_metrics.get("num_samples", 0.0))),
        ("num_valid_targets", float(test_metrics.get("num_valid_targets", 0.0))),
    ]
    key_w = max(len(k) for k, _ in rows)
    val_w = 12
    line = "+" + "-" * (key_w + 2) + "+" + "-" * (val_w + 2) + "+"
    print("[TEST] text metrics")
    print(line)
    print(f"| {'Metric'.ljust(key_w)} | {'Value'.rjust(val_w)} |")
    print(line)
    for k, v in rows:
        if k.startswith("num_"):
            print(f"| {k.ljust(key_w)} | {v:>{val_w}.0f} |")
        else:
            print(f"| {k.ljust(key_w)} | {v:>{val_w}.6f} |")
    print(line)
