from __future__ import annotations

import math
import re
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .loss_utils import compute_structured_losses
from .object_tokens import parse_object_id_from_text, parse_object_token

INVALID_OBJECT_LABEL_ID = -100


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
            # Keep additional special tokens (e.g., <obj_127>) in decoded text.
            txt = tok.decode(new_tokens, skip_special_tokens=False)
        else:
            txt = str(new_tokens.tolist())
        # Remove chat/control specials while preserving object class tokens.
        txt = re.sub(r"<\|[^>]+?\|>", " ", str(txt))
        preds.append(str(txt).strip())
    return preds


def _extract_generated_new_token_ids(
    generated_ids: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_return_sequences: int = 1,
) -> list[list[int]]:
    out: list[list[int]] = []
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
        out.append([int(x) for x in new_tokens.tolist()])
    return out


def _build_object_token_id_to_label_map(processor: Any) -> dict[int, int]:
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return {}
    if not hasattr(tok, "get_vocab"):
        return {}
    try:
        vocab = tok.get_vocab()
    except Exception:
        return {}
    out: dict[int, int] = {}
    for token_str, token_id in vocab.items():
        cls = parse_object_token(str(token_str))
        if cls is None:
            continue
        try:
            tid = int(token_id)
            cid = int(cls)
        except Exception:
            continue
        if tid >= 0 and cid >= 0:
            out[tid] = cid
    return out


def _parse_object_id_from_token_ids(
    token_ids: list[int],
    object_token_id_to_label: dict[int, int],
) -> int | None:
    if not token_ids:
        return None
    if not object_token_id_to_label:
        return None
    for tid in token_ids:
        cls = object_token_id_to_label.get(int(tid), None)
        if cls is not None and int(cls) >= 0:
            return int(cls)
    return None


def _parse_object_id_with_fallback(
    pred_text: str,
    generated_token_ids: list[int],
    object_token_id_to_label: dict[int, int],
) -> tuple[int | None, str]:
    obj_tok = _parse_object_id_from_token_ids(generated_token_ids, object_token_id_to_label)
    if obj_tok is not None and int(obj_tok) >= 0:
        return int(obj_tok), "token_ids"
    obj_txt = _parse_object_id(str(pred_text))
    if obj_txt is not None and int(obj_txt) >= 0:
        return int(obj_txt), "text_fallback"
    return None, "failed"


def _parse_object_id(text: str) -> int | None:
    return parse_object_id_from_text(str(text or ""))


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
    loss_answer_weight: float = 0.1,
    loss_point_weight: float = 1.0,
    loss_object_weight: float = 1.5,
    loss_use_lm_fallback: bool = False,
    show_tqdm: bool = True,
    desc: str = "Eval",
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    answer_loss_sum = 0.0
    point_loss_sum = 0.0
    object_loss_sum = 0.0
    sample_count = 0
    point_w = float(loss_point_weight)
    object_w = float(loss_object_weight)

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
                loss_mask_object=batch.get("loss_mask_object", None),
                weight_answer=float(loss_answer_weight),
                weight_point=point_w,
                weight_object=object_w,
                fallback_loss=(loss if bool(loss_use_lm_fallback) else None),
            )
            eval_loss = structured["loss"]

            loss_sum += float(eval_loss.detach().item()) * float(bsz)
            answer_loss_sum += float(structured["loss_answer"].detach().item()) * float(bsz)
            point_loss_sum += float(structured["loss_point"].detach().item()) * float(bsz)
            object_loss_sum += float(structured["loss_object"].detach().item()) * float(bsz)
            sample_count += bsz
            if show_tqdm:
                eval_iter.set_postfix(loss=f"{(loss_sum / max(sample_count, 1)):.4f}")

    if sample_count <= 0:
        return {
            "loss": 0.0,
            "loss_total": 0.0,
            "loss_answer": 0.0,
            "loss_point": 0.0,
            "loss_object": 0.0,
        }
    denom = float(sample_count)
    total = float(loss_sum / denom)
    answer = float(answer_loss_sum / denom)
    point = float(point_loss_sum / denom)
    obj = float(object_loss_sum / denom)
    return {
        "loss": total,
        "loss_total": total,
        "loss_answer": answer,
        "loss_point": point,
        "loss_object": obj,
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
    object_token_valid_num = 0
    object_parse_fail_top1_num = 0
    object_parse_fail_beam_num = 0
    object_parse_top1_from_token_num = 0
    object_parse_top1_from_text_num = 0
    object_parse_beam_from_token_num = 0
    object_parse_beam_from_text_num = 0
    avg_l2_sum = 0.0
    min_l2_sum = 0.0
    l2_den = 0
    beam_k = max(1, int(num_beams))
    object_token_id_to_label = _build_object_token_id_to_label_map(processor)

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
            generated_ids_cpu = generated_ids.detach().cpu()
            input_ids_cpu = joint_inputs["input_ids"].detach().cpu()
            attention_mask_cpu = (
                joint_inputs.get("attention_mask", None).detach().cpu()
                if torch.is_tensor(joint_inputs.get("attention_mask", None))
                else None
            )

            preds_flat = _decode_generated_text(
                processor=processor,
                generated_ids=generated_ids_cpu,
                input_ids=input_ids_cpu,
                attention_mask=attention_mask_cpu,
                num_return_sequences=beam_k,
            )
            gen_token_ids_flat = _extract_generated_new_token_ids(
                generated_ids=generated_ids_cpu,
                input_ids=input_ids_cpu,
                attention_mask=attention_mask_cpu,
                num_return_sequences=beam_k,
            )

            bsz = len(target_texts)
            preds_by_sample: list[list[str]] = []
            gen_ids_by_sample: list[list[list[int]]] = []
            for i in range(bsz):
                s = i * beam_k
                e = s + beam_k
                cand = preds_flat[s:e]
                cand_ids = gen_token_ids_flat[s:e]
                if not cand:
                    cand = [""]
                if not cand_ids:
                    cand_ids = [[]]
                preds_by_sample.append(cand)
                gen_ids_by_sample.append(cand_ids)

            for i, pred_list in enumerate(preds_by_sample):
                total += 1
                if i >= len(target_texts):
                    continue
                pred_top1 = str(pred_list[0]) if pred_list else ""
                top3_obj: list[int] = []
                top3_sources: list[str] = []
                pred_token_ids_list = gen_ids_by_sample[i] if i < len(gen_ids_by_sample) else [[] for _ in pred_list]
                top1_obj, top1_src = _parse_object_id_with_fallback(
                    pred_text=pred_top1,
                    generated_token_ids=(pred_token_ids_list[0] if pred_token_ids_list else []),
                    object_token_id_to_label=object_token_id_to_label,
                )
                for j, pred in enumerate(pred_list):
                    pred_ids = pred_token_ids_list[j] if j < len(pred_token_ids_list) else []
                    obj, src = _parse_object_id_with_fallback(
                        pred_text=str(pred),
                        generated_token_ids=pred_ids,
                        object_token_id_to_label=object_token_id_to_label,
                    )
                    if (obj is None) or (int(obj) < 0):
                        continue
                    obj_i = int(obj)
                    if obj_i in top3_obj:
                        continue
                    top3_obj.append(obj_i)
                    top3_sources.append(str(src))
                    if len(top3_obj) >= 3:
                        break
                pred_obj = int(top3_obj[0]) if top3_obj else int(INVALID_OBJECT_LABEL_ID)
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
                if pred_obj >= 0:
                    object_token_valid_num += 1
                if str(top1_src) == "token_ids":
                    object_parse_top1_from_token_num += 1
                elif str(top1_src) == "text_fallback":
                    object_parse_top1_from_text_num += 1
                else:
                    object_parse_fail_top1_num += 1
                if len(top3_obj) <= 0:
                    object_parse_fail_beam_num += 1
                elif any(str(x) == "token_ids" for x in top3_sources):
                    object_parse_beam_from_token_num += 1
                elif any(str(x) == "text_fallback" for x in top3_sources):
                    object_parse_beam_from_text_num += 1
                else:
                    object_parse_fail_beam_num += 1

                gt_obj = -1
                if torch.is_tensor(target_label):
                    gt_obj = int(target_label[i].item())
                    if gt_obj >= 0:
                        acc1_den += 1
                        acc3_den += 1
                        acc1_num += int(int(pred_obj) == gt_obj)
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
                    multiacc1_num += int(int(pred_obj) in set(gt_multi))

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
            "ObjectTokenValidRate": 0.0,
            "ObjectParseFailTop1Rate": 0.0,
            "ObjectParseFailBeamRate": 0.0,
            "ObjectParseTop1FromTokenRate": 0.0,
            "ObjectParseTop1FromTextRate": 0.0,
            "ObjectParseBeamFromTokenRate": 0.0,
            "ObjectParseBeamFromTextRate": 0.0,
            "ObjectParseFailTop1Count": 0.0,
            "ObjectParseFailBeamCount": 0.0,
            "ObjectParseTop1FromTokenCount": 0.0,
            "ObjectParseTop1FromTextCount": 0.0,
            "ObjectParseBeamFromTokenCount": 0.0,
            "ObjectParseBeamFromTextCount": 0.0,
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
        "ObjectTokenValidRate": float(object_token_valid_num / max(valid_total, 1)),
        "ObjectParseFailTop1Rate": float(object_parse_fail_top1_num / max(valid_total, 1)),
        "ObjectParseFailBeamRate": float(object_parse_fail_beam_num / max(valid_total, 1)),
        "ObjectParseTop1FromTokenRate": float(object_parse_top1_from_token_num / max(valid_total, 1)),
        "ObjectParseTop1FromTextRate": float(object_parse_top1_from_text_num / max(valid_total, 1)),
        "ObjectParseBeamFromTokenRate": float(object_parse_beam_from_token_num / max(valid_total, 1)),
        "ObjectParseBeamFromTextRate": float(object_parse_beam_from_text_num / max(valid_total, 1)),
        "ObjectParseFailTop1Count": float(object_parse_fail_top1_num),
        "ObjectParseFailBeamCount": float(object_parse_fail_beam_num),
        "ObjectParseTop1FromTokenCount": float(object_parse_top1_from_token_num),
        "ObjectParseTop1FromTextCount": float(object_parse_top1_from_text_num),
        "ObjectParseBeamFromTokenCount": float(object_parse_beam_from_token_num),
        "ObjectParseBeamFromTextCount": float(object_parse_beam_from_text_num),
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
        ("ObjectTokenValidRate", float(test_metrics.get("ObjectTokenValidRate", 0.0))),
        ("ObjectParseFailTop1Rate", float(test_metrics.get("ObjectParseFailTop1Rate", 0.0))),
        ("ObjectParseFailBeamRate", float(test_metrics.get("ObjectParseFailBeamRate", 0.0))),
        ("ObjectParseTop1FromTokenRate", float(test_metrics.get("ObjectParseTop1FromTokenRate", 0.0))),
        ("ObjectParseTop1FromTextRate", float(test_metrics.get("ObjectParseTop1FromTextRate", 0.0))),
        ("ObjectParseBeamFromTokenRate", float(test_metrics.get("ObjectParseBeamFromTokenRate", 0.0))),
        ("ObjectParseBeamFromTextRate", float(test_metrics.get("ObjectParseBeamFromTextRate", 0.0))),
        ("ObjectParseFailTop1Count", float(test_metrics.get("ObjectParseFailTop1Count", 0.0))),
        ("ObjectParseFailBeamCount", float(test_metrics.get("ObjectParseFailBeamCount", 0.0))),
        ("ObjectParseTop1FromTokenCount", float(test_metrics.get("ObjectParseTop1FromTokenCount", 0.0))),
        ("ObjectParseTop1FromTextCount", float(test_metrics.get("ObjectParseTop1FromTextCount", 0.0))),
        ("ObjectParseBeamFromTokenCount", float(test_metrics.get("ObjectParseBeamFromTokenCount", 0.0))),
        ("ObjectParseBeamFromTextCount", float(test_metrics.get("ObjectParseBeamFromTextCount", 0.0))),
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
        if k.startswith("num_") or k.endswith("Count"):
            print(f"| {k.ljust(key_w)} | {v:>{val_w}.0f} |")
        else:
            print(f"| {k.ljust(key_w)} | {v:>{val_w}.6f} |")
    print(line)
