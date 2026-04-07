from __future__ import annotations

import math
import re
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .common import chat_text, normalize_text, to_device
from .loss_utils import compute_structured_losses
from .processor_collate import component_masks


def parse_object_text(text: str) -> str | None:
    """Extract the label text from a generated 'Object: <label>' line.

    Returns the stripped label string, or None if the line is absent or empty.
    Works for both pure-text format ('Object: television') and the legacy slot
    format ('Object: <obj_emb>') – though the latter is treated as unparseable
    for retrieval purposes.
    """
    m = re.search(r"(?im)^\s*object\s*:\s*(\S.*?)\s*$", str(text or ""))
    if m is None:
        return None
    val = str(m.group(1)).strip()
    if not val or val == "<obj_emb>":
        return None
    return val


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
        if bsz > 0:
            src_i = min(max(0, int(src_i)), bsz - 1)
        if attention_mask is not None and torch.is_tensor(attention_mask) and attention_mask.dim() >= 2 and bsz > 0:
            start = int(attention_mask[src_i].sum().item())
        else:
            start = int(input_ids.shape[1])
        new_tokens = generated_ids[i, start:]
        if tok is not None:
            txt = tok.decode(new_tokens, skip_special_tokens=False)
        else:
            txt = str(new_tokens.tolist())
        txt = re.sub(r"<\|[^>]+?\|>", " ", str(txt))
        out.append(str(txt).strip())
    return out


def parse_point(text: str) -> tuple[float, float] | None:
    num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    m = re.search(rf"(?im)^\s*point\s*:\s*({num})\s*[,\s]+\s*({num})\b", str(text or ""))
    if m is None:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return None


def l2_stats(pred_xy: tuple[float, float], gt_points: torch.Tensor) -> tuple[float, float] | None:
    if (not torch.is_tensor(gt_points)) or gt_points.numel() < 2:
        return None
    if int(gt_points.numel()) % 2 != 0:
        return None
    pts = gt_points.to(dtype=torch.float32).view(-1, 2)
    if int(pts.shape[0]) <= 0:
        return None
    px = float(pred_xy[0])
    py = float(pred_xy[1])
    dists = [math.sqrt((px - float(pts[j, 0].item())) ** 2 + (py - float(pts[j, 1].item())) ** 2) for j in range(int(pts.shape[0]))]
    if not dists:
        return None
    return float(sum(dists) / len(dists)), float(min(dists))


def topk_similarity(query: torch.Tensor, bank: torch.Tensor, k: int, temperature: float) -> list[int]:
    if (not torch.is_tensor(query)) or query.dim() != 1:
        return []
    if (not torch.is_tensor(bank)) or bank.dim() != 2 or int(bank.shape[0]) <= 0:
        return []
    q = F.normalize(query.unsqueeze(0), p=2, dim=-1)
    b = F.normalize(bank, p=2, dim=-1)
    t = max(float(temperature), 1e-6)
    sim = (q @ b.t()) / t
    kk = min(max(1, int(k)), int(b.shape[0]))
    return [int(x) for x in torch.topk(sim.squeeze(0), k=kk, largest=True, sorted=True).indices.tolist()]


def pred_outputs_from_generated(
    *,
    model: torch.nn.Module,
    processor: Any,
    scene_images: list[Any],
    text_inputs: list[str],
    generated_texts: list[str],
    device: torch.device,
    amp_dtype: torch.dtype,
    point_head_enabled: bool = True,
) -> dict[str, Any]:
    """Re-encode prompt + generated text and run a forward pass to extract:
    - ``pred_object_emb`` [B, D]  : L2-normalized object-slot embeddings
    - ``pred_ok``         [B]     : bool — True if object-span pooling succeeded
    - ``pred_point_xy``   [B, 2]  : point head prediction in [0, 1]² (None if head disabled)
    - ``point_head_valid``[B]     : bool — True if point_span_mask had tokens

    Returns a dict with those keys (values may be None if unavailable).
    """
    if not scene_images or not text_inputs or not generated_texts:
        return {"pred_object_emb": None, "pred_ok": None, "pred_point_xy": None, "point_head_valid": None}
    if not (len(scene_images) == len(text_inputs) == len(generated_texts)):
        return {"pred_object_emb": None, "pred_ok": None, "pred_point_xy": None, "point_head_valid": None}

    chat = [
        chat_text(
            processor=processor,
            user_text=text_inputs[i],
            assistant_text=generated_texts[i],
            with_image=True,
            add_generation_prompt=False,
        )
        for i in range(len(generated_texts))
    ]
    joint = processor(
        text=chat,
        images=scene_images,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    valid = torch.ones((len(generated_texts),), dtype=torch.float32)
    _a, pt_mask, obj_mask = component_masks(
        processor=processor,
        joint_inputs=dict(joint),
        target_texts=[str(x) for x in generated_texts],
        target_valid=valid,
    )
    joint = to_device(dict(joint), device=device)
    obj_mask = obj_mask.to(device=device, dtype=torch.bool)
    pt_mask = pt_mask.to(device=device, dtype=torch.bool) if bool(point_head_enabled) else None

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
        out = model(
            joint_inputs=joint,
            labels=None,
            object_slot_mask=obj_mask,
            point_span_mask=pt_mask,
            use_cache=False,
        )

    pred_emb = out.get("pred_object_emb", None)
    pred_ok = out.get("object_slot_valid", None)
    pred_point_xy = out.get("pred_point_xy", None)
    point_head_valid = out.get("point_head_valid", None)

    if not torch.is_tensor(pred_ok):
        pred_ok = obj_mask.any(dim=1)

    return {
        "pred_object_emb": pred_emb.detach() if torch.is_tensor(pred_emb) else None,
        "pred_ok": pred_ok.detach() if torch.is_tensor(pred_ok) else None,
        "pred_point_xy": pred_point_xy.detach() if torch.is_tensor(pred_point_xy) else None,
        "point_head_valid": point_head_valid.detach() if torch.is_tensor(point_head_valid) else None,
    }


def pred_embeddings_from_generated(
    *,
    model: torch.nn.Module,
    processor: Any,
    scene_images: list[Any],
    text_inputs: list[str],
    generated_texts: list[str],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Backward-compatible wrapper around pred_outputs_from_generated().

    Returns (pred_object_emb, pred_ok) tuple as before.
    """
    outs = pred_outputs_from_generated(
        model=model,
        processor=processor,
        scene_images=scene_images,
        text_inputs=text_inputs,
        generated_texts=generated_texts,
        device=device,
        amp_dtype=amp_dtype,
        point_head_enabled=False,
    )
    return outs["pred_object_emb"], outs["pred_ok"]


def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    loss_answer_weight: float = 0.1,
    loss_point_weight: float = 1.0,
    loss_object_weight: float = 1.5,
    loss_slot_weight: float = 0.0,
    loss_use_lm_fallback: bool = False,
    label_embedding_bank: torch.Tensor | None = None,
    object_loss_mode: str = "retrieval",
    object_temperature: float = 0.07,
    # ── Point regression ───────────────────────────────────────────────────
    loss_point_reg_weight: float = 0.0,
    point_reg_loss_type: str = "smooth_l1",
    point_head_enabled: bool = True,
    # ──────────────────────────────────────────────────────────────────────
    show_tqdm: bool = True,
    desc: str = "Eval",
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    answer_sum = 0.0
    point_sum = 0.0
    object_sum = 0.0
    slot_sum = 0.0
    point_reg_sum = 0.0
    count = 0
    point_w = float(loss_point_weight)
    object_w = float(loss_object_weight)

    with torch.no_grad():
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not show_tqdm)
        for batch in it:
            joint = to_device(batch["joint_inputs"], device=device)
            labels = batch["labels"].to(device)
            if torch.all(labels.eq(-100)):
                continue
            bsz = int(labels.shape[0])
            obj_mask = batch.get("loss_mask_object", None)
            if torch.is_tensor(obj_mask):
                obj_mask = obj_mask.to(device=device, dtype=torch.bool)
            y = batch.get("target_label", None)
            if torch.is_tensor(y):
                y = y.to(device=device, dtype=torch.long)
            y_valid = batch.get("target_object_valid", None)
            if torch.is_tensor(y_valid):
                y_valid = y_valid.to(device=device, dtype=torch.float32)

            # Point regression inputs
            pt_mask = None
            if bool(point_head_enabled) and float(loss_point_reg_weight) > 0.0:
                pt_mask = batch.get("loss_mask_point", None)
                if torch.is_tensor(pt_mask):
                    pt_mask = pt_mask.to(device=device, dtype=torch.bool)
            target_point = batch.get("target_point", None)
            if torch.is_tensor(target_point):
                target_point = target_point.to(device=device, dtype=torch.float32)
            target_point_valid = batch.get("target_point_valid", None)
            if torch.is_tensor(target_point_valid):
                target_point_valid = target_point_valid.to(device=device, dtype=torch.float32)

            pass_labels = labels if bool(loss_use_lm_fallback) else None
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                out = model(
                    joint_inputs=joint,
                    labels=pass_labels,
                    object_slot_mask=obj_mask,
                    point_span_mask=pt_mask,
                    use_cache=False,
                )
            lm_loss = out.get("loss", None)
            if bool(loss_use_lm_fallback) and lm_loss is None:
                raise RuntimeError("Model forward must return loss when lm_fallback is enabled.")
            losses = compute_structured_losses(
                logits=out.get("logits", None),
                labels=labels,
                loss_mask_answer=batch.get("loss_mask_answer", None),
                loss_mask_point=batch.get("loss_mask_point", None),
                loss_mask_object=batch.get("loss_mask_object", None),
                pred_object_emb=out.get("pred_object_emb", None),
                target_label=y,
                target_object_valid=y_valid,
                label_embedding_bank=label_embedding_bank,
                object_loss_mode=object_loss_mode,
                object_temperature=object_temperature,
                weight_answer=float(loss_answer_weight),
                weight_point=point_w,
                weight_object=object_w,
                weight_slot=float(loss_slot_weight),
                pred_point_xy=out.get("pred_point_xy", None),
                target_point=target_point,
                target_point_valid=target_point_valid,
                point_head_valid=out.get("point_head_valid", None),
                weight_point_reg=float(loss_point_reg_weight),
                point_reg_loss_type=str(point_reg_loss_type),
                fallback_loss=(lm_loss if bool(loss_use_lm_fallback) else None),
            )
            total = losses["loss"]
            loss_sum += float(total.detach().item()) * float(bsz)
            answer_sum += float(losses["loss_answer"].detach().item()) * float(bsz)
            point_sum += float(losses["loss_point"].detach().item()) * float(bsz)
            object_sum += float(losses["loss_object"].detach().item()) * float(bsz)
            slot_sum += float(losses["loss_slot"].detach().item()) * float(bsz)
            point_reg_sum += float(losses["loss_point_reg"].detach().item()) * float(bsz)
            count += bsz
            if show_tqdm:
                it.set_postfix(loss=f"{(loss_sum / max(count, 1)):.4f}")

    if count <= 0:
        return {
            "loss": 0.0,
            "loss_total": 0.0,
            "loss_answer": 0.0,
            "loss_point": 0.0,
            "loss_object": 0.0,
            "loss_slot": 0.0,
            "loss_point_reg": 0.0,
        }
    d = float(count)
    total = float(loss_sum / d)
    return {
        "loss": total,
        "loss_total": total,
        "loss_answer": float(answer_sum / d),
        "loss_point": float(point_sum / d),
        "loss_object": float(object_sum / d),
        "loss_slot": float(slot_sum / d),
        "loss_point_reg": float(point_reg_sum / d),
    }


def run_test_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    processor: Any,
    label_embedding_bank: torch.Tensor,
    retrieval_label_texts: list[str],
    query_text_to_label_id: dict[str, int] | None = None,
    query_id_to_label_text: dict[int, str] | None = None,
    retrieval_top_k: int = 3,
    object_temperature: float = 0.07,
    show_tqdm: bool = True,
    desc: str = "Test",
    max_new_tokens: int = 16,
    num_beams: int = 3,
    point_head_enabled: bool = True,
) -> dict[str, float]:
    model.eval()
    if (not torch.is_tensor(label_embedding_bank)) or label_embedding_bank.dim() != 2:
        raise RuntimeError("label_embedding_bank must be a [N, D] tensor for retrieval evaluation.")
    if not isinstance(retrieval_label_texts, list):
        raise RuntimeError("retrieval_label_texts must be a list[str].")

    bank_n = int(label_embedding_bank.shape[0])
    text_n = int(len(retrieval_label_texts))
    keep = min(bank_n, text_n)
    if keep <= 0:
        raise RuntimeError("retrieval label bank is empty.")
    bank = label_embedding_bank[:keep].to(device=device, dtype=torch.float32)
    labels = [str(x) for x in retrieval_label_texts[:keep]]

    total = 0
    valid_total = 0
    exact = 0
    contains = 0
    avg_l2_sum = 0.0
    min_l2_sum = 0.0
    l2_den = 0
    # Point regression head metrics
    reg_l2_sum = 0.0
    reg_l2_den = 0
    acc1_den = 0       # samples with valid primary GT  (for acc@1, acc@3)
    multiacc1_den = 0  # samples with valid multi GT    (for multiacc@1)
    retrieval_den = 0  # backward-compat alias → acc1_den
    retrieval_acc1 = 0
    retrieval_acc3 = 0
    retrieval_multiacc1 = 0
    query_valid = 0
    object_valid = 0
    parse_fail_top1 = 0
    parse_fail_beam = 0
    parse_top1_token = 0
    parse_beam_token = 0
    beam_k = max(1, int(num_beams))

    with torch.no_grad():
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not show_tqdm)
        for batch in it:
            joint = to_device(batch["joint_inputs"], device=device)
            target_texts = [str(x) for x in batch.get("target_text", [])]
            target_valid = batch.get("target_text_valid", None)
            target_text_label = batch.get("target_label_text", None)
            target_label_ids = batch.get("target_label_ids", None)
            gt_points = batch.get("gt_points", None)
            if target_valid is None:
                target_valid = torch.ones((len(target_texts),), dtype=torch.float32)
            target_valid = target_valid.to(dtype=torch.float32)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                generated_ids = model.generate(
                    joint_inputs=joint,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=beam_k,
                    num_return_sequences=beam_k,
                )
            generated_ids_cpu = generated_ids.detach().cpu()
            input_ids_cpu = joint["input_ids"].detach().cpu()
            attn_cpu = joint.get("attention_mask", None)
            if torch.is_tensor(attn_cpu):
                attn_cpu = attn_cpu.detach().cpu()

            flat = decode_generated(
                processor=processor,
                generated_ids=generated_ids_cpu,
                input_ids=input_ids_cpu,
                attention_mask=attn_cpu,
                num_return_sequences=beam_k,
            )
            bsz = len(target_texts)
            preds_by_sample: list[list[str]] = []
            for i in range(bsz):
                s = i * beam_k
                e = s + beam_k
                cand = flat[s:e]
                preds_by_sample.append(cand if cand else [""])
            top1_texts = [str(preds_by_sample[i][0]) if preds_by_sample[i] else "" for i in range(bsz)]

            scene_images = batch.get("scene_images", None)
            text_inputs = batch.get("text_inputs", None)

            # Second forward pass: get object embedding + point head prediction
            fwd_outs = pred_outputs_from_generated(
                model=model,
                processor=processor,
                scene_images=list(scene_images) if isinstance(scene_images, list) else [],
                text_inputs=[str(x) for x in text_inputs] if isinstance(text_inputs, list) else [],
                generated_texts=top1_texts,
                device=device,
                amp_dtype=amp_dtype,
                point_head_enabled=bool(point_head_enabled),
            )
            pred_emb = fwd_outs["pred_object_emb"]
            pred_ok = fwd_outs["pred_ok"]
            pred_point_xy = fwd_outs["pred_point_xy"]       # [B, 2] or None
            pt_head_valid = fwd_outs["point_head_valid"]    # [B] bool or None

            for i in range(bsz):
                total += 1
                pred_top1 = top1_texts[i] if i < len(top1_texts) else ""

                # ── Text-parsed point L2 ─────────────────────────────────
                pt = parse_point(pred_top1)
                if pt is not None and isinstance(gt_points, list) and i < len(gt_points):
                    stats = l2_stats(pt, gt_points[i])
                    if stats is not None:
                        avg_l2_sum += float(stats[0])
                        min_l2_sum += float(stats[1])
                        l2_den += 1

                # ── Point head regression L2 ─────────────────────────────
                if (
                    bool(point_head_enabled)
                    and torch.is_tensor(pred_point_xy)
                    and torch.is_tensor(pt_head_valid)
                    and i < int(pred_point_xy.shape[0])
                    and i < int(pt_head_valid.numel())
                    and bool(pt_head_valid[i].item())
                    and isinstance(gt_points, list)
                    and i < len(gt_points)
                ):
                    reg_xy_i = pred_point_xy[i].to(dtype=torch.float32)
                    reg_pt = (float(reg_xy_i[0].item()), float(reg_xy_i[1].item()))
                    reg_stats = l2_stats(reg_pt, gt_points[i])
                    if reg_stats is not None:
                        reg_l2_sum += float(reg_stats[0])
                        reg_l2_den += 1

                if float(target_valid[i].item()) <= 0.0:
                    continue

                valid_total += 1
                tgt_norm = normalize_text(target_texts[i])
                pred_norm = normalize_text(pred_top1)
                exact += int(pred_norm == tgt_norm)
                contains += int((tgt_norm != "") and (tgt_norm in pred_norm))

                # Primary GT: single label used for acc@1 / acc@3.
                gt_primary: set[str] = set()
                if isinstance(target_text_label, list) and i < len(target_text_label):
                    t = normalize_text(str(target_text_label[i]))
                    if t:
                        gt_primary.add(t)

                # Multi GT: all alternative labels (incl. primary) for multiacc@1.
                gt_multi: set[str] = set()
                if isinstance(target_label_ids, list) and i < len(target_label_ids):
                    raw_multi = target_label_ids[i]
                    if isinstance(raw_multi, list) and isinstance(query_id_to_label_text, dict):
                        for x in raw_multi:
                            xid = int(x)
                            if xid < 0:
                                continue
                            t = normalize_text(str(query_id_to_label_text.get(xid, "")))
                            if t:
                                gt_multi.add(t)
                # Fallback: when no multi-label IDs are available, treat as primary only.
                if not gt_multi:
                    gt_multi = set(gt_primary)

                if gt_primary:
                    acc1_den += 1
                    retrieval_den = acc1_den  # keep alias in sync
                if gt_multi:
                    multiacc1_den += 1

                # --- object prediction: span-hidden retrieval (primary) ------
                ok = False
                if (
                    torch.is_tensor(pred_emb)
                    and torch.is_tensor(pred_ok)
                    and i < int(pred_emb.shape[0])
                    and i < int(pred_ok.numel())
                ):
                    ok = bool(pred_ok[i].item())

                q = None
                if ok and torch.is_tensor(pred_emb):
                    q_cand = pred_emb[i].to(device=device, dtype=torch.float32)
                    if float(q_cand.norm().item()) > 0:
                        q = q_cand

                # --- fallback: parse generated object text -> exact match ---
                gen_obj_text = parse_object_text(pred_top1)
                if q is None and gen_obj_text is None:
                    parse_fail_top1 += 1
                    parse_fail_beam += 1
                    continue

                if q is not None:
                    query_valid += 1
                    object_valid += 1
                    topk_idx = topk_similarity(q, bank, retrieval_top_k, object_temperature)
                    topk_text = [normalize_text(labels[j]) for j in topk_idx if 0 <= int(j) < len(labels)]
                    topk_text = [x for x in topk_text if x]
                elif gen_obj_text is not None:
                    # pure-text fallback: use parsed object string directly
                    norm_gen = normalize_text(gen_obj_text)
                    topk_text = [norm_gen] if norm_gen else []
                    object_valid += 1
                else:
                    topk_text = []

                if not topk_text:
                    parse_fail_top1 += 1
                    parse_fail_beam += 1
                    continue

                parse_top1_token += 1
                parse_beam_token += 1
                pred_label = topk_text[0]
                # acc@1 / acc@3: top-k prediction vs primary GT label only.
                if gt_primary:
                    retrieval_acc1 += int(pred_label in gt_primary)
                    retrieval_acc3 += int(any(x in gt_primary for x in topk_text[: min(3, len(topk_text))]))
                # multiacc@1: top-1 prediction vs full multi-label GT set.
                if gt_multi:
                    retrieval_multiacc1 += int(pred_label in gt_multi)

            if show_tqdm and valid_total > 0:
                it.set_postfix(
                    l2=f"{(avg_l2_sum / max(l2_den, 1)):.4f}",
                    reg_l2=f"{(reg_l2_sum / max(reg_l2_den, 1)):.4f}",
                    acc1=f"{(retrieval_acc1 / max(retrieval_den, 1)):.4f}",
                    acc3=f"{(retrieval_acc3 / max(retrieval_den, 1)):.4f}",
                )

    if total <= 0:
        return {
            "ExactMatch": 0.0,
            "Contains": 0.0,
            "Avg L2": 0.0,
            "Min L2": 0.0,
            "PointL2": 0.0,
            "RegressionL2": 0.0,
            "acc@1": 0.0,
            "acc@3": 0.0,
            "multiacc@1": 0.0,
            "ObjectTokenValidRate": 0.0,
            "ObjectParseFailTop1Rate": 0.0,
            "ObjectParseFailBeamRate": 0.0,
            "ObjectParseTop1FromTokenRate": 0.0,
            "ObjectParseBeamFromTokenRate": 0.0,
            "ObjectParseFailTop1Count": 0.0,
            "ObjectParseFailBeamCount": 0.0,
            "ObjectParseTop1FromTokenCount": 0.0,
            "ObjectParseBeamFromTokenCount": 0.0,
            "RetrievalQueryValidRate": 0.0,
            "RetrievalAcc@1": 0.0,
            "RetrievalAcc@3": 0.0,
            "RetrievalMultiAcc@1": 0.0,
            "RetrievalDen": 0.0,
            "num_samples": 0.0,
            "num_valid_targets": 0.0,
        }

    return {
        "ExactMatch": float(exact / max(valid_total, 1)),
        "Contains": float(contains / max(valid_total, 1)),
        "Avg L2": float(avg_l2_sum / max(l2_den, 1)),
        "Min L2": float(min_l2_sum / max(l2_den, 1)),
        "PointL2": float(avg_l2_sum / max(l2_den, 1)),
        "RegressionL2": float(reg_l2_sum / max(reg_l2_den, 1)),
        # acc@1 / acc@3 : primary GT label only  (denominator = acc1_den)
        # multiacc@1    : any label in multi-GT set (denominator = multiacc1_den)
        "acc@1": float(retrieval_acc1 / max(acc1_den, 1)),
        "acc@3": float(retrieval_acc3 / max(acc1_den, 1)),
        "multiacc@1": float(retrieval_multiacc1 / max(multiacc1_den, 1)),
        "ObjectTokenValidRate": float(object_valid / max(valid_total, 1)),
        "ObjectParseFailTop1Rate": float(parse_fail_top1 / max(valid_total, 1)),
        "ObjectParseFailBeamRate": float(parse_fail_beam / max(valid_total, 1)),
        "ObjectParseTop1FromTokenRate": float(parse_top1_token / max(valid_total, 1)),
        "ObjectParseBeamFromTokenRate": float(parse_beam_token / max(valid_total, 1)),
        "ObjectParseFailTop1Count": float(parse_fail_top1),
        "ObjectParseFailBeamCount": float(parse_fail_beam),
        "ObjectParseTop1FromTokenCount": float(parse_top1_token),
        "ObjectParseBeamFromTokenCount": float(parse_beam_token),
        "RetrievalQueryValidRate": float(query_valid / max(valid_total, 1)),
        "RetrievalAcc@1": float(retrieval_acc1 / max(acc1_den, 1)),
        "RetrievalAcc@3": float(retrieval_acc3 / max(acc1_den, 1)),
        "RetrievalMultiAcc@1": float(retrieval_multiacc1 / max(multiacc1_den, 1)),
        "RetrievalDen": float(acc1_den),
        "RetrievalMultiDen": float(multiacc1_den),
        "num_samples": float(total),
        "num_valid_targets": float(valid_total),
    }


def print_test_metrics_table(test_metrics: dict[str, float]) -> None:
    rows = [
        ("ExactMatch", float(test_metrics.get("ExactMatch", 0.0))),
        ("Contains", float(test_metrics.get("Contains", 0.0))),
        ("Avg L2 (text)", float(test_metrics.get("Avg L2", test_metrics.get("PointL2", 0.0)))),
        ("Min L2 (text)", float(test_metrics.get("Min L2", 0.0))),
        ("RegressionL2 (head)", float(test_metrics.get("RegressionL2", 0.0))),
        ("acc@1", float(test_metrics.get("acc@1", 0.0))),
        ("acc@3", float(test_metrics.get("acc@3", 0.0))),
        ("multiacc@1", float(test_metrics.get("multiacc@1", 0.0))),
        ("RetrievalQueryValidRate", float(test_metrics.get("RetrievalQueryValidRate", 0.0))),
        ("RetrievalAcc@1", float(test_metrics.get("RetrievalAcc@1", 0.0))),
        ("RetrievalAcc@3", float(test_metrics.get("RetrievalAcc@3", 0.0))),
        ("RetrievalMultiAcc@1", float(test_metrics.get("RetrievalMultiAcc@1", 0.0))),
        ("RetrievalDen (primary)", float(test_metrics.get("RetrievalDen", 0.0))),
        ("RetrievalMultiDen", float(test_metrics.get("RetrievalMultiDen", 0.0))),
        ("num_samples", float(test_metrics.get("num_samples", 0.0))),
        ("num_valid_targets", float(test_metrics.get("num_valid_targets", 0.0))),
    ]
    key_w = max(len(k) for k, _ in rows)
    val_w = 12
    line = "+" + "-" * (key_w + 2) + "+" + "-" * (val_w + 2) + "+"
    print("[TEST] metrics")
    print(line)
    print(f"| {'Metric'.ljust(key_w)} | {'Value'.rjust(val_w)} |")
    print(line)
    for k, v in rows:
        if k.startswith("num_") or "Den" in k:
            print(f"| {k.ljust(key_w)} | {v:>{val_w}.0f} |")
        else:
            print(f"| {k.ljust(key_w)} | {v:>{val_w}.6f} |")
    print(line)
