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
        src_i = min(max(0, i // nrs), bsz - 1) if bsz > 0 else i
        if attention_mask is not None and torch.is_tensor(attention_mask) and attention_mask.dim() >= 2 and bsz > 0:
            start = int(attention_mask[src_i].sum().item())
        else:
            start = int(input_ids.shape[1])
        new_tokens = generated_ids[i, start:]
        txt = tok.decode(new_tokens, skip_special_tokens=False) if tok is not None else str(new_tokens.tolist())
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
    px, py = float(pred_xy[0]), float(pred_xy[1])
    dists = [
        math.sqrt((px - float(pts[j, 0].item())) ** 2 + (py - float(pts[j, 1].item())) ** 2)
        for j in range(int(pts.shape[0]))
    ]
    return float(sum(dists) / len(dists)), float(min(dists))


def topk_similarity(query: torch.Tensor, bank: torch.Tensor, k: int, temperature: float) -> list[int]:
    if (not torch.is_tensor(query)) or query.dim() != 1:
        return []
    if (not torch.is_tensor(bank)) or bank.dim() != 2 or int(bank.shape[0]) <= 0:
        return []
    q = F.normalize(query.unsqueeze(0), p=2, dim=-1)
    b = F.normalize(bank, p=2, dim=-1)
    sim = (q @ b.t()) / max(float(temperature), 1e-6)
    kk = min(max(1, int(k)), int(b.shape[0]))
    return [int(x) for x in torch.topk(sim.squeeze(0), k=kk, largest=True, sorted=True).indices.tolist()]


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
    if not scene_images or not text_inputs or not generated_texts:
        return None, None
    if not (len(scene_images) == len(text_inputs) == len(generated_texts)):
        return None, None

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
    _fmt, _p, obj_mask = component_masks(
        processor=processor,
        joint_inputs=dict(joint),
        target_texts=[str(x) for x in generated_texts],
        target_valid=valid,
    )
    joint = to_device(dict(joint), device=device)
    obj_mask = obj_mask.to(device=device, dtype=torch.bool)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
        out = model(
            joint_inputs=joint,
            labels=None,
            object_slot_mask=obj_mask,
            use_cache=False,
        )
    pred = out.get("pred_object_emb", None)
    pred_valid = out.get("object_slot_valid", None)
    if not torch.is_tensor(pred):
        return None, None
    if not torch.is_tensor(pred_valid):
        pred_valid = obj_mask.any(dim=1)
    return pred.detach(), pred_valid.detach()


def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    loss_fmt_weight: float = 0.6,
    loss_point_weight: float = 2.0,
    loss_object_weight: float = 0.5,
    loss_use_lm_fallback: bool = False,
    label_embedding_bank: torch.Tensor | None = None,
    object_temperature: float = 0.07,
    show_tqdm: bool = True,
    desc: str = "Eval",
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    fmt_sum = 0.0
    point_sum = 0.0
    object_sum = 0.0
    count = 0

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

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                out = model(joint_inputs=joint, labels=labels, object_slot_mask=obj_mask, use_cache=False)
            lm_loss = out.get("loss", None)
            if lm_loss is None:
                raise RuntimeError("Model forward must return loss during evaluation.")
            losses = compute_structured_losses(
                logits=out.get("logits", None),
                labels=labels,
                loss_mask_fmt=batch.get("loss_mask_answer", None),
                loss_mask_point=batch.get("loss_mask_point", None),
                loss_mask_object=batch.get("loss_mask_object", None),
                pred_object_emb=out.get("pred_object_emb", None),
                target_label=y,
                target_object_valid=y_valid,
                label_embedding_bank=label_embedding_bank,
                object_temperature=object_temperature,
                weight_fmt=float(loss_fmt_weight),
                weight_point=float(loss_point_weight),
                weight_object=float(loss_object_weight),
                fallback_loss=(lm_loss if bool(loss_use_lm_fallback) else None),
            )
            total = losses["loss"]
            loss_sum += float(total.detach().item()) * float(bsz)
            fmt_sum += float(losses["loss_fmt"].detach().item()) * float(bsz)
            point_sum += float(losses["loss_point"].detach().item()) * float(bsz)
            object_sum += float(losses["loss_object"].detach().item()) * float(bsz)
            count += bsz
            if show_tqdm:
                it.set_postfix(loss=f"{(loss_sum / max(count, 1)):.4f}")

    if count <= 0:
        return {"loss": 0.0, "loss_total": 0.0, "loss_fmt": 0.0, "loss_point": 0.0, "loss_object": 0.0}
    d = float(count)
    total_avg = float(loss_sum / d)
    return {
        "loss": total_avg,
        "loss_total": total_avg,
        "loss_fmt": float(fmt_sum / d),
        "loss_point": float(point_sum / d),
        "loss_object": float(object_sum / d),
    }


def run_test_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    processor: Any,
    label_embedding_bank: torch.Tensor,
    retrieval_label_texts: list[str],
    query_id_to_label_text: dict[int, str] | None = None,
    retrieval_top_k: int = 3,
    object_temperature: float = 0.07,
    show_tqdm: bool = True,
    desc: str = "Test",
    max_new_tokens: int = 16,
    num_beams: int = 3,
) -> dict[str, float]:
    model.eval()
    if (not torch.is_tensor(label_embedding_bank)) or label_embedding_bank.dim() != 2:
        raise RuntimeError("label_embedding_bank must be a [N, D] tensor for retrieval evaluation.")

    keep = min(int(label_embedding_bank.shape[0]), len(retrieval_label_texts))
    if keep <= 0:
        raise RuntimeError("retrieval label bank is empty.")
    bank = label_embedding_bank[:keep].to(device=device, dtype=torch.float32)
    label_texts = [str(x) for x in retrieval_label_texts[:keep]]

    total = 0
    valid_total = 0
    avg_l2_sum = 0.0
    min_l2_sum = 0.0
    l2_den = 0
    # Retrieval counters.
    # acc@1, acc@3: primary GT only.  multiacc@1: full multi-label GT.
    retrieval_den = 0        # samples with a primary GT label
    multiacc_den = 0         # samples with at least one GT label (primary or multi)
    retrieval_acc1 = 0
    retrieval_acc3 = 0
    retrieval_multiacc1 = 0
    parse_fail = 0           # failed embedding extractions (no <obj_emb> in generated text)
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
            # Top-1 beam per sample.
            top1_texts = [flat[i * beam_k] if (i * beam_k) < len(flat) else "" for i in range(bsz)]

            scene_images = batch.get("scene_images", None)
            text_inputs_raw = batch.get("text_inputs", None)
            pred_emb, pred_ok = pred_embeddings_from_generated(
                model=model,
                processor=processor,
                scene_images=list(scene_images) if isinstance(scene_images, list) else [],
                text_inputs=[str(x) for x in text_inputs_raw] if isinstance(text_inputs_raw, list) else [],
                generated_texts=top1_texts,
                device=device,
                amp_dtype=amp_dtype,
            )

            for i in range(bsz):
                total += 1
                pred_top1 = top1_texts[i]

                # Localization: Avg L2 and Min L2 over all annotator GT points.
                pt = parse_point(pred_top1)
                if pt is not None and isinstance(gt_points, list) and i < len(gt_points):
                    stats = l2_stats(pt, gt_points[i])
                    if stats is not None:
                        avg_l2_sum += float(stats[0])
                        min_l2_sum += float(stats[1])
                        l2_den += 1

                if float(target_valid[i].item()) <= 0.0:
                    continue
                valid_total += 1

                # Build GT label sets.
                # gt_primary_set: canonical single GT label → used for acc@1 and acc@3.
                # gt_multi_set:   all valid GT labels       → used for multiacc@1.
                gt_primary_set: set[str] = set()
                if isinstance(target_text_label, list) and i < len(target_text_label):
                    t = normalize_text(str(target_text_label[i]))
                    if t:
                        gt_primary_set.add(t)

                gt_multi_set: set[str] = set(gt_primary_set)
                if isinstance(target_label_ids, list) and i < len(target_label_ids):
                    raw_multi = target_label_ids[i]
                    if isinstance(raw_multi, list) and isinstance(query_id_to_label_text, dict):
                        for x in raw_multi:
                            xid = int(x)
                            if xid < 0:
                                continue
                            t = normalize_text(str(query_id_to_label_text.get(xid, "")))
                            if t:
                                gt_multi_set.add(t)

                if gt_primary_set:
                    retrieval_den += 1
                if gt_multi_set:
                    multiacc_den += 1

                # Object embedding extraction.
                emb_ok = (
                    torch.is_tensor(pred_emb)
                    and torch.is_tensor(pred_ok)
                    and i < int(pred_emb.shape[0])
                    and i < int(pred_ok.numel())
                    and bool(pred_ok[i].item())
                )
                if not emb_ok:
                    parse_fail += 1
                    continue

                q = pred_emb[i].to(device=device, dtype=torch.float32)
                if float(q.norm().item()) <= 0:
                    parse_fail += 1
                    continue

                topk_idx = topk_similarity(q, bank, retrieval_top_k, object_temperature)
                topk_text = [normalize_text(label_texts[j]) for j in topk_idx if 0 <= int(j) < len(label_texts)]
                topk_text = [x for x in topk_text if x]
                if not topk_text:
                    parse_fail += 1
                    continue

                pred_label = topk_text[0]

                # Acc@1, Acc@3: top-k prediction vs. primary GT.
                if gt_primary_set:
                    retrieval_acc1 += int(pred_label in gt_primary_set)
                    retrieval_acc3 += int(any(x in gt_primary_set for x in topk_text[:3]))

                # MultiAcc@1: top-1 prediction vs. full multi-label GT set.
                if gt_multi_set:
                    retrieval_multiacc1 += int(pred_label in gt_multi_set)

            if show_tqdm and valid_total > 0:
                it.set_postfix(
                    l2=f"{(avg_l2_sum / max(l2_den, 1)):.4f}",
                    acc1=f"{(retrieval_acc1 / max(retrieval_den, 1)):.4f}",
                    macc1=f"{(retrieval_multiacc1 / max(multiacc_den, 1)):.4f}",
                )

    if total <= 0:
        return _empty_test_metrics()

    return {
        "Avg L2": float(avg_l2_sum / max(l2_den, 1)),
        "Min L2": float(min_l2_sum / max(l2_den, 1)),
        "acc@1": float(retrieval_acc1 / max(retrieval_den, 1)),
        "acc@3": float(retrieval_acc3 / max(retrieval_den, 1)),
        "multiacc@1": float(retrieval_multiacc1 / max(multiacc_den, 1)),
        "RetrievalQueryValidRate": float((valid_total - parse_fail) / max(valid_total, 1)),
        "RetrievalAcc@1": float(retrieval_acc1 / max(retrieval_den, 1)),
        "RetrievalAcc@3": float(retrieval_acc3 / max(retrieval_den, 1)),
        "RetrievalMultiAcc@1": float(retrieval_multiacc1 / max(multiacc_den, 1)),
        "ParseFailRate": float(parse_fail / max(valid_total, 1)),
        "RetrievalDen": float(retrieval_den),
        "MultiAccDen": float(multiacc_den),
        "num_samples": float(total),
        "num_valid_targets": float(valid_total),
    }


def _empty_test_metrics() -> dict[str, float]:
    return {
        "Avg L2": 0.0,
        "Min L2": 0.0,
        "acc@1": 0.0,
        "acc@3": 0.0,
        "multiacc@1": 0.0,
        "RetrievalQueryValidRate": 0.0,
        "RetrievalAcc@1": 0.0,
        "RetrievalAcc@3": 0.0,
        "RetrievalMultiAcc@1": 0.0,
        "ParseFailRate": 0.0,
        "RetrievalDen": 0.0,
        "MultiAccDen": 0.0,
        "num_samples": 0.0,
        "num_valid_targets": 0.0,
    }


def print_test_metrics_table(test_metrics: dict[str, float]) -> None:
    rows = [
        ("Avg L2", float(test_metrics.get("Avg L2", 0.0))),
        ("Min L2", float(test_metrics.get("Min L2", 0.0))),
        ("acc@1", float(test_metrics.get("acc@1", 0.0))),
        ("acc@3", float(test_metrics.get("acc@3", 0.0))),
        ("multiacc@1", float(test_metrics.get("multiacc@1", 0.0))),
        ("RetrievalQueryValidRate", float(test_metrics.get("RetrievalQueryValidRate", 0.0))),
        ("RetrievalAcc@1", float(test_metrics.get("RetrievalAcc@1", 0.0))),
        ("RetrievalAcc@3", float(test_metrics.get("RetrievalAcc@3", 0.0))),
        ("RetrievalMultiAcc@1", float(test_metrics.get("RetrievalMultiAcc@1", 0.0))),
        ("ParseFailRate", float(test_metrics.get("ParseFailRate", 0.0))),
        ("RetrievalDen", float(test_metrics.get("RetrievalDen", 0.0))),
        ("MultiAccDen", float(test_metrics.get("MultiAccDen", 0.0))),
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
        if k.startswith("num_") or k.endswith("Den"):
            print(f"| {k.ljust(key_w)} | {v:>{val_w}.0f} |")
        else:
            print(f"| {k.ljust(key_w)} | {v:>{val_w}.6f} |")
    print(line)
