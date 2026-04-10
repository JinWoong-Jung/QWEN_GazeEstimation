from __future__ import annotations

import math
import re
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .common import normalize_text, to_device
from .loss_utils import compute_answer_loss


# ---------------------------------------------------------------------------
# CLIP text encoder
# ---------------------------------------------------------------------------

class CLIPTextEncoder:
    """Encode free-form text strings into L2-normalized CLIP embeddings.

    The CLIP model must be the same one used to pre-compute the label bank
    embeddings in ``data/gazefollow/label-embeds/``.
    """

    def __init__(self, model_path: str, device: torch.device) -> None:
        from transformers import CLIPModel, CLIPTokenizer  # lazy import
        print(f"[INFO] loading CLIP text encoder from: {model_path}")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts → L2-normalized [N, D] float32 tensor."""
        if not texts:
            return torch.zeros((0,), device=self.device, dtype=torch.float32)
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)  # [N, D]
        return F.normalize(feats.float(), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_object_text(text: str) -> str | None:
    """Extract label from 'Object: <label>' line. Returns None if absent."""
    m = re.search(r"(?im)^\s*object\s*:\s*(\S.*?)\s*$", str(text or ""))
    if m is None:
        return None
    val = str(m.group(1)).strip()
    return val if val else None


def parse_point(text: str) -> tuple[float, float] | None:
    num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    m = re.search(rf"(?im)^\s*point\s*:\s*({num})\s*[,\s]+\s*({num})\b", str(text or ""))
    if m is None:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return None


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
        if (
            attention_mask is not None
            and torch.is_tensor(attention_mask)
            and attention_mask.dim() >= 2
            and bsz > 0
        ):
            start = int(attention_mask[src_i].sum().item())
        else:
            start = int(input_ids.shape[1])
        new_tokens = generated_ids[i, start:]
        txt = (
            tok.decode(new_tokens, skip_special_tokens=False)
            if tok is not None
            else str(new_tokens.tolist())
        )
        txt = re.sub(r"<\|[^>]+?\|>", " ", str(txt))
        out.append(str(txt).strip())
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


def topk_similarity(
    query: torch.Tensor, bank: torch.Tensor, k: int, temperature: float
) -> list[int]:
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    loss_answer_weight: float = 1.0,
    show_tqdm: bool = True,
    desc: str = "Eval",
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    answer_sum = 0.0
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
                loss_mask_answer=batch.get("loss_mask_answer", None),
                weight_answer=float(loss_answer_weight),
            )
            loss_sum += float(losses["loss"].detach().item()) * float(bsz)
            answer_sum += float(losses["loss_answer"].detach().item()) * float(bsz)
            count += bsz
            if show_tqdm:
                it.set_postfix(loss=f"{(loss_sum / max(count, 1)):.4f}")

    if count <= 0:
        return {"loss": 0.0, "loss_answer": 0.0}
    d = float(count)
    return {
        "loss": float(loss_sum / d),
        "loss_answer": float(answer_sum / d),
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
    clip_encoder: CLIPTextEncoder | None = None,
    object_temperature: float = 0.07,
    show_tqdm: bool = True,
    desc: str = "Test",
    max_new_tokens: int = 16,
    num_beams: int = 3,
) -> dict[str, float]:
    """Generate text, parse point + object, CLIP-encode for object retrieval."""
    model.eval()
    if (not torch.is_tensor(label_embedding_bank)) or label_embedding_bank.dim() != 2:
        raise RuntimeError("label_embedding_bank must be a [N, D] tensor.")
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
    avg_l2_sum = 0.0
    min_l2_sum = 0.0
    l2_den = 0
    acc1_den = 0
    multiacc1_den = 0
    retrieval_acc1 = 0
    retrieval_acc3 = 0
    retrieval_multiacc1 = 0
    parse_fail = 0
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

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
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
            top1_texts = [
                flat[i * beam_k] if (i * beam_k) < len(flat) else ""
                for i in range(bsz)
            ]

            # ── Batch CLIP encode all parsed object texts at once ─────────
            gen_obj_texts: list[str | None] = [parse_object_text(top1_texts[i]) for i in range(bsz)]
            obj_embeddings: list[torch.Tensor | None] = [None] * bsz
            if clip_encoder is not None:
                valid_clip_indices = [i for i, t in enumerate(gen_obj_texts) if t is not None]
                valid_clip_texts = [gen_obj_texts[i] for i in valid_clip_indices]
                if valid_clip_texts:
                    encoded_batch = clip_encoder.encode(valid_clip_texts)  # [N, D]
                    for j, vi in enumerate(valid_clip_indices):
                        if j < int(encoded_batch.shape[0]):
                            obj_embeddings[vi] = encoded_batch[j]

            for i in range(bsz):
                total += 1
                pred = top1_texts[i]

                # ── Point L2 ─────────────────────────────────────────────
                pt = parse_point(pred)
                if pt is not None and isinstance(gt_points, list) and i < len(gt_points):
                    stats = l2_stats(pt, gt_points[i])
                    if stats is not None:
                        avg_l2_sum += float(stats[0])
                        min_l2_sum += float(stats[1])
                        l2_den += 1

                if float(target_valid[i].item()) <= 0.0:
                    continue
                valid_total += 1

                # ── Exact match ───────────────────────────────────────────
                tgt_norm = normalize_text(target_texts[i])
                pred_norm = normalize_text(pred)
                exact += int(pred_norm == tgt_norm)

                # ── Primary / multi GT labels ─────────────────────────────
                gt_primary: set[str] = set()
                if isinstance(target_text_label, list) and i < len(target_text_label):
                    t = normalize_text(str(target_text_label[i]))
                    if t:
                        gt_primary.add(t)

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
                if not gt_multi:
                    gt_multi = set(gt_primary)

                if gt_primary:
                    acc1_den += 1
                if gt_multi:
                    multiacc1_den += 1

                # ── Object retrieval via CLIP ─────────────────────────────
                gen_obj = gen_obj_texts[i]
                if gen_obj is None:
                    parse_fail += 1
                    continue

                if clip_encoder is not None:
                    q = obj_embeddings[i]
                    if q is None:
                        parse_fail += 1
                        continue
                    topk_idx = topk_similarity(q, bank, retrieval_top_k, object_temperature)
                    topk_text = [
                        normalize_text(labels[j])
                        for j in topk_idx
                        if 0 <= j < len(labels)
                    ]
                    topk_text = [x for x in topk_text if x]
                else:
                    # Fallback: use parsed text directly (exact string match)
                    norm_gen = normalize_text(gen_obj)
                    topk_text = [norm_gen] if norm_gen else []

                if not topk_text:
                    parse_fail += 1
                    continue

                pred_label = topk_text[0]
                if gt_primary:
                    retrieval_acc1 += int(pred_label in gt_primary)
                    retrieval_acc3 += int(
                        any(x in gt_primary for x in topk_text[: min(3, len(topk_text))])
                    )
                if gt_multi:
                    retrieval_multiacc1 += int(pred_label in gt_multi)

            if show_tqdm and l2_den > 0:
                it.set_postfix(
                    l2=f"{(avg_l2_sum / max(l2_den, 1)):.4f}",
                    acc1=f"{(retrieval_acc1 / max(acc1_den, 1)):.4f}",
                )

    if total <= 0:
        return {
            "ExactMatch": 0.0,
            "Avg L2": 0.0,
            "Min L2": 0.0,
            "PointL2": 0.0,
            "acc@1": 0.0,
            "acc@3": 0.0,
            "multiacc@1": 0.0,
            "ObjectParseFail": 0.0,
            "RetrievalAcc@1": 0.0,
            "RetrievalAcc@3": 0.0,
            "RetrievalMultiAcc@1": 0.0,
            "RetrievalDen": 0.0,
            "num_samples": 0.0,
            "num_valid_targets": 0.0,
        }

    return {
        "ExactMatch": float(exact / max(valid_total, 1)),
        "Avg L2": float(avg_l2_sum / max(l2_den, 1)),
        "Min L2": float(min_l2_sum / max(l2_den, 1)),
        "PointL2": float(avg_l2_sum / max(l2_den, 1)),
        "acc@1": float(retrieval_acc1 / max(acc1_den, 1)),
        "acc@3": float(retrieval_acc3 / max(acc1_den, 1)),
        "multiacc@1": float(retrieval_multiacc1 / max(multiacc1_den, 1)),
        "ObjectParseFail": float(parse_fail / max(valid_total, 1)),
        "RetrievalAcc@1": float(retrieval_acc1 / max(acc1_den, 1)),
        "RetrievalAcc@3": float(retrieval_acc3 / max(acc1_den, 1)),
        "RetrievalMultiAcc@1": float(retrieval_multiacc1 / max(multiacc1_den, 1)),
        "RetrievalDen": float(acc1_den),
        "num_samples": float(total),
        "num_valid_targets": float(valid_total),
    }


def print_test_metrics_table(test_metrics: dict[str, float]) -> None:
    rows = [
        ("ExactMatch", float(test_metrics.get("ExactMatch", 0.0))),
        ("Avg L2 (text)", float(test_metrics.get("Avg L2", test_metrics.get("PointL2", 0.0)))),
        ("Min L2 (text)", float(test_metrics.get("Min L2", 0.0))),
        ("acc@1", float(test_metrics.get("acc@1", 0.0))),
        ("acc@3", float(test_metrics.get("acc@3", 0.0))),
        ("multiacc@1", float(test_metrics.get("multiacc@1", 0.0))),
        ("ObjectParseFail", float(test_metrics.get("ObjectParseFail", 0.0))),
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
