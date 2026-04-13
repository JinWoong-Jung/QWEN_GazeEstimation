from __future__ import annotations

import math
import re
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from .common import normalize_text, to_device
from .loss_utils import compute_answer_loss


# ---------------------------------------------------------------------------
# Two-line stopping criteria — stop after "Point: …\nObject: …" is complete
# ---------------------------------------------------------------------------

class _TwoLineStoppingCriteria(StoppingCriteria):
    """Stop generation once ≥2 newline tokens appear in the generated portion.

    The expected output format is:
        Point:X,Y\\n
        Object:Z

    We stop as soon as a second newline is produced, preventing the model
    from appending explanations or repeated lines beyond the two-line answer.
    Works with beam search (input_ids shape: [batch*num_beams, seq_len]).
    """

    def __init__(self, prompt_len: int, newline_ids: frozenset[int]) -> None:
        self.prompt_len = int(prompt_len)
        self.newline_ids = newline_ids

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        for row in input_ids:
            generated = row[self.prompt_len :].tolist()
            nl_count = sum(1 for t in generated if t in self.newline_ids)
            if nl_count < 2:
                return False
        return True


def make_two_line_stopping_criteria(
    processor: Any,
    prompt_len: int,
) -> StoppingCriteriaList | None:
    """Build a :class:`StoppingCriteriaList` for two-line output termination.

    Returns ``None`` if the newline token IDs cannot be resolved (in which case
    the caller falls back to ``max_new_tokens`` alone).
    """
    try:
        tok = getattr(processor, "tokenizer", None) or processor
        nl_ids: set[int] = set()
        for txt in ("\n", "\n\n"):
            ids = tok.encode(txt, add_special_tokens=False)
            nl_ids.update(int(i) for i in ids if int(i) > 0)
        if not nl_ids:
            return None
        criteria = _TwoLineStoppingCriteria(
            prompt_len=int(prompt_len),
            newline_ids=frozenset(nl_ids),
        )
        return StoppingCriteriaList([criteria])
    except Exception:
        return None


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

_OBJ_SLOT_TOKEN = "<obj_emb>"
def parse_object_text(text: str) -> str | None:
    """Extract label from 'Object: <label>' line. Returns None if absent or slot token."""
    m = re.search(r"(?im)^\s*object\s*:\s*(\S.*?)\s*$", str(text or ""))
    if m is None:
        return None
    val = str(m.group(1)).strip()
    if not val or val == _OBJ_SLOT_TOKEN:
        return None
    return val


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


def retrieve_topk_from_object_texts(
    object_texts: list[str | None],
    *,
    clip_encoder: CLIPTextEncoder | None,
    bank: torch.Tensor,
    retrieval_top_k: int,
    object_temperature: float,
    bank_canonical_ids: list[int] | None = None,
) -> list[list[int]]:
    per_sample_topk_ids: list[list[int]] = [[] for _ in range(len(object_texts))]
    if clip_encoder is None or (not torch.is_tensor(bank)) or bank.dim() != 2 or int(bank.shape[0]) <= 0:
        return per_sample_topk_ids

    valid_pairs = [(i, str(txt).strip()) for i, txt in enumerate(object_texts) if str(txt or "").strip()]
    if not valid_pairs:
        return per_sample_topk_ids

    sample_indices = [i for i, _ in valid_pairs]
    texts = [txt for _, txt in valid_pairs]
    query_embs = clip_encoder.encode(texts).to(device=bank.device, dtype=torch.float32)
    if query_embs.dim() != 2 or int(query_embs.shape[0]) != len(sample_indices):
        return per_sample_topk_ids

    b = F.normalize(bank, p=2, dim=-1)
    t = max(float(object_temperature), 1e-6)
    sim = (query_embs @ b.t()) / t
    kk = min(max(1, int(retrieval_top_k)), int(b.shape[0]))
    topk_indices = torch.topk(sim, k=kk, dim=1, largest=True, sorted=True).indices.tolist()
    bank_ids = bank_canonical_ids if bank_canonical_ids is not None else list(range(int(b.shape[0])))
    for row_i, sample_i in enumerate(sample_indices):
        per_sample_topk_ids[sample_i] = [int(bank_ids[int(j)]) for j in topk_indices[row_i]]
    return per_sample_topk_ids


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
    retrieval_top_k: int = 3,
    clip_encoder: CLIPTextEncoder | None = None,
    object_temperature: float = 0.07,
    show_tqdm: bool = True,
    desc: str = "Test",
    max_new_tokens: int = 16,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    bank_canonical_ids: list[int] | None = None,
) -> dict[str, float]:
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
    _bank_ids: list[int] = bank_canonical_ids if bank_canonical_ids is not None else list(range(keep))

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
            target_label = batch.get("target_label", None)
            target_label_ids = batch.get("target_label_ids", None)
            gt_points = batch.get("gt_points", None)
            if target_valid is None:
                target_valid = torch.ones((len(target_texts),), dtype=torch.float32)
            target_valid = target_valid.to(dtype=torch.float32)

            prompt_len = int(joint["input_ids"].shape[1])
            stopping = make_two_line_stopping_criteria(processor, prompt_len)

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
            gen_obj_texts: list[str | None] = [parse_object_text(preds[i]) if i < len(preds) else None for i in range(bsz)]
            per_sample_topk_ids = retrieve_topk_from_object_texts(
                gen_obj_texts,
                clip_encoder=clip_encoder,
                bank=bank,
                retrieval_top_k=int(retrieval_top_k),
                object_temperature=float(object_temperature),
                bank_canonical_ids=_bank_ids,
            )

            for i in range(bsz):
                total += 1
                pred = preds[i] if i < len(preds) else ""

                pt = parse_point(pred)
                if pt is not None and isinstance(gt_points, list) and i < len(gt_points):
                    stats = l2_stats(pt, gt_points[i])
                    if stats is not None:
                        avg_l2_sum += float(stats[0])
                        min_l2_sum += float(stats[1])
                        l2_den += 1

                if float(target_valid[i].item()) > 0.0:
                    valid_total += 1
                    exact += int(normalize_text(pred) == normalize_text(target_texts[i]))

                gt_id = -1
                if torch.is_tensor(target_label) and i < int(target_label.shape[0]):
                    gt_id = int(target_label[i].item())

                gt_multi_ids: set[int] = set()
                if isinstance(target_label_ids, list) and i < len(target_label_ids):
                    gt_multi_ids = {int(x) for x in target_label_ids[i] if int(x) >= 0}
                if not gt_multi_ids and gt_id >= 0:
                    gt_multi_ids = {gt_id}

                if gt_id >= 0:
                    acc1_den += 1
                if gt_multi_ids:
                    multiacc1_den += 1

                if gen_obj_texts[i] is None:
                    if float(target_valid[i].item()) > 0.0:
                        parse_fail += 1
                    continue

                topk_label_ids = per_sample_topk_ids[i]
                if not topk_label_ids:
                    if float(target_valid[i].item()) > 0.0:
                        parse_fail += 1
                    continue

                pred_label_id = int(topk_label_ids[0])
                if gt_id >= 0:
                    retrieval_acc1 += int(pred_label_id == gt_id)
                    retrieval_acc3 += int(gt_id in topk_label_ids[:3])
                if gt_multi_ids:
                    retrieval_multiacc1 += int(pred_label_id in gt_multi_ids)

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
        "num_samples": float(total),
        "num_valid_targets": float(valid_total),
    }


def collect_generation_samples(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    processor: Any,
    label_embedding_bank: torch.Tensor,
    retrieval_label_texts: list[str],
    retrieval_top_k: int = 3,
    clip_encoder: CLIPTextEncoder | None = None,
    object_temperature: float = 0.07,
    show_tqdm: bool = True,
    desc: str = "GenerationPreview",
    max_new_tokens: int = 16,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    max_samples: int = 8,
    bank_canonical_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    limit = max(0, int(max_samples))
    if limit <= 0:
        return []

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
    _bank_ids: list[int] = bank_canonical_ids if bank_canonical_ids is not None else list(range(keep))

    previews: list[dict[str, Any]] = []
    beam_k = max(1, int(num_beams))

    with torch.no_grad():
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not show_tqdm)
        for batch in it:
            joint = to_device(batch["joint_inputs"], device=device)
            target_texts = [str(x) for x in batch.get("target_text", [])]
            target_valid = batch.get("target_text_valid", None)
            target_label = batch.get("target_label", None)
            target_label_ids = batch.get("target_label_ids", None)
            gt_points = batch.get("gt_points", None)
            prompt_texts = [str(x) for x in batch.get("text_input", [])]
            target_label_texts = [str(x) for x in batch.get("target_label_text", [])]
            image_rels = [str(x) for x in batch.get("image_rel", [])]
            if target_valid is None:
                target_valid = torch.ones((len(target_texts),), dtype=torch.float32)
            target_valid = target_valid.to(dtype=torch.float32)

            prompt_len = int(joint["input_ids"].shape[1])
            stopping = make_two_line_stopping_criteria(processor, prompt_len)
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
            gen_obj_texts: list[str | None] = [parse_object_text(preds[i]) if i < len(preds) else None for i in range(bsz)]
            per_sample_topk_ids = retrieve_topk_from_object_texts(
                gen_obj_texts,
                clip_encoder=clip_encoder,
                bank=bank,
                retrieval_top_k=int(retrieval_top_k),
                object_temperature=float(object_temperature),
                bank_canonical_ids=_bank_ids,
            )

            for i in range(bsz):
                pred = preds[i] if i < len(preds) else ""
                pt = parse_point(pred)
                stats = None
                serialized_gt_points: list[list[float]] = []
                if isinstance(gt_points, list) and i < len(gt_points):
                    stats = l2_stats(pt, gt_points[i]) if pt is not None else None
                    if torch.is_tensor(gt_points[i]) and gt_points[i].numel() >= 2:
                        pts = gt_points[i].detach().cpu().to(dtype=torch.float32).view(-1, 2)
                        serialized_gt_points = [
                            [float(pts[j, 0].item()), float(pts[j, 1].item())]
                            for j in range(int(pts.shape[0]))
                        ]

                gt_id = -1
                if torch.is_tensor(target_label) and i < int(target_label.shape[0]):
                    gt_id = int(target_label[i].item())

                gt_multi_ids: set[int] = set()
                if isinstance(target_label_ids, list) and i < len(target_label_ids):
                    gt_multi_ids = {int(x) for x in target_label_ids[i] if int(x) >= 0}
                if not gt_multi_ids and gt_id >= 0:
                    gt_multi_ids = {gt_id}

                topk_ids = per_sample_topk_ids[i]
                topk_labels = [labels[int(idx)] for idx in topk_ids if 0 <= int(idx) < len(labels)]
                target_is_valid = (
                    bool(float(target_valid[i].item()) > 0.0)
                    if i < int(target_valid.numel())
                    else False
                )
                target_text = target_texts[i] if i < len(target_texts) else ""
                previews.append(
                    {
                        "sample_index": int(len(previews)),
                        "image_rel": image_rels[i] if i < len(image_rels) else "",
                        "prompt_text": prompt_texts[i] if i < len(prompt_texts) else "",
                        "target_text": target_text,
                        "target_text_valid": target_is_valid,
                        "generated_text": pred,
                        "exact_match": (
                            bool(normalize_text(pred) == normalize_text(target_text))
                            if target_is_valid
                            else None
                        ),
                        "parsed_point": [float(pt[0]), float(pt[1])] if pt is not None else None,
                        "gt_points": serialized_gt_points,
                        "avg_l2": (float(stats[0]) if stats is not None else None),
                        "min_l2": (float(stats[1]) if stats is not None else None),
                        "parsed_object_text": gen_obj_texts[i],
                        "target_label_text": (
                            target_label_texts[i] if i < len(target_label_texts) else ""
                        ),
                        "target_label_id": gt_id,
                        "target_label_ids": sorted(gt_multi_ids),
                        "retrieved_topk_ids": topk_ids,
                        "retrieved_topk_labels": topk_labels,
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
        print(f"[SAMPLE {idx}] image={image_rel}")
        print(f"  prompt   : {sample.get('prompt_text', '')}")
        print(f"  target   : {sample.get('target_text', '')}")
        print(f"  generated: {sample.get('generated_text', '')}")
        print(f"  exact    : {sample.get('exact_match', None)}")
        print(f"  gt_label : {sample.get('target_label_text', '')} ({sample.get('target_label_id', -1)})")
        print(f"  pred_obj : {sample.get('parsed_object_text', None)}")
        print(f"  topk_obj : {sample.get('retrieved_topk_labels', [])}")
        print(f"  gt_points: {sample.get('gt_points', [])}")
        print(f"  pred_pt  : {sample.get('parsed_point', None)}")
        print(f"  avg_l2   : {sample.get('avg_l2', None)}")
        print(f"  min_l2   : {sample.get('min_l2', None)}")


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
