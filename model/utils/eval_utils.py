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
from .point_tokens import parse_point_token_pair, render_point_text_human


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


def clean_retrieval_text(text: str, max_tokens: int = 6) -> str:
    """Clean a raw generated object string into a compact retrieval query.

    Applied in order:

    1. Lowercase and normalize whitespace.
    2. Strip leading/trailing punctuation.
    3. Remove consecutive duplicate tokens  (``"laptop laptop"`` → ``"laptop"``).
    4. Remove trailing repeated n-grams     (``"a b a b"`` → ``"a b"``).
    5. Detect cyclic repetition from pos-0  (``"a b c a b c"`` → ``"a b c"``).
    6. Hard-truncate to *max_tokens* tokens as a final guard.

    The input string is never mutated; a new cleaned string is returned.
    Empty / whitespace-only input returns ``""``.
    """
    if not text:
        return ""

    # 1. Lowercase + collapse whitespace
    t = re.sub(r"\s+", " ", str(text).lower().strip())

    # 2. Strip leading/trailing punctuation (preserve internal hyphens/apostrophes)
    t = t.strip(".,;:!?\"'()[]{}/\\")
    t = t.strip()
    if not t:
        return ""

    # 3. Tokenize on whitespace and strip per-token punctuation
    #    (e.g. "chair." mid-string after splitting "a chair. a chair.")
    _PUNCT = ".,;:!?\"'()[]{}/\\"
    tokens: list[str] = [tok.strip(_PUNCT) for tok in t.split()]
    tokens = [tok for tok in tokens if tok]
    if not tokens:
        return ""

    # 4. Remove consecutive duplicate tokens
    deduped: list[str] = [tokens[0]]
    for tok in tokens[1:]:
        if tok != deduped[-1]:
            deduped.append(tok)
    tokens = deduped

    # 5. Detect cyclic repetition from position 0 (run first, before trailing-ngram).
    #    Find minimal period p where tokens[p:2p] == tokens[0:p].
    #    e.g.  ["a","b","c","a","b","c"] → p=3 hit → ["a","b","c"]
    #    e.g.  ["a","b","a","b","a"]     → p=2 hit → ["a","b"]
    for p in range(1, len(tokens) // 2 + 1):
        if tokens[p : 2 * p] == tokens[0:p]:
            tokens = tokens[:p]
            break

    # 6. Remove trailing repeated n-grams (mop up residual):
    #    repeat while last n tokens == n tokens immediately before them.
    #    e.g.  ["x","a","b","a","b"] (p=2 didn't fire) → n=2 hit → ["x","a","b"]
    changed = True
    while changed and len(tokens) >= 2:
        changed = False
        for n in range(1, len(tokens) // 2 + 1):
            if tokens[-n:] == tokens[-(2 * n) : -n]:
                tokens = tokens[:-n]
                changed = True
                break

    # 7. Hard truncation
    tokens = tokens[:max_tokens]

    return " ".join(tokens)


def expand_retrieval_candidates(cleaned: str) -> list[str]:
    """Expand a cleaned phrase into a de-duplicated set of retrieval query candidates.

    Given a *cleaned* string (output of :func:`clean_retrieval_text`), returns an
    ordered list of candidate strings for CLIP retrieval.  Each candidate is tried
    against the label bank and the one with the highest cosine similarity wins.

    Candidates produced (in this order, duplicates silently dropped):
    1. The full cleaned phrase as-is.
    2. Every unigram  (1-token window).
    3. Every bigram   (2-token window), only when len >= 2.
    4. Every trigram  (3-token window), only when len >= 3.

    All candidates are already lowercase because *cleaned* is lowercase.
    Empty *cleaned* input returns an empty list.

    Example
    -------
    >>> expand_retrieval_candidates("laptop screen")
    ['laptop screen', 'laptop', 'screen']

    >>> expand_retrieval_candidates("computer laptop screen")
    ['computer laptop screen', 'computer', 'laptop', 'screen',
     'computer laptop', 'laptop screen',
     'computer laptop screen']  # trigram == full phrase → deduped away
    """
    if not cleaned:
        return []

    tokens = cleaned.split()
    if not tokens:
        return []

    seen: set[str] = set()
    out: list[str] = []

    def _add(s: str) -> None:
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    # 1. Full phrase
    _add(cleaned)

    # 2. Unigrams
    for tok in tokens:
        _add(tok)

    # 3. Bigrams
    for i in range(len(tokens) - 1):
        _add(tokens[i] + " " + tokens[i + 1])

    # 4. Trigrams
    for i in range(len(tokens) - 2):
        _add(tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2])

    return out


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
    m_tok = re.search(
        r"(?im)^\s*point\s*:\s*(<pt\d+_\d+>)\s*[,\s]+\s*(<pt\d+_\d+>)\s*$",
        str(text or ""),
    )
    if m_tok is not None:
        parsed = parse_point_token_pair(m_tok.group(1), m_tok.group(2))
        if parsed is not None:
            return parsed

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
    retrieval_top_k: int = 3,
    clip_encoder: CLIPTextEncoder | None = None,
    object_temperature: float = 0.07,
    show_tqdm: bool = True,
    desc: str = "Test",
    max_new_tokens: int = 16,
    num_beams: int = 3,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    bank_canonical_ids: list[int] | None = None,
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
    _bank_ids: list[int] = bank_canonical_ids if bank_canonical_ids is not None else list(range(keep))
    # Precompute mask: True where bank entry has a valid vocab ID (cid >= 0).
    # Used to skip cid=-1 entries in argmax/topk so pred_label_id is always mappable.
    _valid_bank_mask = torch.tensor([bid >= 0 for bid in _bank_ids], dtype=torch.bool, device=device)
    _n_valid_bank = int(_valid_bank_mask.sum().item())

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

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                generated_ids = model.generate(
                    joint_inputs=joint,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=beam_k,
                    num_return_sequences=beam_k,
                    repetition_penalty=max(float(repetition_penalty), 1.0),
                    no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
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

            # ── Parse → clean → expand → batch CLIP encode ───────────────
            gen_obj_texts: list[str | None] = [parse_object_text(top1_texts[i]) for i in range(bsz)]

            # Precompute per-sample retrieval results (filled below if clip_encoder is set)
            per_sample_selected_bank_idx: list[int | None] = [None] * bsz
            per_sample_topk_bank_indices: list[list[int]] = [[] for _ in range(bsz)]

            if clip_encoder is not None:
                # Expand each raw object text into retrieval query candidates
                per_sample_candidates: list[list[str]] = []
                for raw in gen_obj_texts:
                    if raw is None:
                        per_sample_candidates.append([])
                    else:
                        cleaned = clean_retrieval_text(raw)
                        per_sample_candidates.append(expand_retrieval_candidates(cleaned))

                # Flatten all candidates for a single batch encode call
                all_candidates: list[str] = []
                sample_cand_ranges: list[tuple[int, int]] = []
                for cands in per_sample_candidates:
                    start = len(all_candidates)
                    all_candidates.extend(cands)
                    sample_cand_ranges.append((start, len(all_candidates)))

                if all_candidates and _n_valid_bank > 0:
                    all_cand_embs = clip_encoder.encode(all_candidates)  # [total_cands, D]
                    kk = min(max(1, retrieval_top_k), _n_valid_bank)
                    for i in range(bsz):
                        s, e = sample_cand_ranges[i]
                        if s == e:
                            continue
                        cand_embs = all_cand_embs[s:e]       # [n_cands, D]
                        sim = cand_embs @ bank.t()            # [n_cands, bank_n]
                        per_bank_max, _ = sim.max(dim=0)      # [bank_n]
                        # Mask out bank entries with no valid vocab mapping (cid=-1)
                        # so argmax/topk only selects entries that can match a gt label id.
                        masked = per_bank_max.masked_fill(~_valid_bank_mask, float("-inf"))
                        per_sample_selected_bank_idx[i] = int(masked.argmax().item())
                        per_sample_topk_bank_indices[i] = [
                            int(x) for x in masked.topk(
                                k=kk, largest=True, sorted=True
                            ).indices.tolist()
                        ]

            for i in range(bsz):
                total += 1
                pred = top1_texts[i]

                # ── Point L2 ─────────────────────────────────────────────
                pt = parse_point(pred)
                # Only count predictions in the normalized [0, 1] range.
                # Out-of-range values (e.g. pixel coords output by an untrained model)
                # would inflate Avg L2 to thousands and make the metric meaningless.
                if (
                    pt is not None
                    and 0.0 <= pt[0] <= 1.0
                    and 0.0 <= pt[1] <= 1.0
                    and isinstance(gt_points, list)
                    and i < len(gt_points)
                ):
                    stats = l2_stats(pt, gt_points[i])
                    if stats is not None:
                        avg_l2_sum += float(stats[0])
                        min_l2_sum += float(stats[1])
                        l2_den += 1

                # ── Exact match (target_text_valid 기준) ──────────────────
                if float(target_valid[i].item()) > 0.0:
                    valid_total += 1
                    tgt_norm = normalize_text(target_texts[i])
                    pred_norm = normalize_text(pred)
                    exact += int(pred_norm == tgt_norm)

                # ── ID-space GT labels ────────────────────────────────────
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

                # ── Object retrieval via CLIP ─────────────────────────────
                # Only count parse_fail for samples with valid target text
                gen_obj = gen_obj_texts[i]
                if gen_obj is None:
                    if float(target_valid[i].item()) > 0.0:
                        parse_fail += 1
                    continue

                if clip_encoder is not None:
                    selected_bank_idx = per_sample_selected_bank_idx[i]
                    if selected_bank_idx is None:
                        if float(target_valid[i].item()) > 0.0:
                            parse_fail += 1
                        continue
                    topk_label_ids = [_bank_ids[idx] for idx in per_sample_topk_bank_indices[i]]
                else:
                    topk_label_ids = []

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
    num_beams: int = 3,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    max_samples: int = 8,
    bank_canonical_ids: list[int] | None = None,
    point_decimals: int = 4,
) -> list[dict[str, Any]]:
    """Collect a small number of human-readable generation previews."""
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

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                generated_ids = model.generate(
                    joint_inputs=joint,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=beam_k,
                    num_return_sequences=beam_k,
                    repetition_penalty=max(float(repetition_penalty), 1.0),
                    no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
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

            # Match run_test_metrics retrieval path exactly:
            # parse -> clean -> candidate expansion -> batch CLIP encode -> per-bank max
            gen_obj_texts: list[str | None] = [parse_object_text(top1_texts[i]) for i in range(bsz)]
            per_sample_selected_bank_idx: list[int | None] = [None] * bsz
            per_sample_topk_bank_indices: list[list[int]] = [[] for _ in range(bsz)]

            if clip_encoder is not None:
                per_sample_candidates: list[list[str]] = []
                for raw in gen_obj_texts:
                    if raw is None:
                        per_sample_candidates.append([])
                    else:
                        cleaned = clean_retrieval_text(raw)
                        per_sample_candidates.append(expand_retrieval_candidates(cleaned))

                all_candidates: list[str] = []
                sample_cand_ranges: list[tuple[int, int]] = []
                for cands in per_sample_candidates:
                    start = len(all_candidates)
                    all_candidates.extend(cands)
                    sample_cand_ranges.append((start, len(all_candidates)))

                if all_candidates:
                    all_cand_embs = clip_encoder.encode(all_candidates)  # [total_cands, D]
                    kk = min(max(1, retrieval_top_k), int(bank.shape[0]))
                    for i in range(bsz):
                        s, e = sample_cand_ranges[i]
                        if s == e:
                            continue
                        cand_embs = all_cand_embs[s:e]      # [n_cands, D]
                        sim = cand_embs @ bank.t()           # [n_cands, bank_n]
                        per_bank_max, _ = sim.max(dim=0)     # [bank_n]
                        per_sample_selected_bank_idx[i] = int(per_bank_max.argmax().item())
                        per_sample_topk_bank_indices[i] = [
                            int(x) for x in per_bank_max.topk(
                                k=kk, largest=True, sorted=True
                            ).indices.tolist()
                        ]

            for i in range(bsz):
                pred = top1_texts[i]
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

                topk_ids: list[int] = []
                topk_bank_indices: list[int] = []
                gen_obj = gen_obj_texts[i]
                if gen_obj is not None and clip_encoder is not None:
                    selected_bank_idx = per_sample_selected_bank_idx[i]
                    if selected_bank_idx is not None:
                        topk_bank_indices = per_sample_topk_bank_indices[i]
                        topk_ids = [_bank_ids[idx] for idx in topk_bank_indices]

                target_is_valid = (
                    bool(float(target_valid[i].item()) > 0.0)
                    if i < int(target_valid.numel())
                    else False
                )
                target_text_raw = target_texts[i] if i < len(target_texts) else ""
                generated_text_raw = pred
                target_text = render_point_text_human(target_text_raw, point_decimals=point_decimals)
                generated_text = render_point_text_human(
                    generated_text_raw,
                    point_decimals=point_decimals,
                )
                previews.append(
                    {
                        "sample_index": int(len(previews)),
                        "image_rel": image_rels[i] if i < len(image_rels) else "",
                        "prompt_text": prompt_texts[i] if i < len(prompt_texts) else "",
                        "target_text": target_text,
                        "target_text_raw": target_text_raw,
                        "target_text_valid": target_is_valid,
                        "generated_text": generated_text,
                        "generated_text_raw": generated_text_raw,
                        "exact_match": (
                            bool(normalize_text(pred) == normalize_text(target_text_raw))
                            if target_is_valid
                            else None
                        ),
                        "parsed_point": (
                            [float(pt[0]), float(pt[1])]
                            if pt is not None
                            else None
                        ),
                        "gt_points": serialized_gt_points,
                        "avg_l2": (float(stats[0]) if stats is not None else None),
                        "min_l2": (float(stats[1]) if stats is not None else None),
                        "parsed_object_text": gen_obj,
                        "target_label_text": (
                            target_label_texts[i] if i < len(target_label_texts) else ""
                        ),
                        "target_label_id": gt_id,
                        "target_label_ids": sorted(gt_multi_ids),
                        "retrieved_topk_ids": topk_ids,
                        "retrieved_topk_labels": [
                            labels[idx] for idx in topk_bank_indices if 0 <= int(idx) < len(labels)
                        ],
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
