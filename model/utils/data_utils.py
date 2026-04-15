from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps


def is_normalized_point(x: float, y: float) -> bool:
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def sanitize_bbox_pixels(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = bbox
    x1, x2 = sorted((xmin, xmax))
    y1, y2 = sorted((ymin, ymax))

    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))

    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)

    x1i, y1i = int(round(x1)), int(round(y1))
    x2i, y2i = int(round(x2)), int(round(y2))
    x1i = max(0, min(x1i, width - 1))
    y1i = max(0, min(y1i, height - 1))
    x2i = max(x1i + 1, min(x2i, width))
    y2i = max(y1i + 1, min(y2i, height))
    return x1i, y1i, x2i, y2i


def build_prompt(
    bbox_norm: tuple[float, float, float, float],
    prompt_template: str,
    prompt_text: str = "",
    *,
    point_decimals: int | None = None,
) -> str:
    xmin, ymin, xmax, ymax = bbox_norm
    format_vars: dict[str, Any] = {
        "xmin": float(xmin),
        "ymin": float(ymin),
        "xmax": float(xmax),
        "ymax": float(ymax),
    }
    if point_decimals is not None:
        format_vars["point_decimals"] = int(point_decimals)
    if prompt_template.strip():
        return prompt_template.format(**format_vars)
    if prompt_text.strip():
        try:
            return prompt_text.format(**format_vars)
        except Exception:
            return prompt_text
    raise ValueError(
        "Prompt is empty. Set `prompt.prompt_text` or `prompt.prompt_template` in config.yaml."
    )


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def expand_bbox(
    bbox_px: tuple[float, float, float, float],
    width: int,
    height: int,
    scale: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_px
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(1.0, (x2 - x1) * float(scale))
    bh = max(1.0, (y2 - y1) * float(scale))
    nx1 = max(0.0, cx - 0.5 * bw)
    ny1 = max(0.0, cy - 0.5 * bh)
    nx2 = min(float(width), cx + 0.5 * bw)
    ny2 = min(float(height), cy + 0.5 * bh)
    return nx1, ny1, nx2, ny2


def safe_crop(
    scene: Image.Image,
    gaze_x: float,
    gaze_y: float,
    bbox_px: tuple[float, float, float, float],
    p: float = 0.5,
) -> tuple[Image.Image, float, float, tuple[float, float, float, float]]:
    if random.random() >= float(p):
        return scene, gaze_x, gaze_y, bbox_px

    w, h = scene.size
    gx = clamp01(gaze_x) * (w - 1)
    gy = clamp01(gaze_y) * (h - 1)
    x1, y1, x2, y2 = bbox_px

    min_x = max(0.0, min(gx, x1))
    min_y = max(0.0, min(gy, y1))
    max_x = min(float(w - 1), max(gx, x2))
    max_y = min(float(h - 1), max(gy, y2))

    left = random.uniform(0.0, float(min_x))
    top = random.uniform(0.0, float(min_y))
    right = random.uniform(float(max_x), float(w - 1))
    bottom = random.uniform(float(max_y), float(h - 1))

    crop_w = max(2.0, right - left)
    crop_h = max(2.0, bottom - top)
    if crop_w < 2.0 or crop_h < 2.0:
        return scene, gaze_x, gaze_y, bbox_px

    l = int(max(0, min(w - 2, round(left))))
    t = int(max(0, min(h - 2, round(top))))
    r = int(max(l + 2, min(w, round(right + 1))))
    b = int(max(t + 2, min(h, round(bottom + 1))))
    if r <= l + 1 or b <= t + 1:
        return scene, gaze_x, gaze_y, bbox_px

    cropped = scene.crop((l, t, r, b))
    new_w, new_h = cropped.size

    gx_new = (gx - l) / max(new_w - 1, 1)
    gy_new = (gy - t) / max(new_h - 1, 1)
    if (gx_new < 0.0) or (gx_new > 1.0) or (gy_new < 0.0) or (gy_new > 1.0):
        return scene, gaze_x, gaze_y, bbox_px

    nx1 = max(0.0, min(float(new_w), x1 - l))
    ny1 = max(0.0, min(float(new_h), y1 - t))
    nx2 = max(0.0, min(float(new_w), x2 - l))
    ny2 = max(0.0, min(float(new_h), y2 - t))
    if nx2 <= nx1:
        nx2 = min(float(new_w), nx1 + 1.0)
    if ny2 <= ny1:
        ny2 = min(float(new_h), ny1 + 1.0)

    return cropped, clamp01(gx_new), clamp01(gy_new), (nx1, ny1, nx2, ny2)


def maybe_hflip(
    scene: Image.Image,
    gaze_x: float,
    bbox_px: tuple[float, float, float, float],
    p: float = 0.5,
) -> tuple[Image.Image, float, tuple[float, float, float, float]]:
    if random.random() >= float(p):
        return scene, gaze_x, bbox_px
    w, _ = scene.size
    flipped = ImageOps.mirror(scene)
    x1, y1, x2, y2 = bbox_px
    nx1 = float(w) - x2
    nx2 = float(w) - x1
    ngx = 1.0 - float(gaze_x)
    return flipped, clamp01(ngx), (nx1, y1, nx2, y2)


def color_jitter(scene: Image.Image, p: float = 0.8) -> Image.Image:
    if random.random() >= float(p):
        return scene
    out = scene
    b = random.uniform(0.5, 1.5)
    c = random.uniform(0.5, 1.5)
    s = random.uniform(0.0, 1.5)
    out = ImageEnhance.Brightness(out).enhance(b)
    out = ImageEnhance.Contrast(out).enhance(c)
    out = ImageEnhance.Color(out).enhance(s)
    return out


def apply_train_augmentation(
    scene: Image.Image,
    gaze_x: float,
    gaze_y: float,
    bbox_px: tuple[float, float, float, float],
) -> tuple[Image.Image, float, float, tuple[float, float, float, float]]:
    w, h = scene.size
    bx = sanitize_bbox_pixels(bbox_px, width=w, height=h)
    bbox_f = (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]))

    if random.random() < 0.5:
        scale = random.uniform(1.0, 1.2)
        bbox_f = expand_bbox(bbox_f, width=w, height=h, scale=scale)

    scene, gaze_x, gaze_y, bbox_f = safe_crop(
        scene=scene,
        gaze_x=gaze_x,
        gaze_y=gaze_y,
        bbox_px=bbox_f,
        p=0.5,
    )
    scene, gaze_x, bbox_f = maybe_hflip(
        scene=scene,
        gaze_x=gaze_x,
        bbox_px=bbox_f,
        p=0.5,
    )
    scene = color_jitter(scene, p=0.8)

    nw, nh = scene.size
    x1, y1, x2, y2 = sanitize_bbox_pixels(bbox_f, width=nw, height=nh)
    gaze_x = clamp01(gaze_x)
    gaze_y = clamp01(gaze_y)
    return scene, gaze_x, gaze_y, (float(x1), float(y1), float(x2), float(y2))


def load_vocab2id(vocab2id_path: Path | None) -> tuple[dict[str, int], dict[str, int]]:
    if vocab2id_path is None or (not vocab2id_path.exists()):
        return {}, {}
    obj = torch.load(vocab2id_path) if str(vocab2id_path).endswith(".pt") else None
    if obj is None:
        import json
        obj = json.loads(vocab2id_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return {}, {}

    v: dict[str, int] = {}
    for k, val in obj.items():
        try:
            idx = int(val)
        except Exception:
            continue
        v[str(k)] = idx
    vl = {k.lower(): i for k, i in v.items()}
    return v, vl


def vocab_id(
    label_text: str,
    vocab2id: dict[str, int] | None = None,
    vocab2id_lower: dict[str, int] | None = None,
) -> int:
    if not label_text:
        return -1
    txt = str(label_text).strip()
    if not txt:
        return -1
    v = vocab2id or {}
    vl = vocab2id_lower or {}
    if txt in v:
        return int(v[txt])
    t2 = txt.lower()
    if t2 in vl:
        return int(vl[t2])
    return -1


def direct_vocab_id(
    label_text: str,
    vocab2id: dict[str, int] | None = None,
) -> int:
    if not label_text:
        return -1
    txt = str(label_text).strip()
    if not txt:
        return -1
    v = vocab2id or {}
    return int(v.get(txt, -1))


def clean_label(txt: str) -> str:
    return " ".join(str(txt or "").strip().split())


# ---------------------------------------------------------------------------
# Canonical noun-phrase normaliser for training / answer-template targets
# ---------------------------------------------------------------------------

# Leading articles to strip from the start of a label.
_ARTICLE_PREFIXES: tuple[str, ...] = ("a ", "an ", "the ")

# Prepositions that mark the start of a trailing prepositional phrase.
# We drop everything from the first preposition token onward so that
# "hand on mouse" → "hand" and "glass of water" → "glass".
# Note: "of" is intentionally included even though it appears in some
# compound labels (e.g. "pair of scissors") because we already apply a
# 3-token hard-truncation below, which handles those cases gracefully.
_PREP_TOKENS: frozenset[str] = frozenset({
    "on", "in", "at", "by", "with", "near", "from", "to", "into", "onto",
    "inside", "outside", "over", "under", "through", "of", "for", "about",
    "up", "down", "beside", "behind", "between", "among", "around",
})

# Maximum number of whitespace-delimited tokens kept in the canonical form.
_CANONICAL_MAX_TOKENS: int = 3


def canonicalize_label_text(raw: str) -> str:
    """Normalise a raw label string into a compact 1-3 word canonical noun phrase.

    Applied steps (in order):

    1. Collapse whitespace and strip leading/trailing spaces.
    2. Strip a single leading article (a / an / the), case-insensitive.
    3. Drop all tokens from the first preposition onward (removes
       trailing prepositional phrases such as "on the table").
    4. Hard-truncate to at most :data:`_CANONICAL_MAX_TOKENS` tokens.
    5. Re-join and return the result.

    The original case is preserved (not lowercased) so the result can be
    matched directly against ``vocab2id`` whose keys retain original case.
    An empty input or a label that reduces to nothing returns the
    original (stripped) value unchanged so the caller can still look up
    the ID.

    Examples
    --------
    >>> canonicalize_label_text("the laptop computer")
    'laptop computer'
    >>> canonicalize_label_text("a hand on mouse")
    'hand'
    >>> canonicalize_label_text("glass of water")
    'glass'
    >>> canonicalize_label_text("baseball bat")
    'baseball bat'
    >>> canonicalize_label_text("television")
    'television'
    """
    txt = " ".join(str(raw or "").strip().split())
    if not txt:
        return txt

    # 1. Strip leading article (case-insensitive compare, preserve original case).
    lower = txt.lower()
    for art in _ARTICLE_PREFIXES:
        if lower.startswith(art):
            txt = txt[len(art):].lstrip()
            lower = txt.lower()
            break  # only strip one article

    if not txt:
        return str(raw or "").strip()

    # 2. Tokenise and drop from first preposition onward.
    tokens: list[str] = txt.split()
    keep: list[str] = []
    for tok in tokens:
        if tok.lower() in _PREP_TOKENS:
            break
        keep.append(tok)

    if not keep:
        # Preposition was the very first token (unusual); fall back to full phrase.
        keep = tokens

    # 3. Hard-truncate to _CANONICAL_MAX_TOKENS tokens.
    canonical = " ".join(keep[:_CANONICAL_MAX_TOKENS])
    return canonical if canonical else str(raw or "").strip()


def annotation_candidates(raw: str) -> list[str]:
    base = clean_label(raw)
    if not base:
        return []

    out: list[str] = []
    seen: set[str] = set()

    def _push(v: str) -> None:
        t = clean_label(v)
        if not t:
            return
        key = t.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(t)
        tl = t.lower()
        for pref in ("a ", "an ", "the "):
            if tl.startswith(pref):
                t2 = clean_label(t[len(pref) :])
                if t2:
                    k2 = t2.lower()
                    if k2 not in seen:
                        seen.add(k2)
                        out.append(t2)

    _push(base)
    parts = re.split(r"[|/;,]+", base)
    for part in parts:
        _push(part)
        if "-" in part:
            for sub in part.split("-"):
                _push(sub)
    return out


def split_aliases(split_name: str | None) -> set[str]:
    if split_name is None:
        return set()
    s = str(split_name).strip().lower()
    if not s:
        return set()
    if s in {"val", "validation"}:
        return {"val", "validation"}
    return {s}


def build_fallback_map(
    fallback_csvs: list[Path | None] | None,
    fallback_text_key: str = "annotation",
    split_name: str | None = None,
) -> dict[tuple[str, int], list[str]]:
    out: dict[tuple[str, int], list[str]] = {}
    if not fallback_csvs:
        return out

    split_filter = split_aliases(split_name)
    for csv_path in fallback_csvs:
        if csv_path is None or (not csv_path.exists()):
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    path = str(row["path"]).strip()
                    sample_id = int(float(row["id"]))
                except Exception:
                    continue
                if not path:
                    continue
                if split_filter:
                    row_split = str(row.get("split", "")).strip().lower()
                    if row_split not in split_filter:
                        continue
                text_raw = str(row.get(fallback_text_key, "")).strip()
                cands = annotation_candidates(text_raw)
                if not cands:
                    continue
                key = (path, sample_id)
                if key not in out:
                    out[key] = []
                for cand in cands:
                    if cand not in out[key]:
                        out[key].append(cand)
    return out


def build_embed_id_mapper(
    vocab2id: dict[str, int],
    label_embed_dir: Path | None,
    label_emb_dim: int = 512,
) -> Callable[[str], int] | None:
    bank = build_vocab_embedding_matrix(
        vocab2id=vocab2id,
        label_embed_dir=label_embed_dir,
        label_emb_dim=int(label_emb_dim),
        normalize=True,
    )
    if (not torch.is_tensor(bank)) or bank.dim() != 2 or int(bank.shape[0]) <= 0:
        return None
    bank = bank.to(dtype=torch.float32)
    valid = bank.norm(dim=1).gt(0)
    if int(valid.sum().item()) <= 0:
        return None
    valid_ids = [int(x) for x in torch.nonzero(valid, as_tuple=False).flatten().tolist()]
    valid_bank = bank[valid].contiguous()
    embed_dir = label_embed_dir if (label_embed_dir is not None and label_embed_dir.exists()) else None
    if embed_dir is None:
        return None

    cache: dict[str, int] = {}

    def _map(label_text: str) -> int:
        txt = clean_label(label_text)
        if not txt:
            return -1
        if txt in cache:
            return int(cache[txt])
        emb_path = embed_dir / f"{txt}-emb.pt"
        if not emb_path.exists():
            cache[txt] = -1
            return -1
        try:
            vec = torch.load(emb_path, map_location="cpu")
            if not torch.is_tensor(vec):
                cache[txt] = -1
                return -1
            q = vec.to(dtype=torch.float32).flatten()
            if int(q.numel()) != int(valid_bank.shape[1]):
                cache[txt] = -1
                return -1
            q = F.normalize(q.unsqueeze(0), p=2, dim=-1)
            sim = q @ valid_bank.t()
            j = int(torch.argmax(sim, dim=1).item())
            mapped = int(valid_ids[j]) if 0 <= j < len(valid_ids) else -1
            cache[txt] = int(mapped)
            return int(mapped)
        except Exception:
            cache[txt] = -1
            return -1

    return _map


def load_label_map(
    labels_csv: Path | None,
    vocab2id: dict[str, int] | None = None,
    vocab2id_lower: dict[str, int] | None = None,
    text_key: str = "gaze_pseudo_label",
    fallback_csvs: list[Path | None] | None = None,
    fallback_text_key: str = "annotation",
    split_name: str | None = None,
    label_embed_dir: Path | None = None,
    label_emb_dim: int = 512,
    use_embed_fallback: bool = True,
) -> tuple[dict[tuple[str, int], int], dict[str, int]]:
    stats = {
        "rows": 0,
        "mapped": 0,
        "missing_text": 0,
        "unknown_text": 0,
        "primary_mapped": 0,
        "primary_unknown": 0,
        "embed_fallback_considered": 0,
        "embed_fallback_mapped": 0,
        "fallback_considered": 0,
        "fallback_mapped": 0,
    }
    if labels_csv is None or (not labels_csv.exists()):
        return {}, stats

    v = vocab2id or {}
    out: dict[tuple[str, int], int] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                path = str(row["path"]).strip()
                sample_id = int(float(row["id"]))
            except Exception:
                continue
            stats["rows"] += 1
            key = (path, sample_id)
            label_text = clean_label(str(row.get(text_key, "")))
            label_id = -1
            if not label_text:
                stats["missing_text"] += 1
            else:
                label_id = direct_vocab_id(label_text, v)
                if label_id >= 0:
                    stats["primary_mapped"] += 1
                else:
                    stats["primary_unknown"] += 1

            if label_id >= 0:
                stats["mapped"] += 1
            else:
                stats["unknown_text"] += 1
            out[key] = int(label_id)
    return out, stats


def load_label_text_map(
    labels_csv: Path | None,
    text_key: str = "gaze_pseudo_label",
) -> tuple[dict[tuple[str, int], str], dict[str, int]]:
    stats = {
        "rows": 0,
        "with_text": 0,
        "missing_text": 0,
    }
    if labels_csv is None or (not labels_csv.exists()):
        return {}, stats

    out: dict[tuple[str, int], str] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                path = str(row["path"]).strip()
                sample_id = int(float(row["id"]))
            except Exception:
                continue
            stats["rows"] += 1
            txt = str(row.get(text_key, "")).strip()
            if txt:
                stats["with_text"] += 1
            else:
                stats["missing_text"] += 1
            out[(path, sample_id)] = txt
    return out, stats


def build_vocab_embedding_matrix(
    vocab2id: dict[str, int],
    label_embed_dir: Path | None,
    label_emb_dim: int = 512,
    normalize: bool = True,
) -> torch.Tensor | None:
    if not vocab2id:
        return None
    if label_embed_dir is None or (not label_embed_dir.exists()):
        return None
    vocab_size = max(int(x) for x in vocab2id.values()) + 1
    dim = int(label_emb_dim)
    out = torch.zeros((vocab_size, dim), dtype=torch.float32)
    found = 0
    for txt, idx in vocab2id.items():
        if int(idx) < 0 or int(idx) >= vocab_size:
            continue
        p = label_embed_dir / f"{txt}-emb.pt"
        if not p.exists():
            continue
        try:
            vec = torch.load(p, map_location="cpu")
            if not torch.is_tensor(vec):
                continue
            vec = vec.to(dtype=torch.float32).flatten()
            if vec.numel() != dim:
                continue
            if normalize:
                vec = F.normalize(vec.unsqueeze(0), p=2, dim=-1).squeeze(0)
            out[int(idx)] = vec
            found += 1
        except Exception:
            continue
    if found <= 0:
        return None
    return out


def build_split_bank(
    label_texts: list[str],
    canonical_ids: list[int],
    label_embed_dir: Path | None,
    embedding_dim: int = 512,
    normalize: bool = True,
) -> tuple[list[str], list[int], torch.Tensor]:
    """Build a split-specific retrieval bank from a flat list of label texts.

    Unlike :func:`build_vocab_embedding_matrix`, this function does **not** use
    the vocab2id ordering.  Bank row ``i`` corresponds to ``label_texts[i]`` and
    ``canonical_ids[i]`` — the bank index is **not** equal to the vocab label id.
    Callers must keep ``bank_canonical_ids`` alongside the bank tensor and use it
    to map a retrieved bank row back to the metric id space.

    Labels whose embedding file cannot be found are silently skipped; they do
    not appear in the returned lists or tensor.

    Parameters
    ----------
    label_texts:
        Ordered unique label strings for this split (e.g. val or test).
    canonical_ids:
        Vocab id for each entry in *label_texts* (parallel list).
        Pass ``-1`` for labels that have no valid vocab mapping; they will be
        included in the bank but excluded from metric evaluation by the caller.
    label_embed_dir:
        Directory containing ``{text}-emb.pt`` files.
    embedding_dim:
        Expected embedding dimensionality (default 512 for CLIP-B/32).
    normalize:
        L2-normalize embeddings before returning (default True).

    Returns
    -------
    bank_texts : list[str]
        Labels for which embeddings were found (subset of *label_texts*).
    bank_canonical_ids : list[int]
        Vocab ids parallel to *bank_texts*.
    bank_embs : torch.Tensor
        Float32 tensor of shape ``[len(bank_texts), embedding_dim]``.
        If no embeddings are found, shape is ``[0, embedding_dim]``.
    """
    dim = max(1, int(embedding_dim))
    empty = ([], [], torch.zeros((0, dim), dtype=torch.float32))

    if not label_texts:
        return empty
    if label_embed_dir is None or not Path(label_embed_dir).exists():
        return empty

    embed_dir = Path(label_embed_dir)
    bank_texts: list[str] = []
    bank_ids: list[int] = []
    vecs: list[torch.Tensor] = []

    for txt, cid in zip(label_texts, canonical_ids):
        txt = str(txt).strip()
        if not txt:
            continue

        # Try exact name first, then lowercased variant
        candidates_paths = [
            embed_dir / f"{txt}-emb.pt",
            embed_dir / f"{txt.lower()}-emb.pt",
        ]
        vec: torch.Tensor | None = None
        for p in candidates_paths:
            if not p.exists():
                continue
            try:
                loaded = torch.load(p, map_location="cpu")
                if not torch.is_tensor(loaded):
                    continue
                t = loaded.to(dtype=torch.float32).flatten()
                if int(t.numel()) != dim:
                    continue
                vec = t
                break
            except Exception:
                continue

        if vec is None:
            continue

        if normalize:
            vec = F.normalize(vec.unsqueeze(0), p=2, dim=-1).squeeze(0)

        bank_texts.append(txt)
        bank_ids.append(int(cid))
        vecs.append(vec)

    if not vecs:
        return empty

    bank_embs = torch.stack(vecs, dim=0)  # [N, dim]
    return bank_texts, bank_ids, bank_embs


def load_test_label_map(
    labels_csv: Path | None,
    vocab2id: dict[str, int] | None = None,
    vocab2id_lower: dict[str, int] | None = None,
) -> tuple[dict[str, int], dict[str, str], dict[str, list[int]], dict[str, int]]:
    stats = {
        "rows": 0,
        "mapped": 0,
        "missing_text": 0,
        "unknown_text": 0,
        "conflicts": 0,
    }
    if labels_csv is None or (not labels_csv.exists()):
        return {}, {}, {}, stats

    v = vocab2id or {}
    id_map: dict[str, int] = {}
    text_map: dict[str, str] = {}
    multi_id_map: dict[str, list[int]] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                path = str(row["path"]).strip()
            except Exception:
                continue
            if not path:
                continue
            stats["rows"] += 1

            t_primary = str(row.get("gaze_gt_label", "")).strip()
            t_multi = str(row.get("gaze_gt_labels", "")).strip()

            chosen_text = t_primary
            chosen_id = direct_vocab_id(t_primary, v)
            if not t_primary:
                stats["missing_text"] += 1
                chosen_id = -1
            elif chosen_id < 0:
                stats["unknown_text"] += 1
            else:
                stats["mapped"] += 1
                chosen_id = int(chosen_id)

            parsed_multi: list[int] = []
            if t_multi and chosen_id >= 0:
                parsed = [direct_vocab_id(x.strip(), v) for x in t_multi.split("-") if x.strip()]
                parsed_multi = sorted({int(x) for x in parsed if int(x) >= 0})
            if chosen_id >= 0:
                if not parsed_multi:
                    parsed_multi = [int(chosen_id)]

            prev = id_map.get(path, None)
            if prev is None:
                id_map[path] = int(chosen_id)
            else:
                if int(prev) >= 0 and int(chosen_id) >= 0 and int(prev) != int(chosen_id):
                    stats["conflicts"] += 1
                elif int(prev) < 0 and int(chosen_id) >= 0:
                    id_map[path] = int(chosen_id)

            if chosen_text and (path not in text_map):
                text_map[path] = str(chosen_text)
            elif path not in text_map:
                flags: list[str] = []
                for k in ("outside_frame", "uncertain", "eyes_closed"):
                    vv = str(row.get(k, "")).strip().lower()
                    if vv in {"true", "1", "yes"}:
                        flags.append(k)
                if flags:
                    text_map[path] = ",".join(flags)

            if path not in multi_id_map:
                multi_id_map[path] = list(parsed_multi)
            elif parsed_multi:
                merged = sorted(set(multi_id_map[path]).union(set(parsed_multi)))
                multi_id_map[path] = [int(x) for x in merged if int(x) >= 0]

    return id_map, text_map, multi_id_map, stats


@dataclass
class Record:
    sample_id: int
    image_rel: str
    image_path: Path
    gaze_x: float
    gaze_y: float
    bbox_px: tuple[float, float, float, float]
    label_id: int
    label_text: str = ""


@dataclass
class TestGroup:
    image_rel: str
    image_path: Path
    bbox_px: tuple[float, float, float, float]
    gt_points: list[tuple[float, float]]
    label_id: int = -1
    label_text: str = ""
    label_ids: list[int] = field(default_factory=list)


def load_records(
    annotation_file: Path,
    image_root: Path,
    label_map: dict[tuple[str, int], int] | None = None,
    label_text_map: dict[tuple[str, int], str] | None = None,
    split_prefix: str = "train/",
    strip_split_prefix: bool = True,
    skip_outside: bool = True,
    max_samples: int = 0,
) -> list[Record]:
    rows: list[list[str]] = []
    with annotation_file.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.reader(f) if row]

    records: list[Record] = []
    for row in rows:
        image_rel = str(row[0]).strip()
        if split_prefix and (not image_rel.startswith(split_prefix)):
            continue
        join_rel = image_rel[len(split_prefix) :] if (split_prefix and strip_split_prefix) else image_rel
        image_path = image_root / join_rel
        if not image_path.exists():
            continue

        try:
            sample_id = int(float(row[1]))
            gaze_x = float(row[8])
            gaze_y = float(row[9])
            bbox = (float(row[10]), float(row[11]), float(row[12]), float(row[13]))
        except Exception:
            continue

        if skip_outside and (gaze_x < 0.0 or gaze_y < 0.0):
            continue
        if not is_normalized_point(gaze_x, gaze_y):
            continue

        label_id = -1
        if label_map is not None:
            label_id = int(label_map.get((image_rel, sample_id), -1))
        label_text = ""
        if label_text_map is not None:
            label_text = str(label_text_map.get((image_rel, sample_id), "")).strip()

        records.append(
            Record(
                sample_id=sample_id,
                image_rel=image_rel,
                image_path=image_path,
                gaze_x=gaze_x,
                gaze_y=gaze_y,
                bbox_px=bbox,
                label_id=label_id,
                label_text=label_text,
            )
        )
        if max_samples > 0 and len(records) >= max_samples:
            break
    return records


def round_bbox(
    bbox: tuple[float, float, float, float],
    decimals: int,
) -> tuple[float, float, float, float]:
    return tuple(round(float(v), int(decimals)) for v in bbox)


def load_test_groups(
    annotation_file: Path,
    image_root: Path,
    test_label_map: dict[str, int] | None = None,
    test_label_text_map: dict[str, str] | None = None,
    test_label_ids_map: dict[str, list[int]] | None = None,
    split_prefix: str = "test2/",
    strip_split_prefix: bool = True,
    bbox_round_decimals: int = 3,
    max_groups: int = 0,
) -> list[TestGroup]:
    rows: list[list[str]] = []
    with annotation_file.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.reader(f) if row]

    grouped: dict[tuple[str, tuple[float, float, float, float]], TestGroup] = {}
    for row in rows:
        image_rel = str(row[0]).strip()
        if split_prefix and (not image_rel.startswith(split_prefix)):
            continue

        try:
            gaze_x = float(row[8])
            gaze_y = float(row[9])
            bbox = (float(row[10]), float(row[11]), float(row[12]), float(row[13]))
        except Exception:
            continue
        if not is_normalized_point(gaze_x, gaze_y):
            continue

        join_rel = image_rel[len(split_prefix) :] if (split_prefix and strip_split_prefix) else image_rel
        image_path = image_root / join_rel
        if not image_path.exists():
            continue

        key = (image_rel, round_bbox(bbox, decimals=bbox_round_decimals))
        if key not in grouped:
            grouped[key] = TestGroup(
                image_rel=image_rel,
                image_path=image_path,
                bbox_px=bbox,
                gt_points=[],
            )
        grouped[key].gt_points.append((gaze_x, gaze_y))

    # test label CSV is keyed only by image path. To avoid cross-head label
    # contamination, only attach those path-level labels when the annotation
    # file contains exactly one bbox-group for that image. If multiple bbox
    # groups exist under the same image path, leave labels unset (-1 / empty).
    group_counts_by_image: dict[str, int] = {}
    for image_rel, _bbox_key in grouped.keys():
        group_counts_by_image[image_rel] = int(group_counts_by_image.get(image_rel, 0)) + 1

    for (image_rel, _bbox_key), group in grouped.items():
        if int(group_counts_by_image.get(image_rel, 0)) != 1:
            continue
        group.label_id = int((test_label_map or {}).get(image_rel, -1))
        group.label_text = str((test_label_text_map or {}).get(image_rel, ""))
        group.label_ids = list((test_label_ids_map or {}).get(image_rel, []))

    groups = [g for g in grouped.values() if g.gt_points]
    for g in groups:
        if not g.label_ids and int(g.label_id) >= 0:
            g.label_ids = [int(g.label_id)]
    if max_groups > 0:
        groups = groups[: int(max_groups)]
    return groups
