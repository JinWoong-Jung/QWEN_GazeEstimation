from __future__ import annotations

import re
from typing import Any

COORD_BINS: int = 1000
ANSWER_START: str = ""
ANSWER_END: str = "<|im_end|>"
POINT_PREFIX: str = "Point:"
OBJECT_PREFIX: str = "Object:"
GAZE_OBJ_UNKNOWN: str = "<obj_unknown>"
FORMAT_TOKENS: list[str] = []

REASONING_START: str = "<think>"
REASONING_END: str = "</think>"
REASONING_PREFIX: str = "Reasoning:"

_LOC_RE = re.compile(r"^<loc_(\d+)>$")
_OBJ_RE = re.compile(r"^<obj_(\d+)>$")

# --- parsing regexes (DOTALL so .* spans newlines) ---

# Legacy point-first: "Point: <loc_X><loc_Y>\nObject: <obj_K>  [<think>...</think>]"
# Groups: 1=loc_x, 2=loc_y, 3=obj
_STRICT_RE = re.compile(
    r"^\s*"
    r"Point:\s*(<loc_\d+>)(<loc_\d+>)"
    r"\s*Object:\s*(<obj_\d+>|<obj_unknown>)"
    r"\s*(?:<think>.*?</think>\s*)?"
    r"(?:<\|im_end\|>)?\s*$",
    re.DOTALL,
)

# Object-first direct: "Object: <obj_K>\nPoint: <loc_X><loc_Y>"
# Groups: 1=obj, 2=loc_x, 3=loc_y
_STRICT_RE_OBJ_FIRST = re.compile(
    r"^\s*"
    r"Object:\s*(<obj_\d+>|<obj_unknown>)"
    r"\s*Point:\s*(<loc_\d+>)(<loc_\d+>)"
    r"\s*(?:<\|im_end\|>)?\s*$",
    re.DOTALL,
)

# Flat reasoning-first: "Reasoning: ...\nObject: <obj_K>\nPoint: <loc_X><loc_Y>"
# Groups: 1=obj, 2=loc_x, 3=loc_y
_STRICT_RE_REASONING_FIRST = re.compile(
    r"^\s*Reasoning:.*?"
    r"\s*Object:\s*(<obj_\d+>|<obj_unknown>)"
    r"\s*Point:\s*(<loc_\d+>)(<loc_\d+>)"
    r"\s*(?:<\|im_end\|>)?\s*$",
    re.DOTALL,
)

# Legacy think-first: "<think>...</think>\nObject: <obj_K>\nPoint: <loc_X><loc_Y>"
# Groups: 1=obj, 2=loc_x, 3=loc_y
_STRICT_RE_THINK_FIRST = re.compile(
    r"^\s*<think>.*?</think>"
    r"\s*Object:\s*(<obj_\d+>|<obj_unknown>)"
    r"\s*Point:\s*(<loc_\d+>)(<loc_\d+>)"
    r"\s*(?:<\|im_end\|>)?\s*$",
    re.DOTALL,
)


def normalize_reasoning_text(text: str, max_words: int = 30, max_chars: int = 220) -> str:
    """Collapse whitespace, truncate, and ensure trailing period."""
    text = " ".join(str(text or "").split())
    if not text:
        return text
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    if len(text) > max_chars:
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        text = truncated[:last_space] if last_space > 0 else truncated
    if text and not text.endswith("."):
        text = text + "."
    return text


def _obj_token_width(num_classes: int) -> int:
    return max(3, len(str(max(0, int(num_classes) - 1))))


def _loc_token_width(coord_bins: int = COORD_BINS) -> int:
    return max(3, len(str(max(0, int(coord_bins) - 1))))


def quantize_coord(coord: float, bins: int = COORD_BINS) -> int:
    b = int(bins)
    if b <= 0:
        raise ValueError(f"bins must be positive, got: {bins!r}")
    return int(max(0, min(b - 1, round(float(coord) * (b - 1)))))


def dequantize_coord(bin_idx: int, bins: int = COORD_BINS) -> float:
    return float(int(bin_idx)) / float(max(1, int(bins) - 1))


def format_loc_token(bin_idx: int, width: int = 3) -> str:
    return f"<loc_{int(bin_idx):0{int(width)}d}>"


def format_obj_token(obj_id: int, width: int) -> str:
    return f"<obj_{int(obj_id):0{int(width)}d}>"


def build_gaze_special_tokens(num_classes: int, coord_bins: int = COORD_BINS) -> list[str]:
    tokens: list[str] = []
    coord_n = int(coord_bins)
    if coord_n <= 0:
        raise ValueError(f"coord_bins must be positive, got: {coord_bins!r}")
    loc_w = _loc_token_width(coord_n)
    for i in range(coord_n):
        tokens.append(format_loc_token(i, loc_w))
    w = _obj_token_width(num_classes)
    for i in range(int(num_classes)):
        tokens.append(format_obj_token(i, w))
    tokens.append(GAZE_OBJ_UNKNOWN)
    return tokens


def register_gaze_special_tokens(
    tokenizer: Any,
    num_classes: int,
    coord_bins: int = COORD_BINS,
) -> dict[str, int]:
    tokens = build_gaze_special_tokens(num_classes, coord_bins=coord_bins)
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in tokens if t not in existing]
    if new_tokens:
        try:
            # replace_additional_special_tokens=False preserves existing special tokens
            # (e.g. Qwen vision tokens) while appending our new ones.
            tokenizer.add_special_tokens(
                {"additional_special_tokens": new_tokens},
                replace_additional_special_tokens=False,
            )
        except TypeError:
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    return {tok: int(tokenizer.convert_tokens_to_ids(tok)) for tok in tokens}


def build_structured_target_text(
    point_x: float,
    point_y: float,
    obj_id: int | None,
    num_classes: int,
    *,
    obj_token: str | None = None,
    coord_bins: int = COORD_BINS,
    target_order: str = "object_point",
    reasoning_text: str | None = None,
    force_reasoning_format: bool = False,
) -> str:
    """Build structured target text.

    target_order values:
      "object_point"           — Object → Point  (direct baseline default)
      "point_object"           — Point → Object  (legacy)
      "point_object_reasoning" — Point → Object → Reasoning  (legacy post-hoc)
      "reasoning_object_point" — Reasoning → Object → Point  (causal reasoning)
    """
    coord_n = int(coord_bins)
    bx = quantize_coord(float(point_x), bins=coord_n)
    by = quantize_coord(float(point_y), bins=coord_n)
    loc_w = _loc_token_width(coord_n)
    if str(obj_token or "").strip():
        resolved_obj_tok = str(obj_token).strip()
    else:
        w = _obj_token_width(num_classes)
        resolved_obj_tok = format_obj_token(int(obj_id), w) if obj_id is not None else GAZE_OBJ_UNKNOWN

    point_str = f"{POINT_PREFIX} {format_loc_token(bx, loc_w)}{format_loc_token(by, loc_w)}"
    object_str = f"{OBJECT_PREFIX} {resolved_obj_tok}"

    order = str(target_order or "object_point").strip()

    if order == "reasoning_object_point":
        reasoning_body = normalize_reasoning_text(str(reasoning_text or "").strip())
        if reasoning_body or bool(force_reasoning_format):
            content_line = f"{REASONING_PREFIX} {reasoning_body}" if reasoning_body else REASONING_PREFIX
            return f"{content_line}\n{object_str}\n{point_str}"
        return f"{object_str}\n{point_str}"

    if order == "point_object_reasoning":
        base = f"{point_str}\n{object_str}"
        reasoning_body = str(reasoning_text or "").strip()
        if reasoning_body or bool(force_reasoning_format):
            content_line = f"Reasoning: {reasoning_body}" if reasoning_body else "Reasoning:"
            reasoning_block = f"{REASONING_START}\n{content_line}\n{REASONING_END}"
            return f"{base}\n{reasoning_block}"
        return base

    if order == "point_object":
        return f"{point_str}\n{object_str}"

    # default: "object_point"
    return f"{object_str}\n{point_str}"


def build_structured_target_text_with_reasoning(
    point_x: float,
    point_y: float,
    obj_id: int | None,
    num_classes: int,
    reasoning_text: str | None = None,
    *,
    obj_token: str | None = None,
    coord_bins: int = COORD_BINS,
    force_reasoning_format: bool = False,
    target_order: str = "reasoning_object_point",
) -> str:
    """Build target text, delegating to build_structured_target_text.

    When reasoning_text is present (or force_reasoning_format=True), uses
    target_order (default: reasoning_object_point).  When neither, falls back
    to object_point so the result matches build_structured_target_text().
    """
    effective_order = str(target_order or "reasoning_object_point").strip()
    has_reasoning = bool(reasoning_text) or bool(force_reasoning_format)
    if not has_reasoning:
        # No reasoning content → direct object_point baseline
        effective_order = "object_point"
    return build_structured_target_text(
        point_x=point_x,
        point_y=point_y,
        obj_id=obj_id,
        num_classes=num_classes,
        obj_token=obj_token,
        coord_bins=coord_bins,
        target_order=effective_order,
        reasoning_text=reasoning_text,
        force_reasoning_format=force_reasoning_format,
    )


def _extract_from_match(
    m: re.Match[str],
    *,
    obj_group: int,
    x_group: int,
    y_group: int,
    num_classes: int,
    coord_bins: int,
) -> dict:
    """Extract and validate point/object from a regex match."""
    obj_tok = m.group(obj_group)
    loc_x_tok = m.group(x_group)
    loc_y_tok = m.group(y_group)

    coord_n = int(coord_bins)
    nc = int(num_classes)

    try:
        bx = int(_LOC_RE.match(loc_x_tok).group(1))  # type: ignore[union-attr]
        by = int(_LOC_RE.match(loc_y_tok).group(1))  # type: ignore[union-attr]
    except Exception:
        return {
            "valid_format": False,
            "has_extra_text": False,
            "point_bins": None,
            "point_xy": None,
            "object_id": None,
            "object_unknown": False,
        }

    if bx >= coord_n or by >= coord_n:
        return {
            "valid_format": False,
            "has_extra_text": False,
            "point_bins": None,
            "point_xy": None,
            "object_id": None,
            "object_unknown": False,
        }

    if obj_tok == GAZE_OBJ_UNKNOWN:
        return {
            "valid_format": True,
            "has_extra_text": False,
            "point_bins": (bx, by),
            "point_xy": (dequantize_coord(bx, bins=coord_n), dequantize_coord(by, bins=coord_n)),
            "object_id": None,
            "object_unknown": True,
        }

    try:
        oid = int(_OBJ_RE.match(obj_tok).group(1))  # type: ignore[union-attr]
    except Exception:
        return {
            "valid_format": False,
            "has_extra_text": False,
            "point_bins": None,
            "point_xy": None,
            "object_id": None,
            "object_unknown": False,
        }

    out_of_range = nc > 0 and oid >= nc
    return {
        "valid_format": not out_of_range,
        "has_extra_text": False,
        "point_bins": (bx, by),
        "point_xy": (dequantize_coord(bx, bins=coord_n), dequantize_coord(by, bins=coord_n)),
        "object_id": oid,
        "object_unknown": False,
    }


def parse_structured_output_text(
    text: str,
    num_classes: int,
    coord_bins: int = COORD_BINS,
) -> dict:
    """Parse generated text, accepting all four target orders.

    Tries patterns in priority order:
      1. object_point          Object → Point
      2. reasoning_object_point  Reasoning → Object → Point
      3. point_object / point_object_reasoning  (legacy)
    """
    s = str(text or "").strip()
    coord_n = int(coord_bins)
    if coord_n <= 0:
        raise ValueError(f"coord_bins must be positive, got: {coord_bins!r}")

    _invalid = {
        "valid_format": False,
        "has_extra_text": bool(s),
        "point_bins": None,
        "point_xy": None,
        "object_id": None,
        "object_unknown": False,
    }

    # object_point: groups obj=1, x=2, y=3
    m = _STRICT_RE_OBJ_FIRST.match(s)
    if m is not None:
        return _extract_from_match(m, obj_group=1, x_group=2, y_group=3,
                                   num_classes=num_classes, coord_bins=coord_n)

    # flat reasoning_object_point: "Reasoning: ...\nObject: ...\nPoint: ..."
    m = _STRICT_RE_REASONING_FIRST.match(s)
    if m is not None:
        return _extract_from_match(m, obj_group=1, x_group=2, y_group=3,
                                   num_classes=num_classes, coord_bins=coord_n)

    # legacy think-first: "<think>...</think>\nObject: ...\nPoint: ..."
    m = _STRICT_RE_THINK_FIRST.match(s)
    if m is not None:
        return _extract_from_match(m, obj_group=1, x_group=2, y_group=3,
                                   num_classes=num_classes, coord_bins=coord_n)

    # legacy point_object / point_object_reasoning: groups x=1, y=2, obj=3
    m = _STRICT_RE.match(s)
    if m is not None:
        return _extract_from_match(m, obj_group=3, x_group=1, y_group=2,
                                   num_classes=num_classes, coord_bins=coord_n)

    return _invalid


def parse_structured_output_ids(
    token_ids: list[int],
    tokenizer: Any,
    num_classes: int,
    coord_bins: int = COORD_BINS,
) -> dict:
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return parse_structured_output_text(str(text).strip(), num_classes, coord_bins=coord_bins)


def is_valid_structured_output(parsed: dict) -> bool:
    return bool(parsed.get("valid_format", False))
