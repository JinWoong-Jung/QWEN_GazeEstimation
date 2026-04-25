from __future__ import annotations

import re
from typing import Any

COORD_BINS: int = 1000
ANSWER_START: str = "<|im_start|>"
ANSWER_END: str = "<|im_end|>"
POINT_PREFIX: str = "Point:"
OBJECT_PREFIX: str = "Object:"
GAZE_OBJ_UNKNOWN: str = "<obj_unknown>"
FORMAT_TOKENS: list[str] = []

_LOC_RE = re.compile(r"^<loc_(\d+)>$")
_OBJ_RE = re.compile(r"^<obj_(\d+)>$")
_STRICT_RE = re.compile(
    r"^"
    r"<\|im_start\|>"
    r"\s*Point:\s*"
    r"(<loc_\d+>)"
    r"(<loc_\d+>)"
    r"\s*"
    r"Object:\s*"
    r"(<obj_\d+>|<obj_unknown>)"
    r"\s*"
    r"<\|im_end\|>"
    r"$"
)


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
) -> str:
    coord_n = int(coord_bins)
    bx = quantize_coord(float(point_x), bins=coord_n)
    by = quantize_coord(float(point_y), bins=coord_n)
    loc_w = _loc_token_width(coord_n)
    if str(obj_token or "").strip():
        resolved_obj_tok = str(obj_token).strip()
    else:
        w = _obj_token_width(num_classes)
        resolved_obj_tok = format_obj_token(int(obj_id), w)
    return (
        f"{ANSWER_START}"
        f"{POINT_PREFIX} "
        f"{format_loc_token(bx, loc_w)}"
        f"{format_loc_token(by, loc_w)}"
        f"\n"
        f"{OBJECT_PREFIX} "
        f"{resolved_obj_tok}"
        f"{ANSWER_END}"
    )


def parse_structured_output_text(
    text: str,
    num_classes: int,
    coord_bins: int = COORD_BINS,
) -> dict:
    s = str(text or "").strip()
    m = _STRICT_RE.match(s)
    if m is None:
        return {
            "valid_format": False,
            "has_extra_text": bool(s),
            "point_bins": None,
            "point_xy": None,
            "object_id": None,
            "object_unknown": False,
        }

    loc_x_tok, loc_y_tok, obj_tok = m.group(1), m.group(2), m.group(3)
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

    nc = int(num_classes)
    coord_n = int(coord_bins)
    if coord_n <= 0:
        raise ValueError(f"coord_bins must be positive, got: {coord_bins!r}")

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
        oid = int(_OBJ_RE.match(obj_tok).group(1))    # type: ignore[union-attr]
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


def is_valid_structured_output(parsed: dict) -> bool:
    return bool(parsed.get("valid_format", False))
