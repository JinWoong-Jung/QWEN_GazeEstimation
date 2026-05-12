from __future__ import annotations

import re
from typing import Any

from .special_tokens import (
    ANSWER_END,
    COORD_BINS,
    FORMAT_TOKENS,
    GAZE_OBJ_UNKNOWN,
    GAZE_SCHEMA_MARKERS,
    OBJECT_END_MARKER,
    OBJECT_START_MARKER,
    POINT_END_MARKER,
    POINT_START_MARKER,
    _LOC_RE,
    _OBJ_RE,
    _loc_token_width,
    _obj_token_width,
    build_gaze_special_tokens,
    format_loc_token,
    format_obj_token,
    register_gaze_special_tokens,
)


POINT_PREFIX: str = "Point:"
OBJECT_PREFIX: str = "Object:"
SUPPORTED_TARGET_ORDERS: set[str] = {"point_object", "text_point_object"}
_LEGACY_POINT_OBJECT_RE = re.compile(
    r"^\s*Point:\s*(<loc_\d+>)(<loc_\d+>)\s*"
    r"Object:\s*(<obj_\d+>|<obj_unknown>)\s*$",
    re.DOTALL,
)

def quantize_coord(coord: float, bins: int = COORD_BINS) -> int:
    b = int(bins)
    if b <= 0:
        raise ValueError(f"bins must be positive, got: {bins!r}")
    return int(max(0, min(b - 1, round(float(coord) * (b - 1)))))


def dequantize_coord(bin_idx: int, bins: int = COORD_BINS) -> float:
    return float(int(bin_idx)) / float(max(1, int(bins) - 1))


def _format_point(loc_x: str, loc_y: str) -> str:
    return f"{POINT_START_MARKER}{loc_x}{loc_y}{POINT_END_MARKER}"


def _format_object(obj_tok: str) -> str:
    return f"{OBJECT_START_MARKER}{obj_tok}{OBJECT_END_MARKER}"


def _format_text_point_object(loc_x: str, loc_y: str, obj_tok: str) -> str:
    return f"{POINT_PREFIX}{loc_x}{loc_y}\n{OBJECT_PREFIX}{obj_tok}"


def build_structured_target_text(
    point_x: float,
    point_y: float,
    obj_id: int | None,
    num_classes: int,
    *,
    obj_token: str | None = None,
    coord_bins: int = COORD_BINS,
    target_order: str = "point_object",
) -> str:
    """Build structured target text using the configured point/object schema."""
    coord_n = int(coord_bins)
    bx = quantize_coord(float(point_x), bins=coord_n)
    by = quantize_coord(float(point_y), bins=coord_n)
    loc_w = _loc_token_width(coord_n)
    if str(obj_token or "").strip():
        resolved_obj_tok = str(obj_token).strip()
    else:
        obj_w = _obj_token_width(num_classes)
        resolved_obj_tok = (
            format_obj_token(int(obj_id), obj_w)
            if obj_id is not None
            else GAZE_OBJ_UNKNOWN
        )

    point_span = _format_point(format_loc_token(bx, loc_w), format_loc_token(by, loc_w))
    object_span = _format_object(resolved_obj_tok)
    order = str(target_order or "point_object").strip()

    if order == "point_object":
        return f"{point_span}{object_span}"
    if order == "text_point_object":
        return _format_text_point_object(
            format_loc_token(bx, loc_w),
            format_loc_token(by, loc_w),
            resolved_obj_tok,
        )

    raise ValueError(
        f"unsupported target_order={target_order!r}; expected one of "
        f"{sorted(SUPPORTED_TARGET_ORDERS)}"
    )


def _invalid(has_extra_text: bool) -> dict[str, Any]:
    return {
        "valid_format": False,
        "has_extra_text": bool(has_extra_text),
        "point_bins": None,
        "point_xy": None,
        "object_id": None,
        "object_unknown": False,
    }


def _parse_loc_token(token: str) -> int | None:
    m = _LOC_RE.match(str(token))
    return int(m.group(1)) if m is not None else None


def _parse_obj_token(token: str, num_classes: int) -> tuple[bool, int | None, bool]:
    if str(token) == GAZE_OBJ_UNKNOWN:
        return True, None, True
    m = _OBJ_RE.match(str(token))
    if m is None:
        return False, None, False
    oid = int(m.group(1))
    if int(num_classes) > 0 and oid >= int(num_classes):
        return False, None, False
    return True, oid, False


def _make_parsed(
    x_tok: str,
    y_tok: str,
    obj_tok: str,
    *,
    num_classes: int,
    coord_bins: int,
) -> dict[str, Any]:
    bx = _parse_loc_token(x_tok)
    by = _parse_loc_token(y_tok)
    if bx is None or by is None or bx >= int(coord_bins) or by >= int(coord_bins):
        return _invalid(False)
    obj_ok, object_id, object_unknown = _parse_obj_token(obj_tok, int(num_classes))
    if not obj_ok:
        return _invalid(False)
    return {
        "valid_format": True,
        "has_extra_text": False,
        "point_bins": (bx, by),
        "point_xy": (
            dequantize_coord(bx, bins=int(coord_bins)),
            dequantize_coord(by, bins=int(coord_bins)),
        ),
        "object_id": object_id,
        "object_unknown": object_unknown,
    }


def _strip_optional_eos(text: str) -> str:
    s = str(text or "").strip()
    if s.endswith(ANSWER_END):
        s = s[: -len(ANSWER_END)].strip()
    return s


def _parse_point_span(s: str, start: int) -> tuple[str, str, int] | None:
    if not s.startswith(POINT_START_MARKER, start):
        return None
    pos = start + len(POINT_START_MARKER)

    def _read_loc(pos_: int) -> tuple[str, int] | None:
        end = s.find(">", pos_)
        if end < 0:
            return None
        tok = s[pos_ : end + 1]
        if _LOC_RE.match(tok) is None:
            return None
        return tok, end + 1

    x = _read_loc(pos)
    if x is None:
        return None
    x_tok, pos = x
    y = _read_loc(pos)
    if y is None:
        return None
    y_tok, pos = y
    if not s.startswith(POINT_END_MARKER, pos):
        return None
    return x_tok, y_tok, pos + len(POINT_END_MARKER)


def _parse_object_span(s: str, start: int) -> tuple[str, int] | None:
    if not s.startswith(OBJECT_START_MARKER, start):
        return None
    pos = start + len(OBJECT_START_MARKER)
    if s.startswith(GAZE_OBJ_UNKNOWN, pos):
        obj_tok = GAZE_OBJ_UNKNOWN
    else:
        end = s.find(">", pos)
        if end < 0:
            return None
        obj_tok = s[pos : end + 1]
        if _OBJ_RE.match(obj_tok) is None:
            return None
    pos += len(obj_tok)
    if not s.startswith(OBJECT_END_MARKER, pos):
        return None
    return obj_tok, pos + len(OBJECT_END_MARKER)


def _parse_legacy_text(s: str, start: int, *, num_classes: int, coord_bins: int) -> dict[str, Any] | None:
    body = s[start:].strip()
    m = _LEGACY_POINT_OBJECT_RE.match(body)
    if m is not None:
        return _make_parsed(
            m.group(1),
            m.group(2),
            m.group(3),
            num_classes=num_classes,
            coord_bins=coord_bins,
        )
    return None


def parse_structured_output_text(
    text: str,
    num_classes: int,
    coord_bins: int = COORD_BINS,
) -> dict[str, Any]:
    """Parse legacy Point/Object output, with span-schema backward compatibility."""
    s = _strip_optional_eos(text)
    if not s:
        return _invalid(False)
    coord_n = int(coord_bins)
    if coord_n <= 0:
        raise ValueError(f"coord_bins must be positive, got: {coord_bins!r}")

    pos = 0

    legacy = _parse_legacy_text(s, pos, num_classes=num_classes, coord_bins=coord_n)
    if legacy is not None:
        return legacy

    parsed_point = _parse_point_span(s, pos)
    if parsed_point is not None:
        x_tok, y_tok, pos_after_point = parsed_point
        parsed_object = _parse_object_span(s, pos_after_point)
        if parsed_object is None:
            return _invalid(True)
        obj_tok, end_pos = parsed_object
        if end_pos != len(s):
            return _invalid(True)
        return _make_parsed(
            x_tok,
            y_tok,
            obj_tok,
            num_classes=num_classes,
            coord_bins=coord_n,
        )

    return _invalid(True)


def parse_structured_output_ids(
    token_ids: list[int],
    tokenizer: Any,
    num_classes: int,
    coord_bins: int = COORD_BINS,
) -> dict[str, Any]:
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return parse_structured_output_text(str(text).strip(), num_classes, coord_bins=coord_bins)


def is_valid_structured_output(parsed: dict[str, Any]) -> bool:
    return bool(parsed.get("valid_format", False))
