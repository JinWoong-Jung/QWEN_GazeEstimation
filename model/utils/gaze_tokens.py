from __future__ import annotations

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
    REASONING_END_MARKER,
    REASONING_START_MARKER,
    _LOC_RE,
    _OBJ_RE,
    _loc_token_width,
    _obj_token_width,
    build_gaze_special_tokens,
    format_loc_token,
    format_obj_token,
    register_gaze_special_tokens,
)


def _clean_reasoning_text(text: str) -> str:
    text = str(text or "")
    for marker in GAZE_SCHEMA_MARKERS:
        text = text.replace(marker, "")
    return " ".join(text.split())


def _ensure_trailing_period(text: str) -> str:
    if text and not text.endswith("."):
        return text + "."
    return text


def normalize_reasoning_text(text: str, max_words: int = 60, max_chars: int = 500) -> str:
    """Collapse whitespace, truncate, and ensure trailing period."""
    text = _clean_reasoning_text(text)
    if not text:
        return text
    words = text.split()
    if int(max_words) > 0 and len(words) > int(max_words):
        text = " ".join(words[: int(max_words)])
    if int(max_chars) > 0 and len(text) > int(max_chars):
        truncated = text[: int(max_chars)]
        last_space = truncated.rfind(" ")
        text = truncated[:last_space] if last_space > 0 else truncated
    return _ensure_trailing_period(text)


def sanitize_reasoning_text(text: str) -> str:
    """Normalize reasoning formatting without applying length caps."""
    return _ensure_trailing_period(_clean_reasoning_text(text))


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


def _format_reasoning(reasoning_body: str) -> str:
    return f"{REASONING_START_MARKER}{reasoning_body}{REASONING_END_MARKER}"


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
    """Build structured target text for the active span schema."""
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
    order = str(target_order or "object_point").strip()

    if order == "point_object":
        return f"{point_span}{object_span}"
    if order == "object_point":
        return f"{object_span}{point_span}"
    if order == "reasoning_only":
        reasoning_body = sanitize_reasoning_text(str(reasoning_text or "").strip())
        return _format_reasoning(reasoning_body)
    if order == "reasoning_point_object":
        reasoning_body = sanitize_reasoning_text(str(reasoning_text or "").strip())
        if reasoning_body or bool(force_reasoning_format):
            return f"{_format_reasoning(reasoning_body)}{point_span}{object_span}"
        return f"{point_span}{object_span}"
    if order == "reasoning_object_point":
        reasoning_body = sanitize_reasoning_text(str(reasoning_text or "").strip())
        if reasoning_body or bool(force_reasoning_format):
            return f"{_format_reasoning(reasoning_body)}{object_span}{point_span}"
        return f"{object_span}{point_span}"

    raise ValueError(
        f"unsupported target_order={target_order!r}; expected one of "
        "'point_object', 'object_point', 'reasoning_only', "
        "'reasoning_point_object', 'reasoning_object_point'"
    )


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
    effective_order = str(target_order or "reasoning_object_point").strip()
    if not (bool(reasoning_text) or bool(force_reasoning_format)):
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


def _skip_reasoning_span(s: str, start: int) -> int | None:
    if not s.startswith(REASONING_START_MARKER, start):
        return start
    content_start = start + len(REASONING_START_MARKER)
    end = s.find(REASONING_END_MARKER, content_start)
    if end < 0:
        return None
    return end + len(REASONING_END_MARKER)


def parse_structured_output_text(
    text: str,
    num_classes: int,
    coord_bins: int = COORD_BINS,
) -> dict[str, Any]:
    """Parse active span-schema structured output."""
    s = _strip_optional_eos(text)
    if not s:
        return _invalid(False)
    coord_n = int(coord_bins)
    if coord_n <= 0:
        raise ValueError(f"coord_bins must be positive, got: {coord_bins!r}")

    pos = _skip_reasoning_span(s, 0)
    if pos is None:
        return _invalid(True)

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

    parsed_object = _parse_object_span(s, pos)
    if parsed_object is not None:
        obj_tok, pos_after_object = parsed_object
        parsed_point = _parse_point_span(s, pos_after_object)
        if parsed_point is None:
            return _invalid(True)
        x_tok, y_tok, end_pos = parsed_point
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
