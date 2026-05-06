from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Gaze special-token inventory
# ---------------------------------------------------------------------------

COORD_BINS: int = 1000
ANSWER_END: str = "<|im_end|>"
GAZE_OBJ_UNKNOWN: str = "<obj_unknown>"

REASONING_START_MARKER: str = "<|reasoning_start|>"
REASONING_END_MARKER: str = "<|reasoning_end|>"
POINT_START_MARKER: str = "<|point_start|>"
POINT_END_MARKER: str = "<|point_end|>"
OBJECT_START_MARKER: str = "<|object_start|>"
OBJECT_END_MARKER: str = "<|object_end|>"

GAZE_SCHEMA_MARKERS: list[str] = [
    REASONING_START_MARKER,
    REASONING_END_MARKER,
    POINT_START_MARKER,
    POINT_END_MARKER,
    OBJECT_START_MARKER,
    OBJECT_END_MARKER,
]
FORMAT_TOKENS: list[str] = list(GAZE_SCHEMA_MARKERS)

_LOC_RE = re.compile(r"^<loc_(\d+)>$")
_OBJ_RE = re.compile(r"^<obj_(\d+)>$")


def _obj_token_width(num_classes: int) -> int:
    return max(3, len(str(max(0, int(num_classes) - 1))))


def _loc_token_width(coord_bins: int = COORD_BINS) -> int:
    return max(3, len(str(max(0, int(coord_bins) - 1))))


def format_loc_token(bin_idx: int, width: int = 3) -> str:
    return f"<loc_{int(bin_idx):0{int(width)}d}>"


def format_obj_token(obj_id: int, width: int) -> str:
    return f"<obj_{int(obj_id):0{int(width)}d}>"


def build_gaze_special_tokens(num_classes: int, coord_bins: int = COORD_BINS) -> list[str]:
    tokens: list[str] = list(GAZE_SCHEMA_MARKERS)
    coord_n = int(coord_bins)
    if coord_n <= 0:
        raise ValueError(f"coord_bins must be positive, got: {coord_bins!r}")

    loc_w = _loc_token_width(coord_n)
    for i in range(coord_n):
        tokens.append(format_loc_token(i, loc_w))

    obj_w = _obj_token_width(num_classes)
    for i in range(int(num_classes)):
        tokens.append(format_obj_token(i, obj_w))

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
            # Preserve Qwen/chat/vision tokens while appending gaze tokens.
            tokenizer.add_special_tokens(
                {"additional_special_tokens": new_tokens},
                replace_additional_special_tokens=False,
            )
        except TypeError:
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    return {tok: int(tokenizer.convert_tokens_to_ids(tok)) for tok in tokens}


__all__ = [
    "ANSWER_END",
    "COORD_BINS",
    "FORMAT_TOKENS",
    "GAZE_OBJ_UNKNOWN",
    "GAZE_SCHEMA_MARKERS",
    "OBJECT_END_MARKER",
    "OBJECT_START_MARKER",
    "POINT_END_MARKER",
    "POINT_START_MARKER",
    "REASONING_END_MARKER",
    "REASONING_START_MARKER",
    "_LOC_RE",
    "_OBJ_RE",
    "_loc_token_width",
    "_obj_token_width",
    "build_gaze_special_tokens",
    "format_loc_token",
    "format_obj_token",
    "register_gaze_special_tokens",
]
