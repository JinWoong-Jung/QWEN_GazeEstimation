from __future__ import annotations

import re
from typing import Any


OBJECT_TOKEN_PATTERN = r"<obj_(\d+)>"
OBJECT_LINE_PATTERN = rf"(?im)^\s*object\s*:\s*({OBJECT_TOKEN_PATTERN})\s*$"


def object_token_width(num_classes: int, min_width: int = 3) -> int:
    max_id = max(int(num_classes) - 1, 0)
    return max(int(min_width), len(str(int(max_id))))


def build_object_token(label_id: int, width: int = 3) -> str:
    w = max(1, int(width))
    return f"<obj_{int(label_id):0{w}d}>"


def parse_object_token(token: str) -> int | None:
    s = str(token or "").strip()
    m = re.fullmatch(OBJECT_TOKEN_PATTERN, s)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def is_object_token(text: str) -> bool:
    return parse_object_token(str(text or "")) is not None


def build_all_object_tokens(num_classes: int, width: int = 3) -> list[str]:
    n = max(0, int(num_classes))
    if n <= 0:
        return []
    w = max(1, int(width))
    return [build_object_token(i, width=w) for i in range(n)]


def format_answer(
    point_x: float,
    point_y: float,
    label_id: int,
    *,
    point_decimals: int = 4,
    width: int = 3,
) -> str:
    dec = max(0, int(point_decimals))
    px = max(0.0, min(1.0, float(point_x)))
    py = max(0.0, min(1.0, float(point_y)))
    obj_token = build_object_token(int(label_id), width=max(1, int(width)))
    return f"Point: {px:.{dec}f} {py:.{dec}f}\nObject: {obj_token}"


def register_object_special_tokens(
    tokenizer: Any,
    num_classes: int,
    width: int | None = None,
) -> tuple[int, int, list[str]]:
    n = max(0, int(num_classes))
    w = int(width) if width is not None else object_token_width(n)
    required_tokens = build_all_object_tokens(n, width=w)
    if n <= 0:
        return 0, w, required_tokens
    if tokenizer is None:
        raise RuntimeError("tokenizer is None; cannot register object special tokens.")

    existing_additional = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    merged: list[str] = []
    seen: set[str] = set()
    for t in existing_additional + required_tokens:
        s = str(t)
        if s in seen:
            continue
        seen.add(s)
        merged.append(s)

    if merged == existing_additional:
        return 0, w, required_tokens

    added = 0
    try:
        added = int(
            tokenizer.add_special_tokens(
                {"additional_special_tokens": merged},
                replace_additional_special_tokens=False,
            )
        )
    except TypeError:
        added = int(tokenizer.add_special_tokens({"additional_special_tokens": merged}))
    except Exception:
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        missing = [t for t in required_tokens if t not in vocab]
        added = int(tokenizer.add_tokens(missing, special_tokens=True)) if missing else 0
    return added, w, required_tokens


def parse_object_id_from_text(text: str) -> int | None:
    txt = str(text or "")
    m_line = re.search(OBJECT_LINE_PATTERN, txt)
    if m_line is not None:
        return parse_object_token(m_line.group(1))

    m_any = re.search(OBJECT_TOKEN_PATTERN, txt)
    if m_any is None:
        return None
    return parse_object_token(m_any.group(0))


def parse_object_token_span(text: str) -> tuple[int, int] | None:
    txt = str(text or "")
    m_line = re.search(OBJECT_LINE_PATTERN, txt)
    if m_line is not None:
        return int(m_line.start(1)), int(m_line.end(1))
    return None
