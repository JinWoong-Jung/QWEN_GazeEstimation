from __future__ import annotations

import re
from typing import Any


POINT_MODE_CONTINUOUS = "continuous"
POINT_MODE_BIN = "bin"
POINT_TOKEN_RE = re.compile(r"^<pt(?P<bins>\d+)_(?P<idx>\d+)>$")


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_point_mode(raw: Any) -> str:
    mode = str(raw or POINT_MODE_CONTINUOUS).strip().lower()
    if mode in {POINT_MODE_CONTINUOUS, POINT_MODE_BIN}:
        return mode
    raise ValueError(f"Unsupported point_mode: {raw!r}")


def point_bin_index(x: float, num_bins: int) -> int:
    bins = max(2, int(num_bins))
    return int(round(clamp01(x) * float(bins - 1)))


def point_bin_value(index: int, num_bins: int) -> float:
    bins = max(2, int(num_bins))
    idx = max(0, min(int(index), bins - 1))
    return float(idx) / float(bins - 1)


def point_bin_token(index: int, num_bins: int) -> str:
    bins = max(2, int(num_bins))
    width = len(str(bins - 1))
    idx = max(0, min(int(index), bins - 1))
    return f"<pt{bins}_{idx:0{width}d}>"


def format_point_value(
    x: float,
    *,
    point_mode: str,
    point_decimals: int,
    point_bin_count: int,
) -> str:
    mode = normalize_point_mode(point_mode)
    if mode == POINT_MODE_BIN:
        return point_bin_token(point_bin_index(x, point_bin_count), point_bin_count)
    dec = max(0, int(point_decimals))
    return f"{clamp01(x):.{dec}f}"


def parse_point_token(token: str) -> tuple[int, int] | None:
    m = POINT_TOKEN_RE.match(str(token or "").strip())
    if m is None:
        return None
    try:
        return int(m.group("idx")), int(m.group("bins"))
    except Exception:
        return None


def parse_point_token_pair(x_tok: str, y_tok: str) -> tuple[float, float] | None:
    px = parse_point_token(x_tok)
    py = parse_point_token(y_tok)
    if px is None or py is None:
        return None
    ix, bx = px
    iy, by = py
    if bx != by:
        return None
    return point_bin_value(ix, bx), point_bin_value(iy, by)


def render_point_text_human(text: str, point_decimals: int = 4) -> str:
    txt = str(text or "")
    m = re.search(
        r"(?im)^(\s*point\s*:\s*)(<pt\d+_\d+>)(\s*[,\s]+\s*)(<pt\d+_\d+>)(\s*)$",
        txt,
    )
    if m is None:
        return txt
    parsed = parse_point_token_pair(m.group(2), m.group(4))
    if parsed is None:
        return txt
    dec = max(0, int(point_decimals))
    x, y = parsed
    repl = f"{m.group(1)}{x:.{dec}f}{m.group(3)}{y:.{dec}f}{m.group(5)}"
    return txt[: m.start(0)] + repl + txt[m.end(0) :]


def add_point_bin_tokens(tokenizer: Any, num_bins: int) -> int:
    if tokenizer is None:
        raise RuntimeError("tokenizer is None; cannot register point bin tokens.")

    bins = max(2, int(num_bins))
    current = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    merged: list[str] = []
    seen: set[str] = set()
    width = len(str(bins - 1))
    required = [f"<pt{bins}_{i:0{width}d}>" for i in range(bins)]

    for tok in current + required:
        s = str(tok)
        if s in seen:
            continue
        seen.add(s)
        merged.append(s)
    if merged == current:
        return 0

    try:
        return int(
            tokenizer.add_special_tokens(
                {"additional_special_tokens": merged},
                replace_additional_special_tokens=False,
            )
        )
    except TypeError:
        return int(tokenizer.add_special_tokens({"additional_special_tokens": merged}))
    except Exception:
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        missing = [tok for tok in required if tok not in vocab]
        if not missing:
            return 0
        return int(tokenizer.add_tokens(missing, special_tokens=True))
