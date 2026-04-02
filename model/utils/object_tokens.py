from __future__ import annotations

import re
from typing import Any


OBJ_SLOT = "<obj_emb>"
SLOT_LINE_RE = re.compile(r"(?im)^\s*object\s*:\s*(<obj_emb>)\s*$")
SLOT_ANY_RE = re.compile(r"<obj_emb>")


def add_slot_token(tokenizer: Any) -> int:
    if tokenizer is None:
        raise RuntimeError("tokenizer is None; cannot register object slot token.")

    required = str(OBJ_SLOT)
    current = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    merged: list[str] = []
    seen: set[str] = set()
    for tok in current + [required]:
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
        if required in vocab:
            return 0
        return int(tokenizer.add_tokens([required], special_tokens=True))


def slot_span(text: str) -> tuple[int, int] | None:
    txt = str(text or "")
    m = SLOT_LINE_RE.search(txt)
    if m is not None:
        return int(m.start(1)), int(m.end(1))
    m2 = SLOT_ANY_RE.search(txt)
    if m2 is None:
        return None
    return int(m2.start(0)), int(m2.end(0))
