from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]


def to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


def chat_text(
    processor: Any,
    user_text: str,
    assistant_text: str | None,
    *,
    with_image: bool,
    add_generation_prompt: bool,
) -> str:
    user_txt = str(user_text)
    if hasattr(processor, "apply_chat_template"):
        content: list[dict[str, str]] = []
        if with_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": user_txt})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": str(assistant_text)}]})
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=bool(add_generation_prompt),
            )
        except TypeError:
            return processor.apply_chat_template(messages, tokenize=False)

    prefix = f"<|vision_start|><|image_pad|><|vision_end|>\n{user_txt}" if with_image else user_txt
    if assistant_text is None:
        return prefix
    return f"{prefix}\n{str(assistant_text)}"


def resolve_path(path: str, root: Path = ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if os.environ.get("QWEN_DETERMINISTIC", "").lower() in {"1", "true"}:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True


def parse_dtype(dtype: str) -> torch.dtype | str:
    v = str(dtype).strip().lower()
    if v in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if v in {"fp16", "float16"}:
        return torch.float16
    if v in {"fp32", "float32"}:
        return torch.float32
    return "auto"


def to_autocast_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if dtype == torch.float16:
        return torch.float16
    if dtype == torch.float32:
        return torch.float32
    return torch.bfloat16


def env_flag(name: str) -> bool:
    v = str(os.environ.get(name, "")).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}
