from __future__ import annotations

from typing import Any


def to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


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
