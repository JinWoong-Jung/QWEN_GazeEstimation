from __future__ import annotations

from typing import Any

import torch

from ..modules.preprocess import resize_scene
from .common import chat_text


def mask_padding_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pad_token_id: int | None,
) -> torch.Tensor:
    labels = input_ids.clone()
    if attention_mask is not None:
        labels[attention_mask == 0] = -100
    if pad_token_id is not None and int(pad_token_id) >= 0:
        labels[labels == int(pad_token_id)] = -100
    return labels


def find_subseq(seq: list[int], sub: list[int], *, from_right: bool = False) -> int:
    if len(sub) <= 0:
        return -1
    n, m = len(seq), len(sub)
    if m > n:
        return -1
    if from_right:
        for i in range(n - m, -1, -1):
            if seq[i : i + m] == sub:
                return i
        return -1
    for i in range(n - m + 1):
        if seq[i : i + m] == sub:
            return i
    return -1


def build_answer_mask(
    processor: Any,
    joint_inputs: dict[str, Any],
    target_texts: list[str],
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """Build a bool mask [B, L] that is True for answer-token positions."""
    input_ids = joint_inputs["input_ids"]
    attention_mask = joint_inputs.get("attention_mask", None)
    if not torch.is_tensor(input_ids):
        raise ValueError("joint_inputs['input_ids'] must be a tensor.")
    bsz, seqlen = int(input_ids.shape[0]), int(input_ids.shape[1])
    answer_mask = torch.zeros((bsz, seqlen), dtype=torch.bool)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return answer_mask

    for i in range(bsz):
        if i >= len(target_texts):
            continue
        if i < int(target_valid.numel()) and float(target_valid[i].item()) <= 0.0:
            continue

        valid_len = (
            int(attention_mask[i].sum().item())
            if torch.is_tensor(attention_mask)
            else seqlen
        )
        if valid_len <= 0:
            continue

        seq_ids = [int(x) for x in input_ids[i, :valid_len].tolist()]
        ans_txt = str(target_texts[i])
        out = tokenizer(ans_txt, add_special_tokens=False, return_attention_mask=False)
        ans_ids = out.get("input_ids", [])
        if isinstance(ans_ids, list) and ans_ids and isinstance(ans_ids[0], list):
            ans_ids = ans_ids[0]
        ans_ids = [int(x) for x in ans_ids]
        if not ans_ids:
            continue

        ans_start = find_subseq(seq_ids, ans_ids, from_right=True)
        if ans_start < 0:
            continue
        ans_end = min(valid_len, ans_start + len(ans_ids))
        answer_mask[i, ans_start:ans_end] = True

    return answer_mask


def build_train_inputs(
    processor: Any,
    scene_images: list[Any],
    text_inputs: list[str],
    target_texts: list[str],
    target_valid: torch.Tensor,
    max_text_length: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    if not (len(scene_images) == len(text_inputs) == len(target_texts)):
        raise ValueError("scene/text/target batch sizes must match.")

    chat_texts = [
        chat_text(
            processor=processor,
            user_text=text_inputs[i],
            assistant_text=target_texts[i],
            with_image=True,
            add_generation_prompt=False,
        )
        for i in range(len(text_inputs))
    ]
    # truncation=False: prevents VLM image-token alignment from breaking.
    joint_inputs = processor(
        text=chat_texts,
        images=scene_images,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    attention_mask = joint_inputs.get("attention_mask", None)
    labels = mask_padding_labels(
        input_ids=joint_inputs["input_ids"],
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
    )

    answer_mask = build_answer_mask(
        processor=processor,
        joint_inputs=dict(joint_inputs),
        target_texts=target_texts,
        target_valid=target_valid,
    )

    bsz = int(labels.shape[0])
    # Disable answer mask for invalid samples.
    tv = target_valid.to(dtype=torch.float32).flatten()
    if int(tv.numel()) == bsz:
        bad = tv.le(0)
        if torch.any(bad):
            answer_mask[bad] = False

    # Fully mask labels for samples with no supervision.
    supervised = answer_mask.any(dim=1)
    if torch.any(~supervised):
        labels[~supervised] = -100

    return dict(joint_inputs), labels, answer_mask


def build_infer_inputs(
    processor: Any,
    scene_images: list[Any],
    text_inputs: list[str],
    max_text_length: int,
) -> dict[str, Any]:
    if len(scene_images) != len(text_inputs):
        raise ValueError("scene/text batch sizes must match for inference inputs.")
    chat_texts = [
        chat_text(
            processor=processor,
            user_text=t,
            assistant_text=None,
            with_image=True,
            add_generation_prompt=True,
        )
        for t in text_inputs
    ]
    joint_inputs = processor(
        text=chat_texts,
        images=scene_images,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    return dict(joint_inputs)


class QwenTrainCollator:
    def __init__(
        self,
        processor: Any,
        scene_size: tuple[int, int] = (512, 512),
        max_text_length: int = 256,
    ) -> None:
        self.processor = processor
        self.scene_size = (int(scene_size[0]), int(scene_size[1]))
        self.max_text_length = int(max_text_length)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        scene_images = resize_scene(
            scene_image=[x["scene_image"] for x in batch],
            scene_size=self.scene_size,
        )
        text_inputs = [str(x["text_input"]) for x in batch]
        target_texts = [str(x["target_text"]) for x in batch]
        target_valid = torch.stack(
            [x["target_text_valid"] for x in batch], dim=0
        ).to(dtype=torch.float32).flatten()

        joint_inputs, labels, loss_mask_answer = build_train_inputs(
            processor=self.processor,
            scene_images=scene_images,
            text_inputs=text_inputs,
            target_texts=target_texts,
            target_valid=target_valid,
            max_text_length=self.max_text_length,
        )
        return {
            "joint_inputs": joint_inputs,
            "labels": labels,
            "target_text": target_texts,
            "target_text_valid": target_valid,
            "loss_mask_answer": loss_mask_answer,
            "image_rel": [str(x.get("image_rel", "")) for x in batch],
        }


class QwenTestCollator:
    def __init__(
        self,
        processor: Any,
        scene_size: tuple[int, int] = (512, 512),
        max_text_length: int = 256,
    ) -> None:
        self.processor = processor
        self.scene_size = (int(scene_size[0]), int(scene_size[1]))
        self.max_text_length = int(max_text_length)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        scene_images = resize_scene(
            scene_image=[x["scene_image"] for x in batch],
            scene_size=self.scene_size,
        )
        text_inputs = [str(x["text_input"]) for x in batch]
        joint_inputs = build_infer_inputs(
            processor=self.processor,
            scene_images=scene_images,
            text_inputs=text_inputs,
            max_text_length=self.max_text_length,
        )
        return {
            "joint_inputs": joint_inputs,
            "target_text": [str(x["target_text"]) for x in batch],
            "target_text_valid": torch.stack(
                [x["target_text_valid"] for x in batch], dim=0
            ).to(dtype=torch.float32).flatten(),
            "target_label": torch.tensor(
                [int(x["target_label"]) for x in batch], dtype=torch.long
            ),
            "target_label_ids": [
                [int(v) for v in x.get("target_label_ids", []) if int(v) >= 0]
                for x in batch
            ],
            "target_point": torch.stack(
                [x["target_point"] for x in batch], dim=0
            ).to(dtype=torch.float32),
            "gt_points": [x["gt_points"] for x in batch],
            "target_label_text": [str(x.get("target_label_text", "")) for x in batch],
            "image_rel": [str(x.get("image_rel", "")) for x in batch],
        }
