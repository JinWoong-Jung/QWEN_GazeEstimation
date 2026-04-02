from __future__ import annotations

import re
from typing import Any

import torch

from ..modules.preprocess import resize_scene
from .common import chat_text
from .object_tokens import slot_span


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


def find_subseq(seq: list[int], sub: list[int], *, start: int = 0, from_right: bool = False) -> int:
    if len(sub) <= 0:
        return -1
    n = len(seq)
    m = len(sub)
    if m > n:
        return -1
    if from_right:
        lo = max(0, int(start))
        for i in range(n - m, lo - 1, -1):
            if seq[i : i + m] == sub:
                return i
        return -1
    for i in range(max(0, int(start)), n - m + 1):
        if seq[i : i + m] == sub:
            return i
    return -1


def tokenize_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    txt = str(text)
    try:
        out = tokenizer(
            txt,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        ids = out.get("input_ids", [])
        offs = out.get("offset_mapping", None)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(x) for x in ids]
        if offs is None:
            return ids, None
        if isinstance(offs, list) and offs and isinstance(offs[0], list) and offs and isinstance(offs[0][0], tuple):
            offs = offs[0]
        offsets: list[tuple[int, int]] = []
        for ab in offs:
            if not isinstance(ab, (list, tuple)) or len(ab) != 2:
                offsets.append((0, 0))
            else:
                offsets.append((int(ab[0]), int(ab[1])))
        if len(offsets) != len(ids):
            return ids, None
        return ids, offsets
    except Exception:
        out = tokenizer(
            txt,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        ids = out.get("input_ids", [])
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids], None


def parse_target_spans(text: str) -> tuple[tuple[int, int] | None, tuple[int, int] | None, tuple[int, int] | None]:
    txt = str(text or "")
    num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    m_point = re.search(rf"(?im)^\s*point\s*:\s*({num})\s*[,\s]+\s*({num})\s*$", txt)
    x_span = (int(m_point.start(1)), int(m_point.end(1))) if m_point is not None else None
    y_span = (int(m_point.start(2)), int(m_point.end(2))) if m_point is not None else None
    o_span = slot_span(txt)
    return x_span, y_span, o_span


def component_masks(
    processor: Any,
    joint_inputs: dict[str, Any],
    target_texts: list[str],
    target_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = joint_inputs["input_ids"]
    attention_mask = joint_inputs.get("attention_mask", None)
    if not torch.is_tensor(input_ids):
        raise ValueError("joint_inputs['input_ids'] must be a tensor.")
    bsz, seqlen = int(input_ids.shape[0]), int(input_ids.shape[1])

    answer_mask = torch.zeros((bsz, seqlen), dtype=torch.bool)
    point_mask = torch.zeros((bsz, seqlen), dtype=torch.bool)
    object_mask = torch.zeros((bsz, seqlen), dtype=torch.bool)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return answer_mask, point_mask, object_mask

    for i in range(bsz):
        if i >= len(target_texts):
            continue
        if i < int(target_valid.numel()) and float(target_valid[i].item()) <= 0.0:
            continue

        valid_len = int(attention_mask[i].sum().item()) if torch.is_tensor(attention_mask) else seqlen
        if valid_len <= 0:
            continue

        seq_ids = [int(x) for x in input_ids[i, :valid_len].tolist()]
        ans_txt = str(target_texts[i])
        ans_ids, ans_offsets = tokenize_offsets(tokenizer, ans_txt)
        if len(ans_ids) <= 0:
            continue

        ans_start = find_subseq(seq_ids, ans_ids, from_right=True)
        if ans_start < 0:
            continue
        ans_end = min(valid_len, ans_start + len(ans_ids))
        answer_mask[i, ans_start:ans_end] = True

        if ans_offsets is None or len(ans_offsets) != len(ans_ids):
            continue

        x_span, y_span, o_span = parse_target_spans(ans_txt)
        for j, (a, b) in enumerate(ans_offsets):
            if b <= a:
                continue
            gj = ans_start + j
            if gj < 0 or gj >= seqlen:
                continue
            if x_span is not None and not (b <= x_span[0] or a >= x_span[1]):
                point_mask[i, gj] = True
            if y_span is not None and not (b <= y_span[0] or a >= y_span[1]):
                point_mask[i, gj] = True
            if o_span is not None and not (b <= o_span[0] or a >= o_span[1]):
                object_mask[i, gj] = True

    return answer_mask, point_mask, object_mask


def build_train_inputs(
    processor: Any,
    scene_images: list[Any],
    text_inputs: list[str],
    target_texts: list[str],
    target_valid: torch.Tensor,
    target_point_valid: torch.Tensor | None,
    target_object_valid: torch.Tensor | None,
    max_text_length: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    # For VLM processors (e.g., Qwen3-VL), truncation can break image-token alignment
    # between raw chat text and tokenized input_ids. Keep truncation disabled here.
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
    answer_mask, point_mask, object_mask = component_masks(
        processor=processor,
        joint_inputs=dict(joint_inputs),
        target_texts=target_texts,
        target_valid=target_valid,
    )

    bsz = int(labels.shape[0])
    point_valid = target_point_valid
    object_valid = target_object_valid
    if point_valid is None:
        point_valid = torch.ones((bsz,), dtype=torch.float32)
    if object_valid is None:
        object_valid = target_valid
    point_valid = point_valid.to(dtype=torch.float32).flatten()
    object_valid = object_valid.to(dtype=torch.float32).flatten()

    if point_valid.numel() == bsz:
        invalid_point = point_valid <= 0
        if torch.any(invalid_point):
            point_mask[invalid_point] = False
    if object_valid.numel() == bsz:
        invalid_object = object_valid <= 0
        if torch.any(invalid_object):
            object_mask[invalid_object] = False

    invalid_answer = torch.zeros((bsz,), dtype=torch.bool)
    if point_valid.numel() == bsz:
        invalid_answer = invalid_answer | point_valid.le(0)
    if object_valid.numel() == bsz:
        invalid_answer = invalid_answer | object_valid.le(0)
    if torch.any(invalid_answer):
        answer_mask[invalid_answer] = False

    supervised = answer_mask.any(dim=1) | point_mask.any(dim=1) | object_mask.any(dim=1)
    if torch.any(~supervised):
        labels[~supervised] = -100
    return dict(joint_inputs), labels, answer_mask, point_mask, object_mask


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
    # For VLM processors (e.g., Qwen3-VL), truncation can break image-token alignment
    # between raw chat text and tokenized input_ids. Keep truncation disabled here.
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
        include_raw_inputs: bool = False,
    ) -> None:
        self.processor = processor
        self.scene_size = (int(scene_size[0]), int(scene_size[1]))
        self.max_text_length = int(max_text_length)
        self.include_raw_inputs = bool(include_raw_inputs)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        scene_images_raw = [x["scene_image"] for x in batch]
        scene_images = resize_scene(
            scene_image=scene_images_raw,
            scene_size=self.scene_size,
        )
        text_inputs = [str(x["text_input"]) for x in batch]
        target_texts = [str(x["target_text"]) for x in batch]
        target_valid = torch.stack([x["target_text_valid"] for x in batch], dim=0).to(dtype=torch.float32).flatten()
        target_point_valid = torch.stack(
            [
                x.get("target_point_valid", torch.tensor(1.0, dtype=torch.float32))
                if torch.is_tensor(x.get("target_point_valid", None))
                else torch.tensor(float(x.get("target_point_valid", 1.0)), dtype=torch.float32)
                for x in batch
            ],
            dim=0,
        ).to(dtype=torch.float32).flatten()
        target_object_valid = torch.stack(
            [
                x.get("target_object_valid", x.get("target_text_valid", torch.tensor(1.0, dtype=torch.float32)))
                if torch.is_tensor(x.get("target_object_valid", x.get("target_text_valid", None)))
                else torch.tensor(
                    float(x.get("target_object_valid", x.get("target_text_valid", 1.0))),
                    dtype=torch.float32,
                )
                for x in batch
            ],
            dim=0,
        ).to(dtype=torch.float32).flatten()

        joint_inputs, labels, loss_mask_answer, loss_mask_point, loss_mask_object = build_train_inputs(
            processor=self.processor,
            scene_images=scene_images,
            text_inputs=text_inputs,
            target_texts=target_texts,
            target_valid=target_valid,
            target_point_valid=target_point_valid,
            target_object_valid=target_object_valid,
            max_text_length=self.max_text_length,
        )
        out = {
            "joint_inputs": joint_inputs,
            "labels": labels,
            "target_text": target_texts,
            "target_text_valid": target_valid,
            "target_point_valid": target_point_valid,
            "target_object_valid": target_object_valid,
            "target_label": torch.tensor([int(x["target_label"]) for x in batch], dtype=torch.long),
            "loss_mask_answer": loss_mask_answer,
            "loss_mask_point": loss_mask_point,
            "loss_mask_object": loss_mask_object,
        }
        if self.include_raw_inputs:
            out["scene_images"] = list(scene_images)
            out["text_inputs"] = text_inputs
        return out


class QwenTestCollator:
    def __init__(
        self,
        processor: Any,
        scene_size: tuple[int, int] = (512, 512),
        max_text_length: int = 256,
        include_raw_inputs: bool = False,
    ) -> None:
        self.processor = processor
        self.scene_size = (int(scene_size[0]), int(scene_size[1]))
        self.max_text_length = int(max_text_length)
        self.include_raw_inputs = bool(include_raw_inputs)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        scene_images_raw = [x["scene_image"] for x in batch]
        scene_images = resize_scene(
            scene_image=scene_images_raw,
            scene_size=self.scene_size,
        )
        text_inputs = [str(x["text_input"]) for x in batch]
        joint_inputs = build_infer_inputs(
            processor=self.processor,
            scene_images=scene_images,
            text_inputs=text_inputs,
            max_text_length=self.max_text_length,
        )

        out = {
            "joint_inputs": joint_inputs,
            "target_text": [str(x["target_text"]) for x in batch],
            "target_text_valid": torch.stack([x["target_text_valid"] for x in batch], dim=0).to(dtype=torch.float32).flatten(),
            "target_label": torch.tensor([int(x["target_label"]) for x in batch], dtype=torch.long),
            "target_label_ids": [
                [int(v) for v in x.get("target_label_ids", []) if int(v) >= 0]
                for x in batch
            ],
            "target_point": torch.stack([x["target_point"] for x in batch], dim=0).to(dtype=torch.float32),
            "gt_points": [x["gt_points"] for x in batch],
            "target_label_text": [str(x["target_label_text"]) for x in batch],
            "image_rel": [str(x["image_rel"]) for x in batch],
        }
        if self.include_raw_inputs:
            out["scene_images"] = list(scene_images)
            out["text_inputs"] = text_inputs
        return out
