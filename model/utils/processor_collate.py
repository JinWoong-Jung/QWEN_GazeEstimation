from __future__ import annotations

from typing import Any

import torch

from ..modules.preprocess import resize_scene
from .common import chat_text
from .gaze_tokens import (
    GAZE_OBJ_UNKNOWN,
)
from .gaze_tokens import _LOC_RE, _OBJ_RE


_LABEL_IDS_PAD_LEN: int = 5


def _pad_label_ids(ids_list: list[list[int]]) -> torch.Tensor:
    rows: list[list[int]] = []
    for ids in ids_list:
        truncated = ids[:_LABEL_IDS_PAD_LEN]
        padded = truncated + [-1] * (_LABEL_IDS_PAD_LEN - len(truncated))
        rows.append(padded)
    return torch.tensor(rows, dtype=torch.long)


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
    """Build a bool mask [B, L] that is True for all answer-token positions."""
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


def build_structured_masks(
    processor: Any,
    joint_inputs: dict[str, Any],
    target_texts: list[str],
    target_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-category bool masks [B, L] for structured answer targets.

    Returns (loss_mask_point, loss_mask_object, loss_mask_format).
    point  = tokens matching <loc_*>
    object = tokens matching <obj_*> or <obj_unknown>
    format = all other tokens inside the answer span
    """
    input_ids = joint_inputs["input_ids"]
    attention_mask = joint_inputs.get("attention_mask", None)
    if not torch.is_tensor(input_ids):
        bsz = len(target_texts)
        z = torch.zeros((bsz, 1), dtype=torch.bool)
        return z, z, z

    bsz, seqlen = int(input_ids.shape[0]), int(input_ids.shape[1])
    mask_pt = torch.zeros((bsz, seqlen), dtype=torch.bool)
    mask_obj = torch.zeros((bsz, seqlen), dtype=torch.bool)
    mask_fmt = torch.zeros((bsz, seqlen), dtype=torch.bool)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return mask_pt, mask_obj, mask_fmt

    # Batch tokenize all target texts once (1 call instead of bsz calls).
    raw_out = tokenizer(
        [str(target_texts[i]) if i < len(target_texts) else "" for i in range(bsz)],
        add_special_tokens=False,
        return_attention_mask=False,
    )
    batch_ans_ids: list[list[int]] = raw_out.get("input_ids", [])
    if batch_ans_ids and not isinstance(batch_ans_ids[0], list):
        batch_ans_ids = [batch_ans_ids]

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
        ans_ids = [int(x) for x in (batch_ans_ids[i] if i < len(batch_ans_ids) else [])]

        ans_start = find_subseq(seq_ids, ans_ids, from_right=True)
        if ans_start < 0:
            continue

        tok_strs: list[str] = []
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            raw_toks = tokenizer.convert_ids_to_tokens(ans_ids)
            tok_strs = [str(t) for t in raw_toks]
        else:
            tok_strs = [str(tid) for tid in ans_ids]

        for off, tok_str in enumerate(tok_strs):
            pos = ans_start + off
            if pos >= seqlen:
                continue
            if _LOC_RE.match(tok_str):
                mask_pt[i, pos] = True
            elif tok_str == GAZE_OBJ_UNKNOWN or _OBJ_RE.match(tok_str):
                mask_obj[i, pos] = True
            else:
                mask_fmt[i, pos] = True

    return mask_pt, mask_obj, mask_fmt


def build_train_inputs(
    processor: Any,
    scene_images: list[Any],
    text_inputs: list[str],
    target_texts: list[str],
    target_text_valid: torch.Tensor,
    target_point_valid: torch.Tensor,
    target_object_valid: torch.Tensor,
    target_format_valid: torch.Tensor,
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

    mask_pt, mask_obj, mask_fmt = build_structured_masks(
        processor=processor,
        joint_inputs=dict(joint_inputs),
        target_texts=target_texts,
        target_valid=target_text_valid,
    )

    bsz = int(labels.shape[0])
    tv_text = target_text_valid.to(dtype=torch.float32).flatten()
    tv_point = target_point_valid.to(dtype=torch.float32).flatten()
    tv_object = target_object_valid.to(dtype=torch.float32).flatten()
    tv_format = target_format_valid.to(dtype=torch.float32).flatten()
    if int(tv_text.numel()) == bsz:
        bad_text = tv_text.le(0)
        if torch.any(bad_text):
            mask_pt[bad_text] = False
            mask_obj[bad_text] = False
            mask_fmt[bad_text] = False
    if int(tv_point.numel()) == bsz:
        bad_point = tv_point.le(0)
        if torch.any(bad_point):
            mask_pt[bad_point] = False
    if int(tv_object.numel()) == bsz:
        bad_object = tv_object.le(0)
        if torch.any(bad_object):
            mask_obj[bad_object] = False
    if int(tv_format.numel()) == bsz:
        bad_format = tv_format.le(0)
        if torch.any(bad_format):
            mask_fmt[bad_format] = False

    supervised = mask_pt.any(dim=1) | mask_obj.any(dim=1) | mask_fmt.any(dim=1)
    if torch.any(~supervised):
        labels[~supervised] = -100

    return dict(joint_inputs), labels, mask_pt, mask_obj, mask_fmt


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
        max_text_length: int = 256,
        scene_size: tuple[int, int] | None = None,
    ) -> None:
        self.processor = processor
        self.max_text_length = int(max_text_length)
        self.scene_size = (int(scene_size[0]), int(scene_size[1])) if scene_size is not None else None

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        raw_images = [x["scene_image"] for x in batch]
        scene_images = (
            resize_scene(scene_image=raw_images, scene_size=self.scene_size)
            if self.scene_size is not None
            else raw_images
        )
        text_inputs = [str(x["text_input"]) for x in batch]
        target_texts = [str(x["target_text"]) for x in batch]
        target_text_valid = torch.stack(
            [x["target_text_valid"] for x in batch], dim=0
        ).to(dtype=torch.float32).flatten()
        target_point_valid = torch.stack(
            [x.get("target_point_valid", x["target_text_valid"]) for x in batch], dim=0
        ).to(dtype=torch.float32).flatten()
        target_object_valid = torch.stack(
            [x.get("target_object_valid", x["target_text_valid"]) for x in batch], dim=0
        ).to(dtype=torch.float32).flatten()
        target_format_valid = torch.stack(
            [x.get("target_format_valid", x["target_text_valid"]) for x in batch], dim=0
        ).to(dtype=torch.float32).flatten()

        (
            joint_inputs, labels,
            loss_mask_point, loss_mask_object, loss_mask_format,
        ) = build_train_inputs(
            processor=self.processor,
            scene_images=scene_images,
            text_inputs=text_inputs,
            target_texts=target_texts,
            target_text_valid=target_text_valid,
            target_point_valid=target_point_valid,
            target_object_valid=target_object_valid,
            target_format_valid=target_format_valid,
            max_text_length=self.max_text_length,
        )

        target_point_bin = torch.stack(
            [x.get("target_point_bin", torch.zeros(2, dtype=torch.long)) for x in batch], dim=0
        )
        target_object_id = torch.stack(
            [x.get("target_object_id", torch.zeros((), dtype=torch.long)) for x in batch], dim=0
        )

        return {
            "joint_inputs": joint_inputs,
            "labels": labels,
            "target_text": target_texts,
            "target_text_valid": target_text_valid,
            "target_point_valid": target_point_valid,
            "target_object_valid": target_object_valid,
            "target_format_valid": target_format_valid,
            "loss_mask_point": loss_mask_point,
            "loss_mask_object": loss_mask_object,
            "loss_mask_format": loss_mask_format,
            "target_point_bin": target_point_bin,
            "target_object_id": target_object_id,
            "image_rel": [str(x.get("image_rel", "")) for x in batch],
        }


class QwenRLCollator:
    """Collator for RL (GRPO) training.

    Produces inference-mode joint_inputs (no answer) plus all GT fields
    needed for reward computation, and returns the resized scene images
    so the caller can rebuild full (prompt + sampled answer) inputs for
    the log-prob forward pass without re-reading files from disk.
    """

    def __init__(
        self,
        processor: Any,
        max_text_length: int = 256,
        scene_size: tuple[int, int] | None = None,
    ) -> None:
        self.processor = processor
        self.max_text_length = int(max_text_length)
        self.scene_size = (int(scene_size[0]), int(scene_size[1])) if scene_size is not None else None

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        raw_images = [x["scene_image"] for x in batch]
        scene_images = (
            resize_scene(scene_image=raw_images, scene_size=self.scene_size)
            if self.scene_size is not None
            else raw_images
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
            "scene_images": scene_images,       # resized PIL images for logprob re-processing
            "text_input": text_inputs,
            "target_text": [str(x["target_text"]) for x in batch],
            "target_text_valid": torch.stack(
                [x["target_text_valid"] for x in batch], dim=0
            ).to(dtype=torch.float32).flatten(),
            "target_point_valid": torch.stack(
                [x.get("target_point_valid", x["target_text_valid"]) for x in batch], dim=0
            ).to(dtype=torch.float32).flatten(),
            "target_object_valid": torch.stack(
                [x.get("target_object_valid", x["target_text_valid"]) for x in batch], dim=0
            ).to(dtype=torch.float32).flatten(),
            "target_format_valid": torch.stack(
                [x.get("target_format_valid", x["target_text_valid"]) for x in batch], dim=0
            ).to(dtype=torch.float32).flatten(),
            "target_point_bin": torch.stack(
                [x.get("target_point_bin", torch.zeros(2, dtype=torch.long)) for x in batch], dim=0
            ),
            "target_object_id": torch.stack(
                [x.get("target_object_id", torch.zeros((), dtype=torch.long)) for x in batch], dim=0
            ),
            "target_label_ids": _pad_label_ids(
                [[int(v) for v in x.get("target_label_ids", []) if int(v) >= 0] for x in batch]
            ),
            "gt_points": [x["gt_points"] for x in batch],
            "target_label_text": [str(x.get("target_label_text", "")) for x in batch],
            "image_rel": [str(x.get("image_rel", "")) for x in batch],
        }


class QwenTestCollator:
    def __init__(
        self,
        processor: Any,
        max_text_length: int = 256,
        scene_size: tuple[int, int] | None = None,
    ) -> None:
        self.processor = processor
        self.max_text_length = int(max_text_length)
        self.scene_size = (int(scene_size[0]), int(scene_size[1])) if scene_size is not None else None

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        raw_images = [x["scene_image"] for x in batch]
        scene_images = (
            resize_scene(scene_image=raw_images, scene_size=self.scene_size)
            if self.scene_size is not None
            else raw_images
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
            "text_input": text_inputs,
            "target_text": [str(x["target_text"]) for x in batch],
            "target_text_valid": torch.stack(
                [x["target_text_valid"] for x in batch], dim=0
            ).to(dtype=torch.float32).flatten(),
            "target_point_valid": torch.stack(
                [x.get("target_point_valid", x["target_text_valid"]) for x in batch], dim=0
            ).to(dtype=torch.float32).flatten(),
            "target_label": torch.tensor(
                [int(x["target_label"]) for x in batch], dtype=torch.long
            ),
            "target_label_ids": _pad_label_ids(
                [[int(v) for v in x.get("target_label_ids", []) if int(v) >= 0] for x in batch]
            ),
            "target_point": torch.stack(
                [x["target_point"] for x in batch], dim=0
            ).to(dtype=torch.float32),
            "target_point_bin": torch.stack(
                [x.get("target_point_bin", torch.zeros(2, dtype=torch.long)) for x in batch], dim=0
            ),
            "target_object_id": torch.stack(
                [x.get("target_object_id", torch.zeros((), dtype=torch.long)) for x in batch], dim=0
            ),
            "gt_points": [x["gt_points"] for x in batch],
            "target_label_text": [str(x.get("target_label_text", "")) for x in batch],
            "target_object_valid": torch.stack(
                [x.get("target_object_valid", x["target_text_valid"]) for x in batch], dim=0
            ).to(dtype=torch.float32).flatten(),
            "target_format_valid": torch.stack(
                [x.get("target_format_valid", x["target_text_valid"]) for x in batch], dim=0
            ).to(dtype=torch.float32).flatten(),
            "image_rel": [str(x.get("image_rel", "")) for x in batch],
        }
