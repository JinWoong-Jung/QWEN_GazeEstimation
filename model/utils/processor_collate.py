from __future__ import annotations

from typing import Any

import torch

from ..modules.preprocess import resize_scene_and_head


def _build_chat_text(processor: Any, text: str, with_image: bool) -> str:
    txt = str(text)
    if hasattr(processor, "apply_chat_template"):
        content: list[dict[str, str]] = []
        if with_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": txt})
        messages = [{"role": "user", "content": content}]
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except TypeError:
            return processor.apply_chat_template(messages, tokenize=False)
    if with_image:
        return f"<|vision_start|><|image_pad|><|vision_end|>\n{txt}"
    return txt


def _build_joint_inputs(
    processor: Any,
    scene_images: list[Any],
    head_images: list[Any],
    text_inputs: list[str],
    head_text: str,
) -> dict[str, Any]:
    bsz = len(text_inputs)
    if not (len(scene_images) == len(head_images) == bsz):
        raise ValueError("scene/head/text batch sizes must match for joint processor inputs.")

    head_texts = [str(head_text) for _ in range(bsz)]
    mixed_texts = list(text_inputs) + head_texts
    mixed_images = list(scene_images) + list(head_images)
    proc_texts = [_build_chat_text(processor, t, with_image=True) for t in mixed_texts]
    return processor(
        text=proc_texts,
        images=mixed_images,
        return_tensors="pt",
        padding=True,
    )


class QwenTrainCollator:
    def __init__(
        self,
        processor: Any,
        head_text: str,
        scene_size: tuple[int, int] = (512, 512),
        head_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.processor = processor
        self.head_text = str(head_text)
        self.scene_size = (int(scene_size[0]), int(scene_size[1]))
        self.head_size = (int(head_size[0]), int(head_size[1]))

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        scene_images_raw = [x["scene_image"] for x in batch]
        head_images_raw = [x["head_image"] for x in batch]
        scene_images, head_images = resize_scene_and_head(
            scene_image=scene_images_raw,
            head_image=head_images_raw,
            scene_size=self.scene_size,
            head_size=self.head_size,
        )
        text_inputs = [x["text_input"] for x in batch]
        target_points = torch.stack([x["target_point"] for x in batch], dim=0)
        target_heatmaps = torch.stack([x["target_heatmap"] for x in batch], dim=0)
        target_labels = torch.tensor([x["target_label"] for x in batch], dtype=torch.long)
        target_label_embs = torch.stack([x["target_label_emb"] for x in batch], dim=0)
        target_label_valid = torch.stack([x["target_label_valid"] for x in batch], dim=0)

        joint_inputs = _build_joint_inputs(
            processor=self.processor,
            scene_images=scene_images,
            head_images=head_images,
            text_inputs=[str(t) for t in text_inputs],
            head_text=self.head_text,
        )
        return {
            "joint_inputs": dict(joint_inputs),
            "joint_bsz": int(len(text_inputs)),
            "target_point": target_points,
            "target_heatmap": target_heatmaps,
            "target_label": target_labels,
            "target_label_emb": target_label_embs,
            "target_label_valid": target_label_valid,
        }


class QwenTestCollator:
    def __init__(
        self,
        processor: Any,
        head_text: str,
        scene_size: tuple[int, int] = (512, 512),
        head_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.processor = processor
        self.head_text = str(head_text)
        self.scene_size = (int(scene_size[0]), int(scene_size[1]))
        self.head_size = (int(head_size[0]), int(head_size[1]))

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        target_label_ids_raw = [x["target_label_ids"] for x in batch]
        max_multi = max([int(t.numel()) for t in target_label_ids_raw] + [1])
        target_label_ids = torch.full((len(batch), max_multi), fill_value=-100, dtype=torch.long)
        for i, t in enumerate(target_label_ids_raw):
            n = int(t.numel())
            if n > 0:
                target_label_ids[i, :n] = t.to(dtype=torch.long)

        scene_images_raw = [x["scene_image"] for x in batch]
        head_images_raw = [x["head_image"] for x in batch]
        scene_images, head_images = resize_scene_and_head(
            scene_image=scene_images_raw,
            head_image=head_images_raw,
            scene_size=self.scene_size,
            head_size=self.head_size,
        )
        text_inputs = [x["text_input"] for x in batch]
        joint_inputs = _build_joint_inputs(
            processor=self.processor,
            scene_images=scene_images,
            head_images=head_images,
            text_inputs=[str(t) for t in text_inputs],
            head_text=self.head_text,
        )
        return {
            "joint_inputs": dict(joint_inputs),
            "joint_bsz": int(len(text_inputs)),
            "gt_points": [x["gt_points"] for x in batch],
            "target_label": torch.tensor([x["target_label"] for x in batch], dtype=torch.long),
            "target_label_ids": target_label_ids,
            "target_label_text": [x["target_label_text"] for x in batch],
        }
