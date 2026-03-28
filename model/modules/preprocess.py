from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def _to_size_tuple(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"size must be int or (h, w), got: {size}")
    return int(size[0]), int(size[1])


def _resize_tensor_image(
    image: torch.Tensor,
    size: tuple[int, int],
    mode: str = "bilinear",
    antialias: bool = True,
) -> torch.Tensor:
    if image.dim() not in {3, 4}:
        raise ValueError(f"tensor image must be CHW or BCHW, got shape={tuple(image.shape)}")

    is_batched = image.dim() == 4
    x = image if is_batched else image.unsqueeze(0)
    resized = F.interpolate(
        x,
        size=size,
        mode=mode,
        align_corners=False if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None,
        antialias=antialias if mode in {"bilinear", "bicubic"} else False,
    )
    return resized if is_batched else resized.squeeze(0)


def _resize_pil_image(
    image: Image.Image,
    size: tuple[int, int],
    resample: int = Image.Resampling.BILINEAR,
) -> Image.Image:
    h, w = size
    return image.resize((w, h), resample=resample)


def _resize_any(
    image: Any,
    size: tuple[int, int],
) -> Any:
    if image is None:
        return None
    if isinstance(image, (list, tuple)):
        resized = [_resize_any(x, size) for x in image]
        return type(image)(resized) if isinstance(image, tuple) else resized
    if isinstance(image, torch.Tensor):
        return _resize_tensor_image(image, size=size)
    if isinstance(image, Image.Image):
        return _resize_pil_image(image, size=size)
    raise TypeError(f"unsupported image type: {type(image)}")


def resize_scene_and_head(
    scene_image: Any,
    head_image: Any,
    scene_size: int | tuple[int, int] = (512, 512),
    head_size: int | tuple[int, int] = (224, 224),
) -> tuple[Any, Any]:
    scene_hw = _to_size_tuple(scene_size)
    head_hw = _to_size_tuple(head_size)
    return _resize_any(scene_image, scene_hw), _resize_any(head_image, head_hw)


class GazeInputResizer(nn.Module):
    def __init__(
        self,
        scene_size: int | tuple[int, int] = (512, 512),
        head_size: int | tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.scene_size = _to_size_tuple(scene_size)
        self.head_size = _to_size_tuple(head_size)

    def forward(self, scene_image: Any, head_image: Any) -> tuple[Any, Any]:
        return resize_scene_and_head(
            scene_image=scene_image,
            head_image=head_image,
            scene_size=self.scene_size,
            head_size=self.head_size,
        )
