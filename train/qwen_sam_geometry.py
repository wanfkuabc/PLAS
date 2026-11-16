"""SAM-coordinate image preparation without importing the legacy LLaVA package."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image


def sam_resize_size(image: Image.Image, image_size: int = 1024) -> Tuple[int, int]:
    width, height = image.size
    scale = float(image_size) / max(height, width)
    return max(1, int(height * scale + 0.5)), max(1, int(width * scale + 0.5))


def sam_image_input(image: Image.Image, image_size: int = 1024) -> torch.Tensor:
    image = image.convert("RGB")
    height, width = sam_resize_size(image, image_size)
    pixels = np.asarray(image.resize((width, height), Image.BILINEAR), dtype=np.float32)
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def sam_aligned_canvas(image: Image.Image, fill: Tuple[int, int, int], image_size: int = 1024) -> Image.Image:
    image = image.convert("RGB")
    height, width = sam_resize_size(image, image_size)
    canvas = Image.new("RGB", (image_size, image_size), fill)
    canvas.paste(image.resize((width, height), Image.BILINEAR), (0, 0))
    return canvas



