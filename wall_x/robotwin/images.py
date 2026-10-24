"""RoboTwin-only square image preprocessing shared by training and serving."""

from PIL import Image
from qwen_vl_utils.vision_process import (
    IMAGE_FACTOR,
    MAX_PIXELS,
    MIN_PIXELS,
    smart_resize,
)


def resize_robotwin_square(
    image: Image.Image,
    size: int,
    *,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Image.Image:
    """Fit the complete view on black padding, then apply Qwen size alignment."""
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("RoboTwin square image size must be a positive integer")
    width, height = image.size
    scale = size / max(width, height)
    resized = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        resample=Image.Resampling.BICUBIC,
    )
    canvas = Image.new(image.mode, (size, size), color=0)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    output_height, output_width = smart_resize(
        size, size, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
    )
    if canvas.size != (output_width, output_height):
        canvas = canvas.resize((output_width, output_height))
    return canvas
