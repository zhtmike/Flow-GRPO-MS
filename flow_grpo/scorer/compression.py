import io
from typing import List, Optional, Union

import mindspore as ms
import numpy as np
from PIL import Image

from .scorer import Scorer


class JpegCompressibilityScorer(Scorer):

    def __init__(self, max_size: int = 256) -> None:
        self.max_size = max_size

    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, ms.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        if isinstance(images, (np.ndarray, ms.Tensor)):
            images = self.array_to_images(images)

        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.convert("RGB").save(buffer, format="JPEG", quality=95)
        # each compressed image file size (kb)
        sizes = [buffer.tell() / 1024 for buffer in buffers]
        # 256 kb: score 0; 0 kb: score 1
        rewards = [max(0, 1 - x / self.max_size) for x in sizes]
        return rewards


class JpegImcompressibilityScorer(JpegCompressibilityScorer):

    def __init__(self, max_size: int = 256) -> None:
        super().__init__(max_size=max_size)

    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, ms.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        rewards = super().__call__(images, prompts)
        rewards = [1 - r for r in rewards]
        return rewards


def test_jpeg_compressibility_scorer():
    scorer = JpegCompressibilityScorer()
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images))


if __name__ == "__main__":
    test_jpeg_compressibility_scorer()
