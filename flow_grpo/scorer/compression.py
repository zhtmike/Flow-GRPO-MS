import io
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from .scorer import Scorer


class JpegCompressibilityScorer(Scorer):

    def __call__(self,
                 images: Union[List[Image.Image], np.ndarray],
                 prompts: Optional[List[str]] = None) -> List[float]:
        if isinstance(images, np.ndarray):
            images = self.array_to_images(images)
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.convert("RGB").save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        rewards = [-x / 500 for x in sizes]
        return rewards


def test_jpeg_compressibility_scorer():
    scorer = JpegCompressibilityScorer()
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images))


if __name__ == "__main__":
    test_jpeg_compressibility_scorer()
