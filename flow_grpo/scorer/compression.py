import io
import os
import tempfile
from typing import List, Optional, Union

import mindspore as ms
import numpy as np
from mindone.diffusers.utils import export_to_video
from PIL import Image

from ._scorer import Scorer

__all__ = [
    "JpegCompressibilityScorer",
    "JpegImcompressibilityScorer",
    "MP4CompressibilityScorer",
    "MP4ImcompressibilityScorer",
]


class JpegCompressibilityScorer(Scorer):

    def __init__(self, max_size: int = 256, quality: int = 95) -> None:
        self.max_size = max_size
        self.quality = quality

    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, ms.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        if isinstance(images, (np.ndarray, ms.Tensor)):
            images = self.array_to_images(images)

        sizes = []
        for image in images:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=self.quality)
            # compressed image file size (kb)
            sizes.append(buffer.tell() / 1024)

        # 256 kb: score 0; 0 kb: score 1
        rewards = [max(0.0, 1 - x / self.max_size) for x in sizes]
        return rewards


class JpegImcompressibilityScorer(JpegCompressibilityScorer):

    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, ms.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        rewards = super().__call__(images, prompts)
        rewards = [1.0 - r for r in rewards]
        return rewards


class MP4CompressibilityScorer(Scorer):

    def __init__(self, max_size: int = 2, fps: int = 15) -> None:
        self.max_size = max_size
        self.fps = fps

    def __call__(
        self,
        videos: Union[List[np.ndarray], List[ms.Tensor]],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        sizes = []
        for video in videos:
            if isinstance(video, ms.Tensor):
                video = video.numpy()

            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmpfile:
                export_to_video(video, tmpfile.name, fps=self.fps)
                filesize = os.path.getsize(tmpfile.name)
                # compressed mp4 file size (mb)
                sizes.append(filesize / 1024 / 1024)

        # 2 mb: score 0; 0 mb: score 1
        rewards = [max(0.0, 1 - x / self.max_size) for x in sizes]
        return rewards


class MP4ImcompressibilityScorer(MP4CompressibilityScorer):

    def __call__(
        self,
        videos: Union[List[np.ndarray], List[ms.Tensor]],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        rewards = super().__call__(videos, prompts)
        rewards = [1.0 - r for r in rewards]
        return rewards


def test_jpeg_compressibility_scorer():
    scorer = JpegCompressibilityScorer()
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images))


def test_mp4_compressibility_scorer():
    scorer = MP4CompressibilityScorer()
    videos = ["assets/video.npy"]
    np_videos = [np.load(video) for video in videos]
    print(scorer(videos=np_videos))


if __name__ == "__main__":
    test_jpeg_compressibility_scorer()
    test_mp4_compressibility_scorer()
