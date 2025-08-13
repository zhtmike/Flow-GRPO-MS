from typing import List, Optional, Union

import mindspore as ms
import numpy as np

from ._scorer import Scorer


class ContrastChangeScorer(Scorer):
    def __init__(self, frame_stride: int = 8) -> None:
        self.frame_stride = frame_stride

    def __call__(
        self,
        videos: Union[List[np.ndarray], List[ms.Tensor]],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        rewards = []
        for video in videos:
            if isinstance(video, ms.Tensor):
                video = video.numpy()
            assert video.shape[0] > self.frame_stride

            video = video[:: self.frame_stride].astype(np.float32)
            result = np.diff(video, axis=0)
            result = result.reshape(result.shape[0], -1)
            result = np.linalg.norm(result, axis=1) / result.shape[1] ** 0.5
            result = result.mean()
            rewards.append(result)

        return rewards


def test_contrast_change_scorer():
    scorer = ContrastChangeScorer()
    videos = np.zeros((9, 224, 224, 3), dtype=np.uint8)
    videos[0] = 255
    np_videos = [videos]
    print(scorer(videos=np_videos))


if __name__ == "__main__":
    test_contrast_change_scorer()
