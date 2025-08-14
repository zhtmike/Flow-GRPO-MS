from typing import List, Optional, Union

import mindspore as ms
import numpy as np

from ._scorer import Scorer


class ContrastChangeScorer(Scorer):
    def __call__(
        self,
        videos: Union[List[np.ndarray], List[ms.Tensor]],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        rewards = []
        for video in videos:
            if isinstance(video, ms.Tensor):
                video = video.numpy().astype(np.float32)

            diff = video[-1] - video[0]
            diff = np.linalg.norm(diff) / diff.size**0.5
            rewards.append(diff)

        return rewards


def test_contrast_change_scorer():
    scorer = ContrastChangeScorer()
    videos = np.zeros((9, 224, 224, 3), dtype=np.float32)
    videos[0] = 1.0
    np_videos = [videos]
    print(scorer(videos=np_videos))


if __name__ == "__main__":
    test_contrast_change_scorer()
