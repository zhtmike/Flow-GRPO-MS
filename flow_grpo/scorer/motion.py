from typing import List, Optional, Union

import mindspore as ms
import numpy as np

from ._scorer import Scorer


class MotionSmoothnessScorer(Scorer):

    def __call__(
        self,
        videos: Union[List[np.ndarray], List[ms.Tensor]],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        diffs = []
        for video in videos:
            if isinstance(video, ms.Tensor):
                video = video.numpy()

            result = np.diff(video, axis=0)
            result = np.abs(result)
            result = np.maximum(result)
            diffs.append(result)

        rewards = [x / 255 for x in diffs]
        return rewards


def test_motion_smoothness_scorer():
    scorer = MotionSmoothnessScorer()
    videos = ["assets/video.npy"]
    np_videos = [np.load(video) for video in videos]
    print(scorer(videos=np_videos))
