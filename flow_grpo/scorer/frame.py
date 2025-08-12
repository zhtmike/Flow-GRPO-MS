from typing import List, Optional, Union

import mindspore as ms
import numpy as np

from ._scorer import Scorer


class FrameSmoothnessScorer(Scorer):

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
            result = result.reshape(result.shape[0], -1)
            result = np.linalg.norm(result, axis=1) / result.shape[1] ** 0.5
            result = result.max()
            diffs.append(result)

        rewards = [x / 255 for x in diffs]
        return rewards


def test_frame_smoothness_scorer():
    scorer = FrameSmoothnessScorer()
    videos = ["assets/video.npy"]
    np_videos = [np.load(video) for video in videos]
    print(scorer(videos=np_videos))


if __name__ == "__main__":
    test_frame_smoothness_scorer()
