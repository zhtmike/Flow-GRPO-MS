import importlib
from typing import Dict, List, Optional, Union

import mindspore as ms
import numpy as np
from PIL import Image

from .scorer import Scorer

AVAILABLE_SCORERS = {
    "aesthetic": ("aesthetic", "AestheticScorer"),
    "jpeg-compressibility": ("compression", "JpegCompressibilityScorer"),
    "jpeg-imcompressibility": ("compression", "JpegImcompressibilityScorer"),
    "pickscore": ("pickscore", "PickScoreScorer"),
    "qwenvl": ("qwenvl", "QwenVLScorer"),
    "qwenvl-vllm": ("vllm", "QwenVLVLLMScorer"),
    "unified-reward-vllm": ("vllm", "UnifiedRewardVLLMScorer"),
}


class MultiScorer(Scorer):

    def __init__(self, scorers: Dict[str, float]) -> None:
        self.score_fn = dict()
        self.scorers = scorers
        self.init_scorer_cls()

    def init_scorer_cls(self):
        for score_name in self.scorers.keys():
            module, cls = AVAILABLE_SCORERS[score_name]
            module = "flow_grpo.scorer." + module
            module = importlib.import_module(module)
            cls = getattr(module, cls)
            self.score_fn[score_name] = cls()

    def __call__(
            self,
            images: Union[List[Image.Image], np.ndarray, ms.Tensor],
            prompts: Optional[List[str]] = None) -> Dict[str, List[float]]:
        score_details = dict()
        total_scores = list()
        for score_name, weight in self.scorers.items():
            scores = self.score_fn[score_name](images, prompts=prompts)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [
                    total + weighted
                    for total, weighted in zip(total_scores, weighted_scores)
                ]

        score_details['avg'] = total_scores
        return score_details


def test_multi_scorer():
    scorers = {"jpeg-compressibility": 1.0}
    scorer = MultiScorer(scorers)
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images))


if __name__ == "__main__":
    test_multi_scorer()
