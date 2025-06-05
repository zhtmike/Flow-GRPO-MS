import os
from typing import List, Optional, Union

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import numpy as np
from mindone.transformers import AutoModel, CLIPModel
from PIL import Image
from transformers import AutoProcessor

from .scorer import Scorer


class PickScoreScorer(Scorer):
    _DEFAULT_MODEL = "yuvalkirstain/PickScore_v1"

    def __init__(self, dtype: ms.Type = ms.bfloat16):
        super().__init__()
        model_path = os.environ.get("PICKSCORE_PATH", self._DEFAULT_MODEL)
        self.processor = AutoProcessor.from_pretrained(model_path)
        with nn.no_init_parameters():
            self.model: CLIPModel = AutoModel.from_pretrained(
                model_path, mindspore_dtype=dtype)

    def __call__(self,
                 images: Union[List[Image.Image], np.ndarray, ms.Tensor],
                 prompts: Optional[List[str]] = None) -> List[float]:
        if isinstance(images, (np.ndarray, ms.Tensor)):
            images = self.array_to_images(images)

        # Preprocess images
        image_inputs = self.processor(images=images,
                                      padding=True,
                                      truncation=True,
                                      max_length=77,
                                      return_tensors="np")
        for k, v in image_inputs.items():
            image_inputs[k] = ms.Tensor(v)

        # Preprocess text
        text_inputs = self.processor(text=prompts,
                                     padding=True,
                                     truncation=True,
                                     max_length=77,
                                     return_tensors="np")
        for k, v in text_inputs.items():
            text_inputs[k] = ms.Tensor(v)

        # Get embeddings
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / mint.norm(
            image_embs, p=2, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / mint.norm(text_embs, p=2, dim=-1, keepdim=True)

        # Calculate scores
        scores = text_embs @ image_embs.T
        scores = scores.diag()
        return scores.tolist()


def test_pickscore_scorer():
    scorer = PickScoreScorer()
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    prompts = ["photo of apple"] * len(images)
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images, prompts=prompts))


if __name__ == "__main__":
    test_pickscore_scorer()
