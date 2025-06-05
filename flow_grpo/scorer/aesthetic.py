import os
from typing import Dict, List, Optional, Union

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import ml_dtypes
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from mindone.transformers import CLIPModel
from PIL import Image
from transformers import CLIPProcessor

from .scorer import Scorer


class AestheticScorer(Scorer):
    _DEFAULT_MODEL = "openai/clip-vit-large-patch14"

    def __init__(self, dtype: ms.Type = ms.bfloat16) -> None:
        super().__init__()
        model_path = os.environ.get("CLIP_PATH", self._DEFAULT_MODEL)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        with nn.no_init_parameters():
            self.clip = CLIPModel.from_pretrained(model_path,
                                                  mindspore_dtype=dtype)
            self.mlp = MLP(dtype=dtype)
        self.dtype = dtype

        pth_path = hf_hub_download(
            repo_id="camenduru/improved-aesthetic-predictor",
            filename="sac+logos+ava1-l14-linearMSE.pth")
        state_dict = self.load_pth(pth_path, dtype)
        ms.load_param_into_net(self.mlp, state_dict, strict_load=True)
        self.clip.set_train(False)
        self.mlp.set_train(False)

    @staticmethod
    def load_pth(pth_path: str, dtype: ms.Type) -> Dict[str, ms.Tensor]:
        torch_data = torch.load(pth_path, map_location="cpu")
        mindspore_data = dict()
        for name, value in torch_data.items():
            if value.dtype == torch.bfloat16:
                mindspore_data[name] = ms.Parameter(
                    ms.Tensor(value.view(dtype=torch.uint16).numpy().view(
                        ml_dtypes.bfloat16),
                              dtype=dtype))
            else:
                mindspore_data[name] = ms.Parameter(
                    ms.Tensor(value.numpy(), dtype=dtype))
        return mindspore_data

    def __call__(self,
                 images: Union[List[Image.Image], np.ndarray, ms.Tensor],
                 prompts: Optional[List[str]] = None) -> List[float]:
        if isinstance(images, (np.ndarray, ms.Tensor)):
            images = self.array_to_images(images)

        inputs = self.processor(images=images, return_tensors="np")
        for k, v in inputs.items():
            inputs[k] = ms.Tensor(v)
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / mint.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1).float().tolist()


class MLP(nn.Cell):

    def __init__(self, dtype: ms.Type = ms.float32):
        super().__init__()
        self.layers = nn.SequentialCell(mint.nn.Linear(768, 1024, dtype=dtype),
                                        mint.nn.Dropout(0.2),
                                        mint.nn.Linear(1024, 128, dtype=dtype),
                                        mint.nn.Dropout(0.2),
                                        mint.nn.Linear(128, 64, dtype=dtype),
                                        mint.nn.Dropout(0.1),
                                        mint.nn.Linear(64, 16, dtype=dtype),
                                        mint.nn.Linear(16, 1, dtype=dtype))

    def construct(self, embed):
        return self.layers(embed)


def test_aesthetic_scorer():
    scorer = AestheticScorer()
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images))


if __name__ == "__main__":
    test_aesthetic_scorer()
