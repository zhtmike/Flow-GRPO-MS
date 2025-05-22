from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict

import numpy as np
from PIL import Image


class Scorer(ABC):

    @abstractmethod
    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray],
        prompts: Optional[List[str]] = None
    ) -> Union[List[float], Dict[str, List[float]]]:
        """Return the scoring value of the images
        """
        pass

    @staticmethod
    def array_to_images(images: np.ndarray) -> List[Image.Image]:
        images = (images * 255).round().clamp(0, 255).to(np.uint8)
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        return images
