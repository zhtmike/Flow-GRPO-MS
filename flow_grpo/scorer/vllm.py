import asyncio
import base64
import os
import re
from io import BytesIO
from typing import List, Optional, Union

import mindspore as ms
import numpy as np
from openai import AsyncOpenAI
from PIL import Image

from .scorer import Scorer


class VLLMSScorer(Scorer):

    @staticmethod
    async def async_process_queries(queries: List[str], model_path: str,
                                    base_url: str) -> List[str]:
        results = await asyncio.gather(
            *(VLLMSScorer._async_query_openai(query, model_path, base_url)
              for query in queries))
        return results

    @staticmethod
    async def _async_query_openai(query: str, model_path: str,
                                  base_url: str) -> str:
        aclient = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        completion = await aclient.chat.completions.create(
            model=model_path,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            temperature=0,
        )
        return completion.choices[0].message.content


class QwenVLVLLMScorer(VLLMSScorer):
    _DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    _task = """
        Your role is to evaluate the aesthetic quality score of given images.
        1. Bad: Extremely blurry, underexposed with significant noise, indiscernible subjects, and chaotic composition.
        2. Poor: Noticeable blur, poor lighting, washed-out colors, and awkward composition with cut-off subjects.
        3. Fair: In focus with adequate lighting, dull colors, decent composition but lacks creativity.
        4. Good: Sharp, good exposure, vibrant colors, thoughtful composition with a clear focal point.
        5. Excellent: Exceptional clarity, perfect exposure, rich colors, masterful composition with emotional impact.

        Please first provide a detailed analysis of the evaluation process, including the criteria for judging aesthetic quality, within the <Thought> tag. Then, give a final score from 1 to 5 within the <Score> tag.
        <Thought>
        [Analyze the evaluation process in detail here]
        </Thought>
        <Score>X</Score>
    """

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.model_path = os.environ.get("QWEN_VL_PATH", self._DEFAULT_MODEL)

    def __call__(self,
                 images: Union[List[Image.Image], np.ndarray, ms.Tensor],
                 prompts: Optional[List[str]] = None) -> List[float]:
        if isinstance(images, (np.ndarray, ms.Tensor)):
            images = self.array_to_images(images)

        images_base64 = [self.pil_image_to_base64(image) for image in images]
        queries = [
            self.prepare_query(image_base64) for image_base64 in images_base64
        ]
        results = asyncio.run(
            self.async_process_queries(queries, self.model_path,
                                       self.base_url))
        rewards = self.extract_scores(results)
        return rewards

    def prepare_query(self, image_base64: str) -> List:
        query = [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_base64
                },
            },
            {
                "type": "text",
                "text": self.task
            },
        ]
        return query

    @staticmethod
    def pil_image_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(
            buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    @staticmethod
    def extract_scores(output_text: List[str]) -> List[float]:
        scores = []
        for text in output_text:
            match = re.search(r'<Score>(\d+)</Score>', text)
            if match:
                scores.append(float(match.group(1)) / 5)
            else:
                scores.append(0)
        return scores


def test_qwen_vl_vllm_scorer():
    scorer = QwenVLVLLMScorer("http://0.0.0.0:9529/v1")
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images))


if __name__ == "__main__":
    test_qwen_vl_vllm_scorer()
