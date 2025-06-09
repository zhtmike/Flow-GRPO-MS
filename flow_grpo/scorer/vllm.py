import asyncio
import base64
import os
import re
from io import BytesIO
from typing import List, Optional, Union

import Levenshtein
import mindspore as ms
import numpy as np
from openai import AsyncOpenAI
from PIL import Image

from .scorer import Scorer


class VLLMScorer(Scorer):

    def __init__(self, base_url: Optional[str] = None):
        # following https://github.com/openai/openai-python/issues/1254
        # we should use a single event loop for AsyncOpenAI call
        self._loop = asyncio.new_event_loop()
        self.aclient = AsyncOpenAI(base_url=base_url, api_key="EMPTY")

    async def async_process_queries(self, queries: List[str], model_path: str,
                                    base_url: str) -> List[str]:
        results = await asyncio.gather(
            *(self._async_query_openai(query, model_path, base_url)
              for query in queries))
        return results

    async def _async_query_openai(self, query: str, model_path: str,
                                  base_url: str) -> str:
        completion = await self.aclient.chat.completions.create(
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

    def __call__(self,
                 images: Union[List[Image.Image], np.ndarray, ms.Tensor],
                 prompts: Optional[List[str]] = None) -> List[float]:
        raise NotImplementedError(
            "This method should be implemented in subclasses.")


class QwenVLVLLMScorer(VLLMScorer):
    _DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    _task = (
        "Your role is to evaluate the aesthetic quality score of given images.\n"
        "1. Bad: Extremely blurry, underexposed with significant noise, indiscernible subjects, and chaotic composition.\n"
        "2. Poor: Noticeable blur, poor lighting, washed-out colors, and awkward composition with cut-off subjects.\n"
        "3. Fair: In focus with adequate lighting, dull colors, decent composition but lacks creativity.\n"
        "4. Good: Sharp, good exposure, vibrant colors, thoughtful composition with a clear focal point.\n"
        "5. Excellent: Exceptional clarity, perfect exposure, rich colors, masterful composition with emotional impact.\n"
        "Please first provide a detailed analysis of the evaluation process, including the criteria for judging aesthetic quality, within the <Thought> tag. "
        "Then, give a final score from 1 to 5 within the <Score> tag.\n"
        "<Thought>\n"
        "[Analyze the evaluation process in detail here]\n"
        "</Thought>\n"
        "<Score>X</Score>")

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = os.environ.get("QWEN_VL_VLLM_URL", base_url)
        self.model_path = os.environ.get("QWEN_VL_PATH", self._DEFAULT_MODEL)
        super().__init__(base_url=self.base_url)

    def __call__(self,
                 images: Union[List[Image.Image], np.ndarray, ms.Tensor],
                 prompts: Optional[List[str]] = None) -> List[float]:
        if isinstance(images, (np.ndarray, ms.Tensor)):
            images = self.array_to_images(images)

        images_base64 = [self.pil_image_to_base64(image) for image in images]
        queries = [
            self.prepare_query(image_base64) for image_base64 in images_base64
        ]
        results = self._loop.run_until_complete(
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
                "text": self._task
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
        scores = [max(0, min(1, score)) for score in scores]
        return scores


class UnifiedRewardVLLMScorer(VLLMScorer):
    _DEFAULT_MODEL = "CodeGoat24/UnifiedReward-qwen-7b"
    _task = (
        "You are given a text caption and a generated image based on that caption. "
        "Your task is to evaluate this image based on two key criteria:\n"
        "1. Alignment with the Caption: Assess how well this image aligns with the provided caption. "
        "Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n"
        "2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, "
        "color accuracy, and overall aesthetic appeal.\nExtract key elements from the provided text caption, "
        "evaluate their presence in the generated image using the format: "
        "'element (type): value' (where value=0 means not generated, and value=1 means generated), "
        "and assign a score from 1 to 5 after 'Final Score:'.\n"
        "Your task is provided as follows:\n"
        "Text Caption: [{prompt}]")

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = os.environ.get("UNIFIED_REWARD_VLLM_URL", base_url)
        self.model_path = os.environ.get("UNIFIED_REWARD_PATH",
                                         self._DEFAULT_MODEL)
        super().__init__(base_url=self.base_url)

    def __call__(self, images: Union[List[Image.Image], np.ndarray, ms.Tensor],
                 prompts: List[str]) -> List[float]:
        if isinstance(images, (np.ndarray, ms.Tensor)):
            images = self.array_to_images(images)

        images_base64 = [self.pil_image_to_base64(image) for image in images]
        queries = [
            self.prepare_query(image_base64, prompt)
            for image_base64, prompt in zip(images_base64, prompts)
        ]
        results = self._loop.run_until_complete(
            self.async_process_queries(queries, self.model_path,
                                       self.base_url))
        rewards = self.extract_scores(results)
        return rewards

    def prepare_query(self, image_base64: str, prompt: str) -> List:
        query = [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_base64
                },
            },
            {
                "type": "text",
                "text": self._task.format(prompt=prompt)
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
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in output_text:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)) / 5)
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        scores = [max(0, min(1, score)) for score in scores]
        return scores


class QwenVLOCRVLLMScorer(VLLMScorer):
    _DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    _task = "Please output only the text content from the image without any additional descriptions or formatting."

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = os.environ.get("QWEN_VL_OCR_VLLM_URL", base_url)
        self.model_path = os.environ.get("QWEN_VL_OCR_PATH",
                                         self._DEFAULT_MODEL)
        super().__init__(base_url=self.base_url)

    def __call__(self, images: Union[List[Image.Image], np.ndarray, ms.Tensor],
                 prompts: List[str]) -> List[float]:
        if isinstance(images, (np.ndarray, ms.Tensor)):
            images = self.array_to_images(images)

        images_base64 = [self.pil_image_to_base64(image) for image in images]
        queries = [
            self.prepare_query(image_base64) for image_base64 in images_base64
        ]
        results = self._loop.run_until_complete(
            self.async_process_queries(queries, self.model_path,
                                       self.base_url))
        rewards = self.calculate_score(results, prompts)
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
                "text": self._task
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
    def calculate_score(output_text: List[str],
                        prompts: List[str]) -> List[float]:
        scores = []
        for text, prompt in zip(output_text, prompts):
            # assume the prompt is in the format: xxx display/show with "words" xxx
            prompt = prompt.split('"')[1]
            # remove any nonvisible characters and convert to lowercase
            prompt = re.sub(r'\s+', '', prompt).lower()
            text = re.sub(r'\s+', '', text).lower()
            if prompt in text:
                dist = 0
            else:
                dist = Levenshtein.distance(text, prompt)

            # recognized many unrelated characters, only add one character penalty
            if dist > len(prompt):
                dist = len(prompt)

            score = 1 - dist / len(prompt)
            scores.append(score)

        return scores


def test_qwen_vl_vllm_scorer():
    scorer = QwenVLVLLMScorer("http://0.0.0.0:9529/v1")
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images))


def test_qwen_vl_ocr_vllm_scorer():
    scorer = QwenVLOCRVLLMScorer("http://0.0.0.0:9529/v1")
    images = [
        "assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg",
        "assets/ocr.jpg"
    ]
    prompts = ['a photo of displaying "OCR".'] * len(images)
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images, prompts=prompts))


def test_unified_reward_vllm_scorer():
    scorer = UnifiedRewardVLLMScorer("http://0.0.0.0:9529/v1")
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    prompts = ["a photo of apple."] * len(images)
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images, prompts=prompts))


if __name__ == "__main__":
    # test_qwen_vl_vllm_scorer()
    # test_qwen_vl_ocr_vllm_scorer()
    test_unified_reward_vllm_scorer()
