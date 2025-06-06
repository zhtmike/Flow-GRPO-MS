import base64
import os
import re
from io import BytesIO
from typing import List, Optional, Union

import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindone.transformers import Qwen2_5_VLForConditionalGeneration
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLProcessor

from .scorer import Scorer


class QwenVLScorer(Scorer):
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

    def __init__(self, dtype: ms.Type = ms.bfloat16) -> None:
        super().__init__()
        model_path = os.environ.get("QWEN_VL_PATH", self._DEFAULT_MODEL)
        with nn.no_init_parameters():
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                mindspore_dtype=dtype,
                attn_implementation="flash_attention_2",
            )
        self.processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(
            model_path, use_fast=False, padding_side="left")

    def __call__(self,
                 images: Union[List[Image.Image], np.ndarray, ms.Tensor],
                 prompts: Optional[List[str]] = None) -> List[float]:
        if isinstance(images, (np.ndarray, ms.Tensor)):
            images = self.array_to_images(images)

        images_base64 = [self.pil_image_to_base64(image) for image in images]
        messages = []
        for base64_qwen in images_base64:
            messages.append([
                {
                    "role":
                    "user",
                    "content": [
                        {
                            "type": "image",
                            "image": base64_qwen
                        },
                        {
                            "type": "text",
                            "text": self._task
                        },
                    ],
                },
            ])

        # Preparation for batch inference
        texts = [
            self.processor.apply_chat_template(msg,
                                               tokenize=False,
                                               add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="np",
        )
        for k, v in inputs.items():
            inputs[k] = ms.Tensor(v)

        # Batch Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        rewards = self.extract_scores(output_texts)
        return rewards

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


def test_qwen_vl_scorer():
    scorer = QwenVLScorer(dtype=ms.bfloat16)
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images))


if __name__ == "__main__":
    test_qwen_vl_scorer()
