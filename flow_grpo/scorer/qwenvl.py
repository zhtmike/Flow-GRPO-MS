import base64
import re
from io import BytesIO
from typing import List

import mindspore as ms
from mindone.transformers import Qwen2_5_VLForConditionalGeneration
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLProcessor

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def pil_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen


def extract_scores(output_text):
    scores = []
    for text in output_text:
        match = re.search(r'<Score>(\d+)</Score>', text)
        if match:
            scores.append(float(match.group(1)) / 5)
        else:
            scores.append(0)
    return scores


class QwenVLScorer:

    def __init__(self, dtype: ms.Type = ms.bfloat16) -> None:
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        self.processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(
            MODEL, use_fast=True)
        self.task = """
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

    def __call__(self, images: List[Image.Image]):
        images_base64 = [pil_image_to_base64(image) for image in images]
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
                            "text": self.task
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
        rewards = extract_scores(output_texts)
        return rewards


def test_qwen_vl_scorer():
    scorer = QwenVLScorer(dtype=ms.bfloat16)
    images = ["../../assets/demo.jpg"]
    pil_images = [Image.open(img) for img in images]
    print(scorer(None, pil_images))


if __name__ == "__main__":
    test_qwen_vl_scorer()
