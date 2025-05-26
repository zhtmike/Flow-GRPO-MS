# TODO: need to refactor
from typing import List, Optional, Tuple, Union

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn.functional as F
import numpy as np
from mindone.transformers import CLIPTextModelWithProjection, T5EncoderModel
from transformers import CLIPTokenizer, T5Tokenizer


def _encode_prompt_with_t5(
    text_encoder: T5EncoderModel,
    tokenizer: T5Tokenizer,
    max_sequence_length: int,
    prompt: str,
    num_images_per_prompt: int = 1,
    text_input_ids: Optional[np.ndarray] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(ms.tensor(text_input_ids),
                                 return_dict=True)[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt,
                                       seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder: CLIPTextModelWithProjection,
    tokenizer: CLIPTokenizer,
    prompt: str,
    text_input_ids: Optional[np.ndarray] = None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(ms.tensor(text_input_ids),
                                 output_hidden_states=True,
                                 return_dict=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt,
                                       seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: Union[List[str], str],
    max_sequence_length: int,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
) -> Tuple[ms.Tensor, ms.Tensor]:
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer,
            text_encoder) in enumerate(zip(clip_tokenizers,
                                           clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i]
            if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = mint.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = mint.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1]
        if text_input_ids_list else None,
    )

    clip_prompt_embeds = F.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]))
    prompt_embeds = mint.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    prompt_embeds = prompt_embeds.float()
    pooled_prompt_embeds = pooled_prompt_embeds.float()
    return prompt_embeds, pooled_prompt_embeds
