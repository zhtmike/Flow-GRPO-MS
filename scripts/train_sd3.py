import argparse
import datetime

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
from mindone.diffusers import StableDiffusion3Pipeline

from flow_grpo.util import requires_grad_


def train(args: argparse.Namespace):
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    run_name = unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(args.num_steps * args.timestep_fraction)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model)

    # freeze parameters of models to save more memory
    requires_grad_(pipeline.vae, False)
    requires_grad_(pipeline.text_encoder, False)
    requires_grad_(pipeline.text_encoder_2, False)
    requires_grad_(pipeline.text_encoder_3, False)
    requires_grad_(pipeline.transformer, True)  # TODO: set to False for LoRA

    text_encoders = [
        pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3
    ]
    tokenizers = [
        pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3
    ]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=False,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = ms.float32
    if args.mixed_precision == "fp16":
        inference_dtype = ms.float16
    elif args.mixed_precision == "bf16":
        inference_dtype = ms.bfloat16

    # cast inference time
    pipeline.vae.to(ms.float32)
    pipeline.text_encoder.to(inference_dtype)
    pipeline.text_encoder_2.to(inference_dtype)
    pipeline.text_encoder_3.to(inference_dtype)

    transformer: nn.Cell = pipeline.transformer
    transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer.get_parameters()))

    # TODO: add EMA
    optimizer = mint.optim.AdamW(transformer_trainable_parameters,
                                 lr=args.learning_rate,
                                 betas=(args.adam_beta1, args.adam_beta2),
                                 weight_decay=args.adam_weight_decay,
                                 eps=args.adam_epsilon)


def main():
    parser = argparse.ArgumentParser(usage="train sd3 with grpo")
    args = parser.parse_args()

    # TODO: hard coded, remove later
    args.num_steps = 10
    args.timestep_fraction = 0.99
    args.model = "stabilityai/stable-diffusion-3.5-medium"
    args.mixed_precision = "fp16"
    args.learning_rate = 3e-4
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_weight_decay = 1e-4
    args.adam_epsilon = 1e-8


if __name__ == "__main__":
    main()
