import argparse
import datetime
import logging
import os
import time
from collections import defaultdict
from concurrent import futures
from functools import partial
from typing import Any, List, Optional

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.distributed as dist
import mindspore.nn as nn
import numpy as np
from mindone.diffusers import (AutoencoderKL, FlowMatchEulerDiscreteScheduler,
                               SD3Transformer2DModel, StableDiffusion3Pipeline)
from mindone.diffusers._peft import LoraConfig, PeftModel, get_peft_model
from mindone.transformers import CLIPTextModelWithProjection, T5EncoderModel
from mindspore.dataset import DictIterator, GeneratorDataset
from tqdm import tqdm
from transformers import CLIPTextConfig, T5Config

from flow_grpo.dataset import DistributedKRepeatSampler, TextPromptDataset
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import \
    pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import \
    sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.scorer import MultiScorer
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.util import (clip_by_global_norm, gather, map_, requires_grad_,
                            save_checkpoint, syn_gradients)

DEFAULT_MODEL = "stabilityai/stable-diffusion-3.5-medium"

tqdm_ = partial(tqdm, dynamic_ncols=True)

logger = logging.getLogger(__name__)


def init_debug_pipeline(args: argparse.Namespace) -> StableDiffusion3Pipeline:
    """Init the pipeline with models containing only 1 layers for easier debugging & faster creating"""
    vae_config = AutoencoderKL.load_config(args.model, subfolder="vae")
    vae_config["layers_per_block"] = 1
    vae = AutoencoderKL.from_config(vae_config)

    transformer_config = SD3Transformer2DModel.load_config(
        args.model, subfolder="transformer")
    transformer_config["num_layers"] = 1
    transformer = SD3Transformer2DModel.from_config(transformer_config)

    text_encoder_config = CLIPTextConfig.from_pretrained(
        args.model, subfolder="text_encoder")
    text_encoder_config.num_hidden_layers = 1
    text_encoder = CLIPTextModelWithProjection(text_encoder_config)

    text_encoder_2_config = CLIPTextConfig.from_pretrained(
        args.model, subfolder="text_encoder_2")
    text_encoder_2_config.num_hidden_layers = 1
    text_encoder_2 = CLIPTextModelWithProjection(text_encoder_2_config)

    text_encoder_3_config = T5Config.from_pretrained(
        args.model, subfolder="text_encoder_3")
    text_encoder_3_config.num_layers = 1
    text_encoder_3 = T5EncoderModel(text_encoder_3_config)

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.model,
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3)
    return pipeline


class NetWithLoss(nn.Cell):

    def __init__(self, transformer: SD3Transformer2DModel,
                 scheduler: FlowMatchEulerDiscreteScheduler,
                 args: argparse.Namespace) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.args = args

    def compute_log_prob(self, latents, next_latents, timesteps, embeds,
                         pooled_embeds, sigma, sigma_prev):
        if self.args.cfg:
            noise_pred = self.transformer(
                hidden_states=mint.cat([latents] * 2),
                timestep=mint.cat([timesteps] * 2),
                encoder_hidden_states=embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = (noise_pred_uncond + self.args.guidance_scale *
                          (noise_pred_text - noise_pred_uncond))
        else:
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timesteps,
                encoder_hidden_states=embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

        # compute the log prob of next_latents given latents under the current model
        prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
            self.scheduler,
            noise_pred.float(),
            timesteps,
            latents.float(),
            prev_sample=next_latents.float(),
            sigma=sigma,
            sigma_prev=sigma_prev)

        return prev_sample, log_prob, prev_sample_mean, std_dev_t

    def construct(self,
                  latents: ms.Tensor,
                  next_latents: ms.Tensor,
                  timesteps: ms.Tensor,
                  embeds: ms.Tensor,
                  pooled_embeds: ms.Tensor,
                  advantages: ms.Tensor,
                  sample_log_probs: ms.Tensor,
                  sigma: ms.Tensor,
                  sigma_prev: ms.Tensor,
                  prev_sample_mean_ref: Optional[ms.Tensor] = None,
                  loss_scaler: Optional[ms.Tensor] = None) -> ms.Tensor:
        if self.args.beta > 0:
            assert prev_sample_mean_ref is not None

        _, log_prob, prev_sample_mean, std_dev_t = self.compute_log_prob(
            latents, next_latents, timesteps, embeds, pooled_embeds, sigma,
            sigma_prev)
        # grpo logic
        advantages = mint.clamp(
            advantages,
            -self.args.adv_clip_max,
            self.args.adv_clip_max,
        )
        ratio = mint.exp(log_prob - sample_log_probs)
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * mint.clamp(
            ratio,
            1.0 - self.args.clip_range,
            1.0 + self.args.clip_range,
        )
        policy_loss = mint.mean(mint.maximum(unclipped_loss, clipped_loss))
        if self.args.beta > 0:
            kl_loss = ((prev_sample_mean - prev_sample_mean_ref)**2).mean(
                dim=(1, 2, 3), keepdim=True) / (2 * std_dev_t**2)
            kl_loss = mint.mean(kl_loss)
            loss = policy_loss + self.args.beta * kl_loss
        else:
            loss = policy_loss

        if loss_scaler is not None:
            loss = loss * loss_scaler

        return loss


def evaluate(pipeline_with_logprob_, args: argparse.Namespace,
             test_iter: DictIterator, pipeline: StableDiffusion3Pipeline,
             text_encoders: List[nn.Cell], tokenizers: List[Any],
             sample_neg_prompt_embeds: ms.Tensor,
             sample_neg_pooled_prompt_embeds: ms.Tensor, ema: EMAModuleWrapper,
             parameters: ms.ParameterTuple, outdir: str,
             executor: futures.ThreadPoolExecutor,
             reward_fn: MultiScorer) -> None:
    if args.ema:
        ema.copy_ema_to(parameters, store_temp=True)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    total_prompts = list()
    all_rewards = defaultdict(list)
    for i, test_batch in tqdm_(
            enumerate(test_iter),
            desc="Eval: ",
            position=0,
    ):
        prompts = test_batch["prompt"].tolist()
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders,
            tokenizers,
            prompts,
            max_sequence_length=128,
        )
        if len(prompt_embeds) < len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(
                prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(
                prompt_embeds)]
        images, _, _, _ = pipeline_with_logprob_(
            pipeline,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=sample_neg_prompt_embeds,
            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
            num_inference_steps=args.eval_num_steps,
            guidance_scale=args.guidance_scale,
            output_type="pil",
            return_dict=False,
            height=args.resolution,
            width=args.resolution,
            determistic=True,
            generator=np.random.default_rng(args.seed))

        # save the image for visualization
        for j, (prompt, image) in enumerate(zip(prompts, images)):
            num = i * len(images) + j
            fname = f"{num}.jpg"
            total_prompts.append((fname, prompt))
            image.save(os.path.join(outdir, fname))

        # calcuate the validation reward
        rewards = executor.submit(reward_fn, images, prompts)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards = rewards.result()
        for k, v in rewards.items():
            all_rewards[k].extend(v)

    avg_rewards = dict()
    for k, v in all_rewards.items():
        avg_rewards[k] = np.mean(v).item()

    logger.info(f"Validation rewards: {avg_rewards}")
    with open(os.path.join(outdir, "prompt.txt"), "w") as f:
        for fname, prompt in total_prompts:
            f.write(f"{fname},{prompt}\n")

    if args.ema:
        ema.copy_temp_to(parameters)
    return


def train(args: argparse.Namespace):
    dist.init_process_group()
    num_processes = dist.get_world_size()
    process_index = dist.get_rank()
    is_main_process = process_index == 0

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not args.run_name:
        args.run_name = unique_id
    else:
        args.run_name += "_" + unique_id
    output_dir = os.path.join("output", args.run_name)

    if args.resume_from:
        # TODO: support resume from
        raise NotImplementedError()

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(args.num_steps * args.timestep_fraction)

    logger.info(f"\n{args}")

    # set the same seed for model inialization
    ms.set_seed(args.seed)

    # load scheduler, tokenizer and models.
    if args.debug:
        ms.runtime.launch_blocking()
        pipeline = init_debug_pipeline(args)
    else:
        with nn.no_init_parameters():
            pipeline: StableDiffusion3Pipeline = StableDiffusion3Pipeline.from_pretrained(
                args.model)

    # freeze parameters of models to save more memory
    requires_grad_(pipeline.vae, False)
    requires_grad_(pipeline.text_encoder, False)
    requires_grad_(pipeline.text_encoder_2, False)
    requires_grad_(pipeline.text_encoder_3, False)
    requires_grad_(pipeline.transformer, not args.use_lora)

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
        disable=not is_main_process,
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

    if args.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if args.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, args.lora_path)
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer,
                                                  transformer_lora_config)

    trainable_parameters = ms.ParameterTuple(
        filter(lambda p: p.requires_grad,
               pipeline.transformer.get_parameters()))

    # print model size
    transformer_params = sum(
        [param.size for param in pipeline.transformer.get_parameters()])
    vae_params = sum([param.size for param in pipeline.vae.get_parameters()])
    text_encoder_params = sum(
        [param.size for param in pipeline.text_encoder.get_parameters()])
    text_encoder_2_params = sum(
        [param.size for param in pipeline.text_encoder_2.get_parameters()])
    text_encoder_3_params = sum(
        [param.size for param in pipeline.text_encoder_3.get_parameters()])
    total_params = transformer_params + vae_params + text_encoder_params + text_encoder_2_params + text_encoder_3_params
    trainable_params = sum([param.size for param in trainable_parameters])

    logger.info(
        f"total parameter: {total_params:,} (transformer: {transformer_params:,}, vae: {vae_params:,}, tex_encoder: {text_encoder_params:,}, text_encoder_2: {text_encoder_2_params:,}, text_encoder_3: {text_encoder_3_params:,})"
    )
    logger.info(f"total trainable parameter: {trainable_params:,}")

    if args.ema:
        ema = EMAModuleWrapper(trainable_parameters,
                               decay=0.9,
                               update_step_interval=1)

    optimizer = mint.optim.AdamW(trainable_parameters,
                                 lr=args.learning_rate,
                                 betas=(args.adam_beta1, args.adam_beta2),
                                 weight_decay=args.adam_weight_decay,
                                 eps=args.adam_epsilon)

    # prepare prompt and reward fn
    reward_fn = MultiScorer(args.reward_fn)

    # set seed (device_specific is very important to get different prompts on different devices)
    ms.set_seed(args.seed + process_index)

    train_dataset = TextPromptDataset(args.dataset, 'train')
    test_dataset = TextPromptDataset(args.dataset,
                                     'test',
                                     max_num=args.validation_num)
    train_sampler = DistributedKRepeatSampler(dataset=train_dataset,
                                              batch_size=args.train_batch_size,
                                              k=args.num_image_per_prompt,
                                              num_replicas=num_processes,
                                              rank=process_index,
                                              seed=args.seed)

    # create dataloader
    train_dataloader = GeneratorDataset(train_dataset,
                                        column_names="prompt",
                                        batch_sampler=train_sampler,
                                        num_parallel_workers=1)

    test_dataloader = GeneratorDataset(test_dataset,
                                       column_names="prompt",
                                       num_parallel_workers=1,
                                       shuffle=False)
    test_dataloader = test_dataloader.batch(args.test_batch_size,
                                            num_parallel_workers=1,
                                            drop_remainder=False)

    neg_prompt_embed, neg_pooled_prompt_embed = encode_prompt(
        text_encoders, tokenizers, [""], max_sequence_length=128)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(args.train_batch_size,
                                                       1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(args.train_batch_size, 1,
                                                      1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(
        args.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(
        args.train_batch_size, 1)

    if args.num_image_per_prompt == 1:
        args.per_prompt_stat_tracking = False

    # initialize stat tracker
    if args.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(args.global_std)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (args.train_batch_size * num_processes *
                         args.num_batches_per_epoch)
    total_train_batch_size = (args.train_batch_size * num_processes *
                              args.gradient_accumulation_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Train batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {args.num_inner_epochs}")

    if args.resume_from:
        raise NotImplementedError()
    else:
        first_epoch = 0
    global_step = 0
    train_iter = train_dataloader.create_dict_iterator(output_numpy=True)
    test_iter = test_dataloader.create_dict_iterator(output_numpy=True)

    pipeline_with_logprob_ = ms.amp.auto_mixed_precision(pipeline_with_logprob,
                                                         amp_level="auto",
                                                         dtype=inference_dtype)

    net_with_loss = NetWithLoss(pipeline.transformer, pipeline.scheduler, args)
    net_with_loss = ms.amp.auto_mixed_precision(net_with_loss,
                                                amp_level="auto",
                                                dtype=inference_dtype)

    loss_and_grad_fn = ms.value_and_grad(net_with_loss,
                                         grad_position=None,
                                         weights=optimizer.parameters)

    for epoch in range(first_epoch, args.num_epochs):
        #################### SAMPLING ####################
        pipeline.transformer.set_train(False)
        samples = []
        for i in tqdm_(range(args.num_batches_per_epoch),
                       desc=f"Epoch {epoch}: sampling",
                       disable=not is_main_process,
                       position=0):
            prompts = next(train_iter)["prompt"].tolist()

            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders,
                tokenizers,
                prompts,
                max_sequence_length=128,
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="np",
            ).input_ids
            if i == 0 and epoch % args.eval_freq == 0 and is_main_process:
                outdir = os.path.join(output_dir, "visual", f"epoch_{epoch}")
                evaluate(pipeline_with_logprob_, args, test_iter, pipeline,
                         text_encoders, tokenizers, sample_neg_prompt_embeds,
                         sample_neg_pooled_prompt_embeds, ema,
                         trainable_parameters, outdir, executor, reward_fn)
            if i == 0 and epoch % args.save_freq == 0 and epoch > 0 and is_main_process:
                save_checkpoint(trainable_parameters,
                                outdir=os.path.join(output_dir, "ckpt"))
            dist.barrier()

            images, latents, log_probs, kls = pipeline_with_logprob_(
                pipeline,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                output_type="np",
                return_dict=False,
                height=args.resolution,
                width=args.resolution,
                kl_reward=args.kl_reward)

            latents = mint.stack(
                latents,
                dim=1).numpy()  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = mint.stack(
                log_probs,
                dim=1).numpy()  # shape after stack (batch_size, num_steps)
            kls = mint.stack(kls, dim=1).numpy()

            timesteps = pipeline.scheduler.timesteps.repeat(
                args.train_batch_size, 1)  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append({
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds.numpy(),
                "pooled_prompt_embeds": pooled_prompt_embeds.numpy(),
                "timesteps": timesteps.numpy(),
                "latents":
                latents[:, :-1],  # each entry is the latent before timestep t
                "next_latents":
                latents[:, 1:],  # each entry is the latent after timestep t
                "log_probs": log_probs,
                "kl": kls,
                "rewards": rewards,
            })

        # wait for all rewards to be computed
        for sample in tqdm_(samples,
                            desc="Waiting for rewards",
                            disable=not is_main_process,
                            position=0):
            rewards = sample["rewards"].result()
            sample["rewards"] = rewards

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: np.concatenate([s[k] for s in samples], axis=0)
            if not isinstance(samples[0][k], dict) else {
                sub_key: np.concatenate([s[k][sub_key] for s in samples],
                                        axis=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"][
            ..., None] - args.kl_reward * samples["kl"]
        # gather rewards across processes
        gathered_rewards = {
            key: gather(ms.tensor(value))
            for key, value in samples["rewards"].items()
        }
        gathered_rewards = {
            key: value.numpy()
            for key, value in gathered_rewards.items()
        }

        # per-prompt mean/std tracking
        if args.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = gather(ms.tensor(samples["prompt_ids"])).numpy()
            prompts = pipeline.tokenizer.batch_decode(prompt_ids,
                                                      skip_special_tokens=True)
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            logger.debug("total number of prompts", len(prompts))
            logger.debug("total number of unique prompts", len(set(prompts)))
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] -
                          gathered_rewards['avg'].mean()) / (
                              gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = advantages.astype(np.float32)
        samples["advantages"] = advantages.reshape(
            num_processes, -1, advantages.shape[-1])[process_index]

        logger.debug("advantages: %s",
                     np.abs(samples["advantages"]).mean().item())
        logger.debug("kl: %s", samples["kl"].mean().item())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (np.abs(samples["advantages"]).sum(axis=1) != 0)

        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = args.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = np.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = np.random.permutation(
                    len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True

        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == args.num_steps

        logger.info({
            "global_step": global_step,
            "epoch": epoch,
            **{
                f"reward_{key}": value.mean().item()
                for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key
            }, "kl": samples["kl"].mean().item(),
            "advantages": np.abs(samples["advantages"]).mean().item(),
            "actual_batch_size": mask.sum().item() // num_batches
        })

        #################### TRAINING ####################
        for inner_epoch in range(args.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = np.random.permutation(total_batch_size)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = np.stack(
                [np.arange(num_timesteps) for _ in range(total_batch_size)])
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    np.arange(total_batch_size)[:, None],
                    perms,
                ]

            # rebatch for training
            samples_batched = {
                k:
                v.reshape(-1, total_batch_size // args.num_batches_per_epoch,
                          *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x))
                for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.set_train(True)
            grad_accumulated = None
            train_timesteps = list(range(num_train_timesteps))

            if args.gradient_accumulation_steps > 1:
                loss_scaler = ms.Tensor(1 / args.gradient_accumulation_steps)
            else:
                loss_scaler = None

            for i, sample in tqdm_(
                    enumerate(samples_batched),
                    desc=f"Epoch {epoch}.{inner_epoch}: training",
                    position=0,
                    disable=not is_main_process):
                if args.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = mint.cat([
                        train_neg_prompt_embeds[:len(sample["prompt_embeds"])],
                        ms.tensor(sample["prompt_embeds"])
                    ])
                    pooled_embeds = mint.cat([
                        train_neg_pooled_prompt_embeds[:len(
                            sample["pooled_prompt_embeds"])],
                        ms.tensor(sample["pooled_prompt_embeds"])
                    ])
                else:
                    embeds = ms.tensor(sample["prompt_embeds"])
                    pooled_embeds = ms.tensor(sample["pooled_prompt_embeds"])

                avg_loss = list()
                for j in tqdm_(
                        train_timesteps,
                        desc="Timestep",
                        position=1,
                        leave=False,
                        disable=not is_main_process,
                ):
                    latents = ms.tensor(sample["latents"][:, j])
                    next_latents = ms.tensor(sample["next_latents"][:, j])
                    timesteps = ms.tensor(sample["timesteps"][:, j])
                    advantages = ms.tensor(sample["advantages"][:, j])
                    sample_log_probs = ms.tensor(sample["log_probs"][:, j])

                    step_index = [
                        pipeline.scheduler.index_for_timestep(t)
                        for t in timesteps
                    ]
                    prev_step_index = [step + 1 for step in step_index]
                    sigma = pipeline.scheduler.sigmas[step_index].view(
                        -1, 1, 1, 1)
                    sigma_prev = pipeline.scheduler.sigmas[
                        prev_step_index].view(-1, 1, 1, 1)

                    with pipeline.transformer.disable_adapter():
                        _, _, prev_sample_mean_ref, _ = net_with_loss.compute_log_prob(
                            latents, next_latents, timesteps, embeds,
                            pooled_embeds, sigma, sigma_prev)

                    loss, grad = loss_and_grad_fn(
                        latents,
                        next_latents,
                        timesteps,
                        embeds,
                        pooled_embeds,
                        advantages,
                        sample_log_probs,
                        sigma,
                        sigma_prev,
                        prev_sample_mean_ref=prev_sample_mean_ref,
                        loss_scaler=loss_scaler)

                    if (i * num_train_timesteps +
                            j) % args.gradient_accumulation_steps == 0:
                        grad_accumulated = grad
                        logger.debug("Accumuated Gradient is reinitialized.")
                    else:
                        map_(lambda x, y: x.add_(y), grad_accumulated, grad)
                        logger.debug("Accumuated Gradient is updated.")

                    if (i * num_train_timesteps + j +
                            1) % args.gradient_accumulation_steps == 0:
                        syn_gradients(grad_accumulated)
                        clip_by_global_norm(grad_accumulated,
                                            max_norm=args.max_grad_norm)
                        optimizer(grad_accumulated)
                        logger.debug("Parameters are updated.")

                    avg_loss.append(loss.item())
                    global_step += 1

                logger.info({
                    "global_step": global_step,
                    "epoch": epoch,
                    "loss": np.mean(avg_loss).item()
                })

                if args.ema:
                    ema(trainable_parameters, global_step)


def main():
    parser = argparse.ArgumentParser(usage="train sd3 with GRPO")
    args = parser.parse_args()

    # TODO: hard coded, refactor later
    # we follow config general_ocr_sd3 here
    args.num_steps = 10
    args.guidance_scale = 4.5
    args.resolution = 512
    args.timestep_fraction = 1.0  # original 0.99, why?
    args.kl_reward = 0
    args.model = os.environ.get("SD3_PATH", DEFAULT_MODEL)
    args.mixed_precision = "bf16"  # original fp16, but need to add loss scaler
    args.learning_rate = 3e-4
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_weight_decay = 1e-4
    args.adam_epsilon = 1e-8
    args.max_grad_norm = 1.0
    args.reward_fn = {
        "jpeg_compressibility": 1
    }  # we should use {"ocr": 1}, but not implemented yet :(
    args.dataset = "dataset/ocr"
    args.num_epochs = 100  # original 100000
    args.num_inner_epochs = 1
    args.train_batch_size = 3  # original 12, oom
    args.test_batch_size = 3  # original 16
    args.num_batches_per_epoch = 12
    args.num_image_per_prompt = 1  # original 6
    args.gradient_accumulation_steps = args.num_batches_per_epoch // 2
    args.eval_freq = 3  # original 60
    args.eval_num_steps = 40
    args.save_freq = 3  # original 60
    args.cfg = True
    args.beta = 0.001
    args.adv_clip_max = 5
    args.clip_range = 1e-4
    args.run_name = ""
    args.resume_from = ""
    args.seed = 42
    args.use_lora = True
    args.lora_path = None
    args.per_prompt_stat_tracking = True
    args.global_std = True
    args.validation_num = 12
    args.ema = True
    args.debug = False  # True to test with one layer network.

    train(args)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    main()
