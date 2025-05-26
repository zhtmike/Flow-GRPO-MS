# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/flow_grpo/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Tuple, Union

import mindspore as ms
import mindspore.mint as mint
import mindspore.ops as ops
import numpy as np
from mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerOutput)
from mindone.diffusers.utils.mindspore_utils import randn_tensor


def sde_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: ms.Tensor,
    timestep: Union[float, ms.Tensor],
    sample: ms.Tensor,
    prev_sample: Optional[ms.Tensor] = None,
    generator: Optional[np.random.Generator] = None,
    determistic: bool = False,
    sigma: Optional[ms.Tensor] = None,
    sigma_prev: Optional[ms.Tensor] = None
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`ms.Tensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`ms.Tensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    if sigma is None or sigma_prev is None:
        step_index = [scheduler.index_for_timestep(t) for t in timestep]
        prev_step_index = [step + 1 for step in step_index]
        sigma = scheduler.sigmas[step_index].view(-1, 1, 1, 1)
        sigma_prev = scheduler.sigmas[prev_step_index].view(-1, 1, 1, 1)
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma

    std_dev_t = mint.sqrt(sigma /
                          (1 - mint.where(sigma == 1, sigma_max, sigma))) * 0.7

    # our sde
    prev_sample_mean = sample * (1 + std_dev_t**2 /
                                 (2 * sigma) * dt) + model_output * (
                                     1 + std_dev_t**2 * (1 - sigma) /
                                     (2 * sigma)) * dt

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`.")

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * mint.sqrt(
            -1 * dt) * variance_noise

    # No noise is added during evaluation
    if determistic:
        prev_sample = sample + dt * model_output

    log_prob = (-((ops.stop_gradient(prev_sample) - prev_sample_mean)**2) /
                (2 * ((std_dev_t * mint.sqrt(-1 * dt))**2)) -
                mint.log(std_dev_t * mint.sqrt(-1 * dt)) -
                mint.log(mint.sqrt(2 * ms.tensor(math.pi))))

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob, prev_sample_mean, std_dev_t * mint.sqrt(
        -1 * dt)
