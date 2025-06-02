from functools import partial
from typing import Tuple

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.distributed as dist
import mindspore.nn as nn

map_ = lambda *args, **kwargs: tuple(map(*args, **kwargs))


def requires_grad_(cell: nn.Cell, requires_grad: bool = True) -> None:
    """
    Set the requires_grad attribute for all parameters in a cell.
    """
    for p in cell.get_parameters():
        p.requires_grad = requires_grad


def gather(x: ms.Tensor):
    """
    Gather the tensor across all devices in the distributed environment.
    """
    size = dist.get_world_size()
    if size == 1:
        return x

    output = mint.zeros((x.shape[0] * size, *x.shape[1:]), dtype=x.dtype)
    dist.all_gather_into_tensor(output, x)
    return output


def syn_gradients(gradients: Tuple[ms.Tensor]) -> ms.Tensor:
    """
    Synchronize gradients across all devices.
    """
    size = dist.get_world_size()
    if size == 1:
        return gradients

    map_(dist.all_reduce, gradients)
    map_(lambda x: x / size, gradients)
    return gradients


def clip_by_global_norm(grads: Tuple[ms.Tensor],
                        max_norm: float,
                        norm_type: float = 2.0) -> ms.Tensor:
    """
    Clips the gradients by global norm.
    """
    total_norm = _get_total_norm(grads, norm_type)
    _clip_grads_with_norm_(grads, max_norm, total_norm)
    return total_norm


def _get_total_norm(tensors: Tuple[ms.Tensor],
                    norm_type: float = 2.0) -> ms.Tensor:
    norms = map_(partial(mint.linalg.vector_norm, ord=norm_type), tensors)
    total_norm = mint.linalg.vector_norm(mint.stack(norms), norm_type)
    return total_norm


def _clip_grads_with_norm_(grads: Tuple[ms.Tensor], max_norm: float,
                           total_norm: ms.Tensor) -> None:
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = mint.clamp(clip_coef, max=1.0)
    map_(lambda g: g.mul_(clip_coef_clamped), grads)
