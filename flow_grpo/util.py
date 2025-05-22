import mindspore.nn as nn


def requires_grad_(cell: nn.Cell, requires_grad: bool = True) -> None:
    for p in cell.get_parameters():
        p.requires_grad = requires_grad
