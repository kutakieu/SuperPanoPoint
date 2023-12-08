from typing import Literal

import torch
import torch.optim as optim
from einops import rearrange
from torch import Tensor, nn

LossFunctionType = Literal["cross_entropy", "binary_cross_entropy"]
OptimizerType = Literal["sgd", "adam"]


def loss_function_factory(loss_function_name: LossFunctionType):
    if loss_function_name == "cross_entropy":
        weight = torch.full([65], 4096.0, dtype=torch.float32)
        weight[-1] = 1
        return nn.CrossEntropyLoss(weight=weight)
    elif loss_function_name == "binary_cross_entropy":
        return nn.BCELoss()
    raise NotImplementedError

def optimizer_factory(optimizer_name: OptimizerType, learning_rate: float, net):
    if optimizer_name == "sgd":
        return optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "adam":
        return optim.Adam(net.parameters(), lr=learning_rate)
    raise NotImplementedError

def make_pointness_loss_fn(weight=1000.0):
        weight = torch.full([65], weight, dtype=torch.float32)
        weight[-1] = 1
        return nn.CrossEntropyLoss(weight=weight)

def descriptor_loss_fn(desc: Tensor, warped_desc: Tensor, correspondence_mask: Tensor, _lambda: float=512.0, pos_margin = 1.0, neg_margin = 0.2):
    desc = rearrange(desc, "b c h w -> b (h w) c")
    warped_desc = rearrange(warped_desc, "b c h w -> b (h w) c")
    ele_wise_dot = torch.einsum("bnc,bmc->bnm", desc, warped_desc)
    pos_corres_loss = correspondence_mask * torch.maximum(torch.zeros_like(ele_wise_dot), pos_margin - ele_wise_dot)
    neg_corres_loss = (1 - correspondence_mask) * torch.maximum(torch.zeros_like(ele_wise_dot), ele_wise_dot - neg_margin)
    return torch.mean(_lambda * pos_corres_loss + 0.1 * neg_corres_loss)
