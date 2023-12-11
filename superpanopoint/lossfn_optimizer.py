from typing import Literal

import numpy as np
import torch
import torch.optim as optim
from einops import rearrange
from torch import Tensor, nn

LossFunctionType = Literal["homographic", "contrastive"]
OptimizerType = Literal["sgd", "adam"]


def loss_function_factory(cfg):
    if cfg.dataset.type == "homographic":
        return make_pointness_loss_fn(cfg.training.loss.get("pointness_positive_weight", 1000.0)), \
                make_descriptor_loss_fn(
                    cfg.training.loss.get("desc_positive_weight", 1024.0), 
                    cfg.training.loss.get("desc_positive_margin", 1.0), 
                    cfg.training.loss.get("desc_negative_margin", 0.2)
                )
    elif cfg.dataset.type == "contrastive":
        return nn.CrossEntropyLoss(weight=torch.Tensor([1, cfg.training.loss.get("pointness_positive_weight", 1.0)])), \
                make_contrastive_descriptor_loss_fn()
    raise NotImplementedError

def optimizer_factory(optimizer_name: OptimizerType, learning_rate: float, net):
    if optimizer_name == "sgd":
        return optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "adam":
        return optim.Adam(net.parameters(), lr=learning_rate)
    raise NotImplementedError

def make_pointness_loss_fn(size=65, weight=1000.0):
        weight = torch.full([size], weight, dtype=torch.float32)
        weight[-1] = 1
        return nn.CrossEntropyLoss(weight=weight)

def make_descriptor_loss_fn(_lambda: float=1024.0, pos_margin = 1.0, neg_margin = 0.2):
    def descriptor_loss_fn(desc: Tensor, warped_desc: Tensor, correspondence_mask: Tensor, incorrespondence_mask: Tensor):
        desc = rearrange(desc, "b c h w -> b (h w) c")
        warped_desc = rearrange(warped_desc, "b c h w -> b (h w) c")
        ele_wise_dot = torch.einsum("bnc,bmc->bnm", desc, warped_desc)
        pos_corres_loss = correspondence_mask * torch.maximum(torch.zeros_like(ele_wise_dot), pos_margin - ele_wise_dot)
        neg_corres_loss = incorrespondence_mask * torch.maximum(torch.zeros_like(ele_wise_dot), ele_wise_dot - neg_margin)
        return torch.mean(_lambda * pos_corres_loss + neg_corres_loss)
    return descriptor_loss_fn
