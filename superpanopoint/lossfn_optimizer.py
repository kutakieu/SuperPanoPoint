from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
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
        # return nn.CrossEntropyLoss(weight=torch.Tensor([1, cfg.training.loss.get("pointness_positive_weight", 1.0)])), \
        return make_pointness_map_loss_fn(), \
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

def make_pointness_map_loss_fn():
    lossfn = F.binary_cross_entropy_with_logits
    def pointness_map_loss_fn(pointness: Tensor, target_pointness: Tensor):
        return lossfn(pointness, target_pointness)
    return pointness_map_loss_fn

def make_descriptor_loss_fn(_lambda: float=1024.0, pos_margin = 1.0, neg_margin = 0.2):
    def descriptor_loss_fn(desc: Tensor, warped_desc: Tensor, correspondence_mask: Tensor, incorrespondence_mask: Tensor):
        desc = rearrange(desc, "b c h w -> b (h w) c")
        warped_desc = rearrange(warped_desc, "b c h w -> b (h w) c")
        ele_wise_dot = torch.einsum("bnc,bmc->bnm", desc, warped_desc)
        pos_corres_loss = correspondence_mask * torch.maximum(torch.zeros_like(ele_wise_dot), pos_margin - ele_wise_dot)
        neg_corres_loss = incorrespondence_mask * torch.maximum(torch.zeros_like(ele_wise_dot), ele_wise_dot - neg_margin)
        return torch.mean(_lambda * pos_corres_loss + neg_corres_loss)
    return descriptor_loss_fn

def make_contrastive_descriptor_loss_fn():
    lossfn = nn.CrossEntropyLoss()
    def contrastive_descriptor_loss_fn(desc: Tensor, warped_desc: Tensor, points_idxs: np.ndarray, contrastive_pair_idxs: np.ndarray):
        b, c, h, w = desc.shape
        desc = rearrange(desc, "b c h w -> (b h w) c")
        warped_desc = rearrange(warped_desc, "b c h w -> (b h w) c")

        b, np, nc = contrastive_pair_idxs.shape  # np: number of points, nc: number of contrastive pairs
        for i in range(b):
            points_idxs[i] = points_idxs[i] + i * h * w
        points_idxs = rearrange(points_idxs, "b np -> (b np)")
        desc = rearrange(desc[points_idxs], "(b np) c -> b np c", b=b, np=np)

        contrastive_pair_idxs = rearrange(contrastive_pair_idxs, "b np nc -> b (np nc)")
        for i in range(b):
            contrastive_pair_idxs[i] = contrastive_pair_idxs[i] + i * h * w
        contrastive_pair_idxs = rearrange(contrastive_pair_idxs, "b (np nc) -> (b np nc)", b=b, np=np, nc=nc)
        warped_desc = rearrange(warped_desc[contrastive_pair_idxs], "(b np nc) c -> b np nc c", b=b, np=np, nc=nc)

        ele_wise_dot = torch.einsum("bnc,bnmc->bmn", desc, warped_desc)  # (b, nc, np)
        target = torch.zeros(b, np, dtype=torch.int64, device=desc.device)
        return lossfn(ele_wise_dot, target)
    return contrastive_descriptor_loss_fn

def make_probmap_descriptor_loss_fn():
    lossfn = F.binary_cross_entropy_with_logits
    softmax = nn.Softmax()
    def probmap_descriptor_loss_fn(desc: Tensor, warped_desc: Tensor, similarity_maps):
        loss = 0
        bs = desc.shape[0]
        for i in range(bs):
            cur_desc = desc[i]
            cur_warped_desc = warped_desc[i]
            cur_loss = 0
            for sim_map in similarity_maps:
                (x, y) = sim_map["point"]
                target_probmap = sim_map["similarity_map"]
                desc_vec = cur_desc[:, y, x]
                prob_map = softmax(torch.einsum("c,mnc->mn", desc_vec, cur_warped_desc))
                cur_loss += lossfn(prob_map, target_probmap)
            loss += cur_loss / len(similarity_maps)
        return loss / bs
    return probmap_descriptor_loss_fn
