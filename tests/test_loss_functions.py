import torch
from omegaconf import OmegaConf

from superpanopoint.datasets.dataset_factory import DatasetFactory
from superpanopoint.lossfn_optimizer import (descriptor_loss_fn,
                                             make_pointness_loss_fn)
from superpanopoint.models import model_factory

cfg = OmegaConf.load("tests/config/config_homographic.yaml")
dataset_factory = DatasetFactory(cfg)
train_dataloader = dataset_factory.create_dataset("train").create_dataloader()

model = model_factory(cfg)

def test_detector_loss():
    for batch in train_dataloader:
        img, points, warped_img, warped_points, correspondence_mask = batch

        pred_pointness, pred_desc = model(img)
        pred_warped_pointness, pred_warped_desc = model(warped_img)
        lossfn = make_pointness_loss_fn()
        loss = lossfn(pred_pointness, points)
        assert loss.shape == torch.Size([])

def test_descriptor_loss():
    for batch in train_dataloader:
        img, points, warped_img, warped_points, correspondence_mask = batch

        pred_pointness, pred_desc = model(img)
        pred_warped_pointness, pred_warped_desc = model(warped_img)
        loss = descriptor_loss_fn(pred_desc, pred_warped_desc, correspondence_mask)
        assert loss.shape == torch.Size([])

