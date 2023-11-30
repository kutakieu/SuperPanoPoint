from typing import Dict

import numpy as np
from lightning import pytorch as pl
from torch import Tensor

from superpanopoint.lossfn_optimizer import (descriptor_loss_fn,
                                             make_pointness_loss_fn,
                                             optimizer_factory)
from superpanopoint.models import model_factory
from superpanopoint.utils.logger import get_logger

logger = get_logger(__name__)


class LightningWrapper(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = model_factory(cfg)
        self.lossfn_pointness = make_pointness_loss_fn(cfg.training.get("pointness_loss_weight", 1000.0))
        self.lossfn_descriptor = descriptor_loss_fn
        self.desc_loss_weight = cfg.training.get("desc_loss_weight", 250.0)
        self.optimizer_name = cfg.training.optimizer.name
        self.learning_rate = cfg.training.optimizer.learning_rate
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _clac_pointness_loss(self, pred: Tensor, gt: Tensor):
        return self.lossfn_pointness(pred, gt)

    def _clac_description_loss(self, desc: Tensor, warped_desc: Tensor, correspondence_mask: Tensor):
        return self.lossfn_descriptor(desc, warped_desc, correspondence_mask)

    # override
    def configure_optimizers(self):
        optimizer = optimizer_factory(self.optimizer_name, self.learning_rate, self.net)
        return optimizer

    # override
    def training_step(self, batch, batch_idx):
        if self.cfg.model.name == "magicpoint":
            img, points = batch
            pointness = self.net(img)
            loss_total = self._clac_pointness_loss(pointness, points)
            self.log("loss_pointness_train_step", loss_total, sync_dist=True)
        elif self.cfg.model.name == "superpoint":
            img, points, warped_img, warped_points, correspondence_mask = batch
            pointness, desc = self.net(img)
            warped_pointness, warped_desc = self.net(warped_img)
            loss_warped_pointness = self._clac_pointness_loss(warped_pointness, warped_points)
            loss_desc = self._clac_description_loss(desc, warped_desc, correspondence_mask)
            self.log("loss_desc_train_step", loss_desc, sync_dist=True)
            loss_pointness = self._clac_pointness_loss(pointness, points) + loss_warped_pointness
            self.log("loss_pointness_train_step", loss_pointness, sync_dist=True)
            loss_total = loss_pointness + self.desc_loss_weight * loss_desc
        else:
            raise NotImplementedError
        
        self.training_step_outputs.append({"loss": loss_total.detach().cpu().numpy()})
        return loss_total

    # override
    def on_train_epoch_end(self) -> None:
        loss_avg = np.stack([x["loss"] for x in self.training_step_outputs]).mean()
        self.log("loss_train_epoch", loss_avg, sync_dist=True)
        self.training_step_outputs.clear()

    # override
    def validation_step(self, batch, batch_idx):
        if self.cfg.model.name == "magicpoint":
            img, points = batch
            pointness = self.net(img)
            loss_total = self._clac_pointness_loss(pointness, points)
        else:
            img, points, warped_img, warped_points, correspondence_mask = batch
            pointness, desc = self.net(img)
            warped_pointness, warped_desc = self.net(warped_img)
            loss_warped_pointness = self._clac_pointness_loss(warped_pointness, warped_points)
            loss_desc = self._clac_description_loss(desc, warped_desc, correspondence_mask)
            loss_pointness = self._clac_pointness_loss(pointness, points) + loss_warped_pointness
            loss_total = loss_pointness + self.desc_loss_weight * loss_desc

        # metrics = self.calc_metrics()
        self.validation_step_outputs.append({"loss": loss_total.detach().cpu().numpy(), "metrics": 0})
        return loss_total

    # override
    def on_validation_epoch_end(self) -> None:
        loss_avg = np.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("loss_val", loss_avg, sync_dist=True)

        self.validation_step_outputs.clear()

    # override
    def test_step(self, batch, batch_idx):
        if self.cfg.model.name == "magicpoint":
            img, points = batch
            pointness = self.net(img)
            loss_desc = loss_warped_pointness = 0.0
        else:
            img, points, warped_img, warped_points, correspondence_mask = batch
            pointness, desc = self.net(img)
            warped_pointness, warped_desc = self.net(warped_img)
            loss_warped_pointness = self._clac_pointness_loss(warped_pointness, warped_points)
            loss_desc = self._clac_description_loss(desc, warped_desc, correspondence_mask)
        loss_pointness = self._clac_pointness_loss(pointness, points) + loss_warped_pointness

        # metrics = self.calc_metrics()
        loss_total = loss_pointness + self.desc_loss_weight * loss_desc
        self.test_step_outputs.append({"loss": loss_total.detach().cpu().numpy(), "metrics": 0})
        return loss_pointness

    # override
    def on_test_epoch_end(self) -> None:
        loss_avg = np.stack([x["loss"] for x in self.test_step_outputs]).mean()
        self.log("loss_test", loss_avg, sync_dist=True)

        self.test_step_outputs.clear()

    def calc_metrics(self, pred_batch, gt_batch) -> Dict[str, float]:
        raise NotImplementedError
