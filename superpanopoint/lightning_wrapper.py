from typing import Dict

import numpy as np
from lightning import pytorch as pl
from torch import Tensor

from superpanopoint.lossfn_optimizer import (loss_function_factory,
                                             optimizer_factory)
from superpanopoint.models import model_factory
from superpanopoint.utils.logger import get_logger

logger = get_logger(__name__)


class LightningWrapper(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = model_factory(cfg)
        self.lossfn_pointness, self.lossfn_descriptor = loss_function_factory(cfg)
        self.desc_loss_weight = cfg.training.loss.get("desc_weight", 1)
        self.optimizer_name = cfg.training.optimizer.name
        self.learning_rate = cfg.training.optimizer.learning_rate
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _clac_pointness_loss(self, pred: Tensor, gt: Tensor):
        return self.lossfn_pointness(pred, gt)

    def _clac_description_loss(self, desc: Tensor, warped_desc: Tensor, correspondence_mask: Tensor, incorrespondence_mask: Tensor):
        return self.lossfn_descriptor(desc, warped_desc, correspondence_mask, incorrespondence_mask)
    
    def _calc_contrastive_description_loss(self, desc: Tensor, warped_desc: Tensor, points_idxs: np.ndarray, contrastive_pair_idxs: np.ndarray):
        return self.lossfn_descriptor(desc, warped_desc, points_idxs, contrastive_pair_idxs)

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
            img, points, warped_img, warped_points, correspondence_mask, incorrespondence_mask = batch
            pointness, desc = self.net(img)
            warped_pointness, warped_desc = self.net(warped_img)
            loss_warped_pointness = self._clac_pointness_loss(warped_pointness, warped_points)
            loss_desc = self._clac_description_loss(desc, warped_desc, correspondence_mask, incorrespondence_mask)
            self.log("loss_desc_train_step", loss_desc, sync_dist=True)
            loss_pointness = self._clac_pointness_loss(pointness, points) + loss_warped_pointness
            self.log("loss_pointness_train_step", loss_pointness, sync_dist=True)
            loss_total = loss_pointness + self.desc_loss_weight * loss_desc
        elif self.cfg.model.name == "unet":
            img, warped_img, gt_pointness, gt_warped_pointness, points_idxs, contrastive_points_idxs = batch
            pointness, desc = self.net(img)
            warped_pointness, warped_desc = self.net(warped_img)
            loss_pointness = self._clac_pointness_loss(pointness, gt_pointness) + self._clac_pointness_loss(warped_pointness, gt_warped_pointness)
            self.log("loss_pointness_train_step", loss_pointness, sync_dist=True)
            loss_desc = self._calc_contrastive_description_loss(desc, warped_desc, points_idxs, contrastive_points_idxs)
            self.log("loss_desc_train_step", loss_desc, sync_dist=True)
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
        elif self.cfg.model.name == "superpoint":
            img, points, warped_img, warped_points, correspondence_mask, incorrespondence_mask = batch
            pointness, desc = self.net(img)
            warped_pointness, warped_desc = self.net(warped_img)
            loss_warped_pointness = self._clac_pointness_loss(warped_pointness, warped_points)
            loss_desc = self._clac_description_loss(desc, warped_desc, correspondence_mask, incorrespondence_mask)
            loss_pointness = self._clac_pointness_loss(pointness, points) + loss_warped_pointness
            loss_total = loss_pointness + self.desc_loss_weight * loss_desc
        elif self.cfg.model.name == "unet":
            img, warped_img, gt_pointness, gt_warped_pointness, points_idxs, contrastive_points_idxs = batch
            pointness, desc = self.net(img)
            warped_pointness, warped_desc = self.net(warped_img)
            loss_pointness = self._clac_pointness_loss(pointness, gt_pointness) + self._clac_pointness_loss(warped_pointness, gt_warped_pointness)
            loss_desc = self._calc_contrastive_description_loss(desc, warped_desc, points_idxs, contrastive_points_idxs)
            loss_total = loss_pointness + self.desc_loss_weight * loss_desc
        else:
            raise NotImplementedError

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
            loss_pointness = self._clac_pointness_loss(pointness, points) + loss_warped_pointness
        elif self.cfg.model.name == "superpoint":
            img, points, warped_img, warped_points, correspondence_mask, incorrespondence_mask = batch
            pointness, desc = self.net(img)
            warped_pointness, warped_desc = self.net(warped_img)
            loss_warped_pointness = self._clac_pointness_loss(warped_pointness, warped_points)
            loss_desc = self._clac_description_loss(desc, warped_desc, correspondence_mask, incorrespondence_mask)
            loss_pointness = self._clac_pointness_loss(pointness, points) + loss_warped_pointness
        elif self.cfg.model.name == "unet":
            img, warped_img, gt_pointness, gt_warped_pointness, points_idxs, contrastive_points_idxs = batch
            pointness, desc = self.net(img)
            warped_pointness, warped_desc = self.net(warped_img)
            loss_pointness = self._clac_pointness_loss(pointness, gt_pointness) + self._clac_pointness_loss(warped_pointness, gt_warped_pointness)
            loss_desc = self._calc_contrastive_description_loss(desc, warped_desc, points_idxs, contrastive_points_idxs)
        else:
            raise NotImplementedError

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
