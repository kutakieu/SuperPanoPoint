from typing import Dict

import numpy as np
from lightning import pytorch as pl

from superpanopoint.lossfn_optimizer import (loss_function_factory,
                                             optimizer_factory)
from superpanopoint.models import model_factory
from superpanopoint.utils.logger import get_logger

logger = get_logger(__name__)


class LightningWrapper(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.net = model_factory(cfg)
        self.lossfn_pointness = loss_function_factory("binary_cross_entropy")
        # self.lossfn_descriptor = loss_function_factory(cfg.training.lossfn.descriptor.name)
        self.optimizer_name = cfg.training.optimizer.name
        self.learning_rate = cfg.training.optimizer.learning_rate
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _clac_pointness_loss(self, pred, gt):
        return self.lossfn_pointness(pred, gt)

    def _clac_description_loss(self, pred, gt):
        raise NotImplementedError

    # override
    def configure_optimizers(self):
        optimizer = optimizer_factory(self.optimizer_name, self.learning_rate, self.net)
        return optimizer

    # override
    def training_step(self, batch, batch_idx):
        img, points = batch
        pred_pointness, pred_desc = self.net(img)
        print(pred_pointness.shape, points.shape)
        loss_pointness = self._clac_pointness_loss(pred_pointness, points)
        self.log("loss_train_step", loss_pointness, sync_dist=True)
        self.training_step_outputs.append({"loss": loss_pointness.detach().cpu().numpy()})
        return loss_pointness

    # override
    def on_train_epoch_end(self) -> None:
        loss_avg = np.stack([x["loss"] for x in self.training_step_outputs]).mean()
        self.log("loss_train_epoch", loss_avg, sync_dist=True)
        self.training_step_outputs.clear()

    # override
    def validation_step(self, batch, batch_idx):
        img, points = batch
        pred_pointness, pred_desc = self.net(img)
        loss_pointness = self._clac_pointness_loss(pred_pointness, points)
        # metrics = self.calc_metrics()
        self.validation_step_outputs.append({"loss": loss_pointness.detach().cpu().numpy(), "metrics": 0})
        return loss_pointness

    # override
    def on_validation_epoch_end(self) -> None:
        loss_avg = np.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("loss_val", loss_avg, sync_dist=True)

        self.validation_step_outputs.clear()

    # override
    def test_step(self, batch, batch_idx):
        img, points = batch
        pred_pointness, pred_desc = self.net(img)
        loss_pointness = self._clac_pointness_loss(pred_pointness, points)
        # metrics = self.calc_metrics()
        self.test_step_outputs.append({"loss": loss_pointness.detach().cpu().numpy(), "metrics": 0})
        return loss_pointness

    # override
    def on_test_epoch_end(self) -> None:
        loss_avg = np.stack([x["loss"] for x in self.test_step_outputs]).mean()
        self.log("loss_test", loss_avg, sync_dist=True)

        self.test_step_outputs.clear()

    def calc_metrics(self, pred_batch, gt_batch) -> Dict[str, float]:
        raise NotImplementedError
