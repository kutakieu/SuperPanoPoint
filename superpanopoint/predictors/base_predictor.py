from abc import abstractmethod
from pathlib import Path
from typing import Union

import torch
from omegaconf import DictConfig, OmegaConf

from superpanopoint.lightning_wrapper import LightningWrapper
from superpanopoint.models import model_factory


class BasePredictor:
    def __init__(self, cfg: Union[str, DictConfig], weight_file: Union[str, Path], device="cpu") -> None:
        if isinstance(cfg, str):
            cfg = OmegaConf.load(cfg)
        self.device = device
        self.net = model_factory(cfg)
        self.net.to(self.device)
        if Path(weight_file).suffix == ".ckpt":
            checkpoint = torch.load(weight_file)
            lightning_model = LightningWrapper(cfg)
            lightning_model.load_state_dict(checkpoint['state_dict'])
            self.net = lightning_model.net
        else:
            self.net.load_state_dict(torch.load(weight_file)["state_dict"])
        self.net.eval()
