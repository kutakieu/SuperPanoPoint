from typing import Optional

from omegaconf import DictConfig
from torch import nn

from .decoders.superpoint import (PointDescriptor, PointDetector,
                                  SuperPointDecoder)
from .encoders.vgg import VGG


def model_factory(cfg):
    if cfg.model.name in ["magicpoint", "superpoint"]:
        encoder = _encoder_factory(cfg.model.encoder)
        decoder = _decoder_factory(cfg.model.decoder)
    else:
        raise NotImplementedError
    return nn.Sequential(encoder, decoder)

def _encoder_factory(encoder_cfg: DictConfig):
    if encoder_cfg.name == "vgg":
        return VGG(num_color_chs=encoder_cfg.get("color_channels", 3))
    else:
        raise NotImplementedError

    
def _decoder_factory(decoder_cfg: DictConfig):
    if decoder_cfg.name == "superpoint":
        return SuperPointDecoder(
            detector=PointDetector(in_channels=512, out_channels=65),
            descriptor=PointDescriptor(in_channels=512, out_channels=decoder_cfg.get("description_dim", 256)),
        )
    elif decoder_cfg.name == "magicpoint":
        return PointDetector(in_channels=512, out_channels=65)
    else:
        raise NotImplementedError
