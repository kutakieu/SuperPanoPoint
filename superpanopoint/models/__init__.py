from torch import nn

from .decoders.superpoint import (PointDescriptor, PointDetector,
                                  SuperPointDecoder)
from .encoders.vgg import VGG


def model_factory(cfg):
    encoder = _encoder_factory(cfg.model.encoder)
    decoder = _decoder_factory(cfg.model.decoder, cfg.model.description_dim)
    return nn.Sequential(encoder, decoder)


def _encoder_factory(encoder_name: str):
    if encoder_name == "vgg":
        return VGG()
    else:
        raise NotImplementedError

    
def _decoder_factory(decoder_name: str, description_dim: int):
    if decoder_name == "superpoint":
        return SuperPointDecoder(
            detector=PointDetector(in_channels=512, out_channels=65),
            descriptor=PointDescriptor(in_channels=512, out_channels=description_dim),
        )
    else:
        raise NotImplementedError
