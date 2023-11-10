from typing import Dict, List, Union, cast

import torch
from torch import nn


class VGG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = make_layers(cfgs["D"])

    def forward(self, x):
        return self.layers(x)

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, in_channels: int = 3) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "B": [64, 64, 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "D": [64, 64, 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],
    "E": [64, 64, 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512],
}

if __name__ == "__main__":
    img = torch.rand(1, 3, 512, 512)
    model = VGG()
    out = model(img)
    print(out.shape)
