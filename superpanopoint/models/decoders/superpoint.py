import torch
import torch.nn.functional as F
from torch import nn


class SuperPointDecoder(nn.Module):
    def __init__(self, detector: nn.Module, descriptor: nn.Module) -> None:
        super().__init__()
        self.detector = detector
        self.descriptor = descriptor

    def forward(self, x: torch.Tensor):
        pointness = self.detector(x)
        desc = self.descriptor(x)
        return pointness, desc


class PointDetector(nn.Module):
    def __init__(self, in_channels: int=512, out_channels: int=65) -> None:
        super().__init__()
        self.layers: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        bs, ch, h_c, w_c = x.shape
        pointness = self.layers(x)[:, :-1, :, :].view(bs, h_c*8, w_c*8, 1)
        return pointness

class PointDescriptor(nn.Module):
    def __init__(self, in_channels: int=512, out_channels: int=256) -> None:
        super().__init__()
        self.layers: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            )
        
    def forward(self, x: torch.Tensor):
        desc = self.layers(x)
        desc = F.interpolate(desc, scale_factor=8, mode="bicubic", align_corners=False)
        return desc.permute(0, 2, 3, 1)
    

if __name__ == "__main__":
    from superpanopoint.models.encoders.vgg import VGG
    img = torch.rand(1, 3, 512, 1024)
    model = VGG()
    out = model(img)
    print(out.shape)

    detector_decoder = PointDetector()
    detector_decoder_out = detector_decoder(out)
    print(detector_decoder_out.shape)

    descriptor_decoder = PointDescriptor()
    descriptor_decoder_out = descriptor_decoder(out)
    print(descriptor_decoder_out.shape)
