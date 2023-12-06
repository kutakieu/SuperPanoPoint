import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from superpanopoint.models.utils.postprocess import non_maximum_suppression


def postprocess_pointness(pointness: torch.Tensor, apply_nms=True) -> np.ndarray:
    """
    rearrange pointness tensor to numpy array

    Args:
        - pointness shape: (bs, 65, h_c, w_c).  
            - 65 is 64 + 1 (background)
            - h_c and w_c are height and width of the output of the network
            h_c = h / 8, w_c = w / 8 according to the paper

    Returns:
        - pointness shape: (bs, h, w)
            - h and w are height and width of the original input image
    """
    pointness = pointness.detach().cpu()
    max_idx = torch.argmax(pointness, dim=1, keepdim=True)
    point_mask = torch.FloatTensor(pointness.shape)
    point_mask.zero_().scatter_(1, max_idx, 1)
    point_mask = point_mask[:, :-1, :, :].detach().cpu().numpy()  # (bs, 64, h_c, w_c)
    point_mask = rearrange(point_mask, "b (ch1 ch2) h w -> b h w ch1 ch2", ch1=8, ch2=8)  # (bs, h_c, w_c, 8, 8)
    point_mask = rearrange(point_mask, "b h w c1 c2 -> b (h c1) (w c2)")  # (bs, h_c*8, w_c*8)

    if apply_nms:
        pointness = pointness[:, :-1, :, :].detach().cpu().numpy()  # (bs, 64, h_c, w_c)
        pointness = rearrange(pointness, "b (ch1 ch2) h w -> b h w ch1 ch2", ch1=8, ch2=8)  # (bs, h_c, w_c, 8, 8)
        pointness = rearrange(pointness, "b h w c1 c2 -> b (h c1) (w c2)")  # (bs, h_c*8, w_c*8)
        pointness = pointness * point_mask  # (bs, h, w)
        for i in range(pointness.shape[0]):
            point_mask[i] = non_maximum_suppression(pointness[i])

    return point_mask  # (bs, h, w)

def postprocess_descriptor(desc: torch.Tensor, img_w: int, img_h: int) -> np.ndarray:
    desc = F.interpolate(desc, size=(img_h, img_w), mode="bicubic", align_corners=False)
    return desc.permute(0, 2, 3, 1).detach().cpu().numpy()  # (bs, h, w, ch)


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
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)

class PointDescriptor(nn.Module):
    def __init__(self, in_channels: int=512, out_channels: int=256) -> None:
        super().__init__()
        self.layers: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            )
        
    def forward(self, x: torch.Tensor):
        desc = self.layers(x)
        return desc  # (bs, ch, h, w)
    

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

    dummy_pointness_label = torch.ones(1, 512, 1024, 1)

    loss_fn = nn.BCELoss()
    loss = loss_fn(detector_decoder_out, dummy_pointness_label)
    print(loss)
