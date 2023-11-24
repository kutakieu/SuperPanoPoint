import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from superpanopoint.lightning_wrapper import LightningWrapper
from superpanopoint.models import model_factory

config_path = "config/config_magicpoint.yaml"
checkpoint_path = 'lightning_logs/version_8/checkpoints/epoch=9-step=6820.ckpt'
model_path = 'model.pth'

cfg = OmegaConf.load(config_path)

"""load checkpoint"""
checkpoint = torch.load(checkpoint_path)
print(list(checkpoint.keys()))
lightning_model = LightningWrapper(cfg)
lightning_model.load_state_dict(checkpoint['state_dict'])

"""convert checkpoint to model"""
torch.save({"state_dict": lightning_model.net.state_dict()}, 'model.pth')

"""load model"""
# model = model_factory(cfg)
# weight = torch.load(model_path)
# model.load_state_dict(weight["state_dict"])


from superpanopoint.models.detector import PointDetector

detector = PointDetector(cfg, model_path)

img = np.array(Image.open('data/synthetic/perspective/imgs/0.png'))
points = detector(img, return_as_array=True)
print(points)
for p in points:
    cv2.circle(img, (p[1], p[0]), 2, (0, 0, 255), -1)
cv2.imwrite('result.png', img)
