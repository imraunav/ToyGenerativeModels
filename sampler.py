import torch
from omegaconf import OmegaConf
import cv2
import numpy as np

from diffusion import GaussianDiffusion
from model import UnetDiffusion

config = OmegaConf.load("./config.yaml")
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

diffusion = GaussianDiffusion(**config["diffusion"])
model = UnetDiffusion(**config["model"])
model.to(device)

checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["ema_model"])
img_size = config.get("img_size", 64)


samples = diffusion.sample(model, 1, img_size=(img_size, img_size), save_every=False)
print(f"{samples.min()=}, {samples.max()=}, {samples.mean()=}")
samples = (samples + 1.0) * 0.5 * 255.0

samples = samples.permute(0, 2, 3, 1)
samples = samples.numpy(force=True)
print(f"{samples.shape=}")

samples = np.round(samples)
samples = samples.astype(np.uint8)
samples = samples[0]
# samples = np.moveaxis(samples, [0, 1 ,2], [2, 1, ],)
print(f"{samples.shape=}")
cv2.imwrite("sample.png", samples)