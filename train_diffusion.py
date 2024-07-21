import torch
import torch.nn.functional as F
import numpy as np
import random
from omegaconf import OmegaConf
from copy import deepcopy
import torch.utils
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.utils import save_image, make_grid
from torch.distributed import init_process_group, destroy_process_group
from functools import partial
import os

from diffusion import GaussianDiffusion
from model import UnetDiffusion
from dataset import ImageDataset, Resize, RandomVerticalFlip


# Helper functions
# -------------------------------------------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def update_ema(model, model_ema, ema_rate=0.999):
    for p, p_ema in zip(model.parameters(), model_ema.parameters()):
        ema = ema_rate * p.data + (1.0 - ema_rate) * p_ema.data
        p_ema.data = ema


def data_wrapper(loader):
    while True:
        for batch in loader:
            yield batch


def make_minibatches(batch, mini_batch_size):
    batch_size, c, h, w = batch.shape
    assert batch_size % mini_batch_size == 0, "batch_size % mini_batch_size != 0"
    batch = batch.contiguous()
    batch = batch.view(batch_size // mini_batch_size, mini_batch_size, c, h, w)
    for mini_batch in batch:
        yield mini_batch


def lr_schedule(optimizer, step):
    pass


def forward_backward(batch, model, optimizer, diffusion):
    # make a mini_batch from batch
    model.train()
    device = next(model.parameters()).device
    optimizer.zero_grad()
    loss_accum = 0
    for mini_batch in make_minibatches(batch):
        mini_batch = mini_batch.to(device)
        n = mini_batch.shape[0]
        t = diffusion.sample_timesteps(n).to(device)

        x_t, noise = diffusion.q_sample(mini_batch, t)

        # forward pass
        pred = model(x_t, t)
        loss = F.mse_loss(pred, noise)
        loss = loss / grad_accum_steps
        loss_accum += loss.item()

        # backward pass
        loss.backward()
    return loss_accum


# Instantiations and inits
# -------------------------------------------------------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device set: {device}")
seed = 1337
set_seed(seed)

config = OmegaConf.load("./config.yaml")

diffusion = GaussianDiffusion(**config["diffusion"])
model = UnetDiffusion(**config["model"])
model.to(device)

ddp = config["training"]["ddp"]
root = config["training"]["root"]
batch_size = config["training"]["batch_size"]
mini_batch_size = config["training"]["mini_batch_size"]
ema_rate = config["training"]["ema_rate"]
assert batch_size % mini_batch_size == 0
grad_accum_steps = batch_size // mini_batch_size

if ddp:
    sampler = torch.utils.data.Sampler()
else:
    sampler = None
ds_transforms = Compose(
    [
        RandomVerticalFlip(p=0.5),
        Resize(128, 128),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ]
)
ds = ImageDataset(root, ds_transforms)
loader = torch.utils.data.DataLoader(
    ds,
    batch_size,
    shuffle=True,
    drop_last=True,
    sampler=sampler,
)
data_fetcher = data_wrapper(loader)

os.makedirs("logs/", exist_ok=True)
os.makedirs("sample/", exist_ok=True)
log_file = "logs/log.txt"

param_count = 0
for p in model.parameters():
    param_count += p.numel()
with open(log_file, "w") as f:
    print(f"{param_count=:,}", file=f)

optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
make_minibatches = partial(make_minibatches, mini_batch_size=mini_batch_size)
forward_backward = partial(
    forward_backward,
    model=model,
    optimizer=optimizer,
    diffusion=diffusion,
)

# Training loop
# -------------------------------------------------------------------------------------------------
training_config = config["training"]
max_step = training_config["max_step"]
train_loss = []
for step in range(max_step):
    if step == 1000:
        ema_model = deepcopy(model)
    if step > 1000 and step % 1000 == 0:
        update_ema(model, ema_model, ema_rate)

    # every once in a while generate samples
    if step % 200 == 0:
        samples = diffusion.sample(model, 4, img_size=(128, 128))
        samples = make_grid(samples, nrow=2)
        save_image(samples, f"sample/step_{step}.png")

    # get a batch
    batch = next(data_fetcher)
    loss_accum = forward_backward(batch)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr_schedule(optimizer, step)
    optimizer.step()

    # log
    with open(log_file, "a") as f:
        print(f"step {step:5d} | loss: {loss_accum: .6f} | norm: {norm: .5f}", file=f)
    train_loss.append(loss_accum)

    if step % 1000 == 0 and step > 1000:
        checkpoint = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "ema_model": ema_model.state_dict(),
            "step": step,
            "train_loss": train_loss,
        }
        torch.save("checkpoint.pt", checkpoint)


# final save
checkpoint = {
    "model": model.state_dict(),
    "optim": optimizer.state_dict(),
    "ema_model": ema_model.state_dict(),
    "step": step,
    "train_loss": train_loss,
}
torch.save("checkpoint.pt", checkpoint)
