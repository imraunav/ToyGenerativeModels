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
import torch.distributed as dist
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
def update_ema(target_model, source_model, ema_rate=0.999):
    """
    :param target_model: the model that takes up all the average. This is the EMA model
    :param source_model: the model that gives the new averaging values. This is the Raw model
    """
    for p_tgt, p_src in zip(target_model.parameters(), source_model.parameters()):
        ema = ema_rate * p_tgt.data + (1.0 - ema_rate) * p_src.data
        p_tgt.data = ema


def data_wrapper(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


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
    mini_step = 0
    for mini_batch in make_minibatches(batch):
        if ddp:
            model.require_backward_grad_sync = mini_step == grad_accum_steps - 1
        mini_batch = mini_batch.to(device)

        n = mini_batch.shape[0]
        t = diffusion.sample_timesteps(n).to(device)
        x_t, noise = diffusion.q_sample(mini_batch, t)

        # forward pass
        pred = model(x_t, t)
        loss = F.mse_loss(pred, noise)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        # backward pass
        loss.backward()
        mini_step += 1
    return loss_accum


def syncronize_params(m):
    """
    Broadcast to syncronize the module parameters across all processes.
    Sanity check.
    """
    for p in m.parameters():
        dist.broadcast(p.data, 0)


# Instantiations and inits
# -------------------------------------------------------------------------------------------------
config = OmegaConf.load("./config.yaml")

# Karapthy : https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

seed = config.get("seed", 1337)
print(f"[{device}] Seed set: {seed}")
set_seed(seed)
torch.set_float32_matmul_precision("high")

# model configs -----------------------------------------------------------------------------------
diffusion = GaussianDiffusion(**config["diffusion"])
model = UnetDiffusion(**config["model"])
model.to(device)
if ddp:
    syncronize_params(model)  # sanity check
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

# dataloader configs ------------------------------------------------------------------------------
root = config["training"]["root"]
batch_size = config["training"]["batch_size"]
if ddp:
    assert batch_size % ddp_world_size == 0
    batch_size = batch_size // ddp_world_size
mini_batch_size = config["training"]["mini_batch_size"]
ema_rate = config["training"]["ema_rate"]
assert batch_size % mini_batch_size == 0
grad_accum_steps = batch_size // mini_batch_size

ds_transforms = Compose(
    [
        RandomVerticalFlip(p=0.5),
        Resize(128, 128),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ]
)
ds = ImageDataset(root, ds_transforms)
# ddp requires specialised sampling
sampler = None
if ddp:
    sampler = DistributedSampler(ds, shuffle=True, seed=seed, drop_last=True)
loader = torch.utils.data.DataLoader(
    ds,
    batch_size,
    shuffle=not ddp,
    drop_last=True,
    sampler=sampler,
)
# generator that takes care of the fetching data per iterations
data_fetcher = data_wrapper(loader, sampler)

# logging configs ---------------------------------------------------------------------------------
os.makedirs("logs/", exist_ok=True)
os.makedirs("sample/", exist_ok=True)
log_file = "logs/log.txt"

# start of logging --------------------------------------------------------------------------------
if master_process:
    param_count = 0
    for p in model.parameters():
        param_count += p.numel()
    with open(log_file, "w") as f:
        print(f"{param_count=:,}", file=f)

optimizer = torch.optim.AdamW(raw_model.parameters(), **config["optimizer"])
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
    # check if this is the last step
    last_step = step == (max_step - 1)
    # start of exponential avg
    if step == 1000 and master_process:
        ema_model = deepcopy(raw_model)
        ema_model.requires_grad_(False)  # safety check
    if step > 1000 and master_process:
        update_ema(ema_model, raw_model, ema_rate)

    # every once in a while generate samples
    if step % 200 == 0 and master_process:
        samples = diffusion.sample(raw_model, 4, img_size=(128, 128))
        samples = make_grid(samples, nrow=2)
        save_image(samples, f"sample/step_{step}.png")

    # get a batch
    batch = next(data_fetcher)
    loss_accum = forward_backward(batch)
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr_schedule(optimizer, step)
    optimizer.step()
    # wait for the GPU to finish work, can cause weird behaviors at times if not done.
    if device_type == "cuda":
        torch.cuda.synchronize()

    # log
    if master_process:
        with open(log_file, "a") as f:
            print(
                f"step: {step:6d}/{max_step:6d} | loss: {loss_accum.item():.6f} | norm: {norm:.5f}",
                file=f,
            )
        train_loss.append(loss_accum)

        if last_step or (step % 1000 == 0 and step > 1000):
            checkpoint = {
                "model": raw_model.state_dict(),
                "optim": optimizer.state_dict(),
                "ema_model": ema_model.state_dict(),
                "step": step,
                "train_loss": train_loss,
            }
            torch.save(checkpoint, "checkpoint.pt")

if ddp:
    destroy_process_group()
