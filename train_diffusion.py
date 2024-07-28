import os
import torch
import numpy as np
import random
import torch.distributed as dist
from copy import deepcopy
from datetime import datetime
from functools import partial
from omegaconf import OmegaConf
from time import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.utils import save_image, make_grid

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


def print0(*args, **kwargs):
    """
    Convinient code to print only on master process
    """
    if master_process:
        print(*args, **kwargs)


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


def forward_backward(batch, model, diffusion):
    # make a mini_batch from batch
    model.train()
    loss_accum = 0
    mini_step = 0

    for mini_batch in make_minibatches(batch):
        if ddp:
            # sync grads when on last mini batch
            model.require_backward_grad_sync = (
                mini_step == grad_accum_steps - 1
            )  # Karpaty Trick
            mini_step += 1

        # forward pass
        with autocast:
            loss = diffusion.training_loss(model, mini_batch)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()

        # backward pass
        # loss.backward()
        scaler.scale(loss).backward()
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
    dist.init_process_group(backend="nccl")
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
use_float16 = config.get("use_float16", False)
print(f"[{device}] Seed set: {seed}")
set_seed(seed)
torch.set_float32_matmul_precision("high")

# model configs -----------------------------------------------------------------------------------
diffusion = GaussianDiffusion(**config["diffusion"])
model = UnetDiffusion(**config["model"])
# if use_float16:
#     model = model.half()
model.to(device)
torch.compile(model)
if ddp:
    syncronize_params(model)  # sanity check
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

# dataloader configs ------------------------------------------------------------------------------
root = config["training"]["root"]
batch_size = config["training"]["batch_size"]
img_size = config.get("img_size", 64)
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
        Resize(img_size, img_size),
        Lambda(lambda x: x / 127.5 - 1.0),  # change range to [0, 255] -> [-1, 1]
        Lambda(lambda x: x.astype(np.float32)),  # still np.array
        ToTensor(),
        Lambda(lambda x: x.half() if use_float16 else x),
    ]
)
ds = ImageDataset(root, ds_transforms)
# ddp requires specialised sampling
sampler = None
if ddp:
    sampler = DistributedSampler(ds, shuffle=True, seed=seed, drop_last=True)
loader = DataLoader(
    ds,
    batch_size,
    shuffle=sampler is None,
    drop_last=True,
    sampler=sampler,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2,
)
# generator that takes care of the fetching data per iterations
data_fetcher = data_wrapper(loader, sampler)

# logging configs ---------------------------------------------------------------------------------
os.makedirs("logs/", exist_ok=True)
os.makedirs("sample/", exist_ok=True)
log_file = "logs/log.txt"
with open(log_file, "w") as f:
    print0(f"Logging starts at: {datetime.now()}", file=f)

# start of logging --------------------------------------------------------------------------------
if master_process:
    param_count = 0
    for p in model.parameters():
        param_count += p.numel()
    # with open(log_file, "w") as f:
    print0(f"{param_count=:,}")

optimizer = torch.optim.AdamW(raw_model.parameters(), **config["optimizer"])
scaler = torch.cuda.amp.GradScaler(enabled=use_float16)
autocast = torch.autocast(device_type, torch.float16, enabled=use_float16)

make_minibatches = partial(make_minibatches, mini_batch_size=mini_batch_size)
forward_backward = partial(
    forward_backward,
    model=model,
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
    if step > 0 and step % 250 == 0 and master_process:
        with autocast:
            samples = diffusion.sample(raw_model, 4, img_size=(img_size, img_size))
        samples = make_grid(samples, nrow=2)
        save_image(samples, f"sample/step_{step}.png")

    # get a batch
    tic = time()
    batch = next(data_fetcher)
    optimizer.zero_grad()
    loss_accum = forward_backward(batch)
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr_schedule(optimizer, step)
    # optimizer.step()
    scaler.step(optimizer)
    scaler.update()

    # wait for the GPU to finish work, can cause weird behaviors at times if not done for ddp.
    if device_type == "cuda":
        torch.cuda.synchronize()
    toc = time()
    # log
    print0(
        f"step: {step:6d}/{max_step:6d} | loss: {loss_accum.item():.6f} | norm: {norm:.5f} | time: {(toc - tic):.3f} secs",
    )
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
                "scaler": scaler.state_dict(),
                "ema_model": ema_model.state_dict(),
                "step": step,
                "train_loss": train_loss,
            }
            torch.save(checkpoint, "checkpoint.pt")

if ddp:
    dist.destroy_process_group()
