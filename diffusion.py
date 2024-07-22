import torch
from torch import nn
import numpy as np
import math
from tqdm import tqdm

from model import UnetDiffusion


# function definitions
# -------------------------------------------------------------------------------------------------
def cosinebetas(steps: int):
    f = lambda t: math.cos((t / steps + 0.008) / 1.008 * math.pi * 0.5) ** 2
    f_0 = f(0)
    alpha_cum = lambda t: f(t) / f_0
    beta = [
        min(1 - (alpha_cum(t) / alpha_cum(t - 1)), 0.999)
        for t in np.linspace(1, 1000, steps)
    ]
    return torch.tensor(beta, dtype=torch.float32)


def _totensor(x, broadcast_shape, device_tensor=None, device=None):
    assert device_tensor is not None or device is not None
    if device is None:
        device = device_tensor.device
    while len(x.shape) < len(broadcast_shape):
        x = x[..., None]
    return x.to(device)


# Diffusion class
# -------------------------------------------------------------------------------------------------
class GaussianDiffusion:
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self.betas = cosinebetas(num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def q_sample(self, x_start: torch.tensor, t: torch.tensor):
        """
        Sample for the analytic expression using reparameterisation from the forward process
        :param x_start: x_{0} clean datapoint
        :param t: t timestep
        """
        noise = torch.randn_like(x_start)
        assert x_start.device == t.device, f"{x_start.device} != {t.device}"
        alpha_cumprod_t = _totensor(self.alphas_cumprod[t.cpu()], x_start.shape, t)
        mean = alpha_cumprod_t.sqrt() * x_start
        std = (1.0 - alpha_cumprod_t).sqrt()

        return mean + std * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(0, self.num_timesteps, (n,))

    def p_sample(self, x_t, t, pred, noise):
        """
        p(x_{t-1}|x_{t}, t)

        Sample x_{t-1} given x_{t} and t

        :param x_t: sample at t step
        :param t: time step
        """
        alpha_t = _totensor(self.alphas[t.cpu()], x_t.shape, x_t)
        beta_t = 1 - alpha_t
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(
            _totensor(1.0 - self.alphas_cumprod[t.cpu()], x_t.shape, x_t)
        )
        return (1 / alpha_t.sqrt()) * (
            x_t - pred * beta_t / sqrt_one_minus_alpha_cumprod_t
        ) + beta_t.sqrt() * noise

    @torch.no_grad()
    def sample(self, model, n=1, img_size=(64, 64), x_t=None):
        model.eval()
        device = next(model.parameters()).device
        h, w = img_size
        if x_t is None:
            x_t = torch.randn(n, model.out_ch, h, w, device=device)
        else:
            x_t = x_t.to(device)
        for t in reversed(range(self.num_timesteps)):
            if t > 1:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)
            t = torch.full((n,), t, device=device)
            pred = model(x_t, t)
            x_t = self.p_sample(x_t, t, pred, z)
        x_t = x_t.clip(-1, 1)
        return x_t
