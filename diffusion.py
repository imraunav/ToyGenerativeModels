import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
from torchvision.utils import save_image

from model import UnetDiffusion


# function definitions
# -------------------------------------------------------------------------------------------------
def cosinebetas(steps: int):
    """
    generate betas for Cosine noise schedule
    """
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
        alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        # for posterior calculation
        self.alphas_cumprod_prev = torch.empty_like(self.alphas_cumprod)
        self.alphas_cumprod_prev[1:] = self.alphas_cumprod[:-1]
        self.alphas_cumprod_prev[0] = 1.0

        self.alphas_cumprod_next = torch.empty_like(self.alphas_cumprod)
        self.alphas_cumprod_next[:-1] = self.alphas_cumprod[1:]
        self.alphas_cumprod_next[-1] = 0.0

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start: torch.tensor, t: torch.tensor):
        """
        Sample for the analytic expression using reparameterisation from the forward process
        :param x_start: x_{0} clean datapoint
        :param t: t timestep
        """
        noise = torch.randn_like(x_start)
        assert x_start.device == t.device, f"{x_start.device} != {t.device}"
        sqrt_alpha_cumprod_t = _totensor(
            self.sqrt_alphas_cumprod[t.cpu()], x_start.shape, t
        )
        mean = sqrt_alpha_cumprod_t * x_start
        std = _totensor(self.sqrt_one_minus_alphas_cumprod[t.cpu()], x_start.shape, t)

        return mean + std * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(0, self.num_timesteps, (n,))

    def p_sample(self, x_t, t, model_pred, pertubation):
        """
        p(x_{t-1}|x_{t}, t)

        Sample x_{t-1} given x_{t} and t

        :param x_t: sample at t step
        :param t: time step
        :param model_pred: noise predicted from model
        :param pertubation: small unit normal noise for pertubation
        """
        beta_t = _totensor(self.betas[t.cpu()], x_t.shape, x_t)
        alpha_t = 1.0 - beta_t
        sqrt_one_minus_alpha_cumprod_t = _totensor(
            self.sqrt_one_minus_alphas_cumprod[t.cpu()], x_t.shape, x_t
        )

        pred_mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - model_pred * beta_t / sqrt_one_minus_alpha_cumprod_t
        )
        posterior_variance = _totensor(self.posterior_variance[t.cpu()], x_t.shape, x_t)

        return pred_mean + posterior_variance.sqrt() * pertubation

    @torch.no_grad()
    def sample(self, model, n=1, img_size=(64, 64), x_t=None, save_every=False):
        if save_every:
            import os

            os.makedirs("denoising_steps/", exist_ok=True)
        model.eval()
        device = next(model.parameters()).device
        h, w = img_size
        if x_t is None:
            x_t = torch.randn(n, model.out_ch, h, w, device=device)
        else:
            x_t = x_t.to(device)
        for t in reversed(range(self.num_timesteps)):
            # pertubation; refered to z in algorithm for sampling in Ho et al., 2020
            pertubation = 0
            if t > 1:
                pertubation = torch.randn_like(x_t)
            t = torch.full((n,), t, device=device)
            model_pred = model(x_t, t)
            x_t = self.p_sample(x_t, t, model_pred, pertubation)
            if save_every:
                save_image(model_pred, f"denoising_steps/noise_pred__{t[0].item()}.png")
                save_image(x_t, f"denoising_steps/img_pred__{t[0].item()}.png")
        x_t = x_t.clip(-1.0, 1.0)
        return x_t
