import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
from torchvision.utils import save_image, make_grid


# function definitions
# -------------------------------------------------------------------------------------------------
def cosinebetas(steps: int):
    """
    generate betas for Cosine noise schedule
    """
    # f = lambda t: math.cos((t / steps + 0.008) / 1.008 * math.pi * 0.5) ** 2
    # f_0 = f(0)
    # alpha_cum = lambda t: f(t) / f_0
    # beta = [
    #     min(1 - (alpha_cum(t) / alpha_cum(t - 1)), 0.999)
    #     for t in np.linspace(1, 1000, steps)
    # ]
    # return torch.tensor(beta, dtype=torch.float32)

    alpha_cum = lambda t_: math.cos((t_ + 0.008) / 1.008 * math.pi * 0.5) ** 2
    betas = []
    for i in range(steps):
        t1 = i / steps
        t2 = (i + 1) / steps
        # clipping values at 0.009 worked better somehow, will have to investigate
        betas.append(1 - alpha_cum(t2) / alpha_cum(t1))
    return torch.tensor(betas, dtype=torch.float32).clamp_max(0.9) # max of 0.999 make the pred mu very unstable


def linearbetas(steps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """
    generate betas for Linear noise schedule
    """
    return torch.linspace(beta_start, beta_end, steps)


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
    def __init__(
        self,
        num_timesteps: int,
        skip_timesteps=1,
        schedule_name: str = "cosine",
        use_posterior_variance: bool = True,
        clip_denoised: bool = True,
    ):
        self.num_timesteps = num_timesteps
        self.skip_timesteps = skip_timesteps

        self.use_posterior_variance = use_posterior_variance
        self.clip_denoised = clip_denoised
        if schedule_name == "cosine":
            self.betas = cosinebetas(num_timesteps)
        elif schedule_name == "linear":
            self.betas = linearbetas(num_timesteps)
        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

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
        Sample from the forward process using reparameterisation.
        i.e., sample from q(x_{t}|x_{0})
        :param x_start: x_{0} clean datapoint
        :param t: t timestep

        Return: x_{t}, noise
        """
        noise = torch.randn_like(x_start)
        assert x_start.device == t.device, f"{x_start.device} != {t.device}"
        sqrt_alpha_cumprod_t = _totensor(
            self.sqrt_alphas_cumprod[t.cpu()], x_start.shape, t
        )
        sqrt_one_minus_alpha_cumprod_t = _totensor(
            self.sqrt_one_minus_alphas_cumprod[t.cpu()], x_start.shape, t
        )

        x_t = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise

    # def _xstart_from_eps(self, x_t, t, eps):

    #     recip_sqrt_alpha_cumprod_t = 1.0 / self.sqrt_alphas_cumprod

    #     x_start =
    #     return x_start

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
        recip_sqrt_alpha_t = 1.0 / torch.sqrt(1.0 - beta_t)
        sqrt_one_minus_alpha_cumprod_t = _totensor(
            self.sqrt_one_minus_alphas_cumprod[t.cpu()], x_t.shape, x_t
        )

        pred_mean = recip_sqrt_alpha_t * (
            x_t - model_pred * beta_t / sqrt_one_minus_alpha_cumprod_t
        )
        if self.use_posterior_variance:
            posterior_variance = _totensor(
                self.posterior_variance[t.cpu()], x_t.shape, x_t
            )
        else:
            posterior_variance = beta_t

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
        for t in reversed(range(1, self.num_timesteps, self.skip_timesteps)):
            # pertubation; refered to z in algorithm for sampling in Ho et al., 2020
            pertubation = 0
            if t > 1:
                pertubation = torch.randn_like(x_t)
            t = torch.full((n,), t, device=device)

            model_pred = model(x_t, t)
            x_t = self.p_sample(x_t, t, model_pred, pertubation)
            if self.clip_denoised:
                x_t = x_t.clip(-1.0, 1.0)
        x_t = x_t.clip(-1.0, 1.0)
        return x_t

    def training_loss(self, model, x_start):
        device = next(model.parameters()).device
        n = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (n,))
        # sanity check, also easy while debugging
        t = t.to(x_start.device)
        x_t, noise = self.q_sample(x_start, t)

        x_t = x_t.to(device)
        t = t.to(device)
        noise = noise.to(device)

        # forward pass
        pred_noise = model(x_t, t)
        loss = F.mse_loss(pred_noise, noise)

        return loss


# if __name__ == "__main__":
#     # debugging the scaling of the weights by the noise schedule
#     diff = GaussianDiffusion(1000, "cosine")

#     import matplotlib.pyplot as plt

#     # plt.plot(diff.betas)
#     plt.plot(diff.betas.clamp_max(0.02))

#     plt.show()
