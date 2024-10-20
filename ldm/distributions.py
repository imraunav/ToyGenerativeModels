import torch
import numpy as np


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(
            parameters, chunks=2, dim=1
        )  # split along ch
        # config taken from:
        # https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/modules/distributions/distributions.py#L24
        self.logvar = torch.clamp(self.logvar, -30, 20)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.std = self.var = torch.zeros_like(
                self.mean, device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn_like(
            self.mean, device=self.parameters.device
        )
        return x

    def mode(self):
        return self.mean

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    -1.0 + self.var - self.logvar + torch.square(self.mean),
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    -1.0
                    + (self.var / other.var)
                    - self.logvar
                    + other.logvar
                    + torch.square(self.mean - other.mean) / other.var,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )
