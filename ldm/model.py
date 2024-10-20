from abc import abstractmethod
import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
import numpy as np

from ldm.distributions import DiagonalGaussianDistribution
from ddpm.model import (
    normalization,
    _zero_init,
    TimestepSequential,
    ResBlock,
)


# module definitions
# -------------------------------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, ch, num_heads, dropout_p=0.0):
        assert ch % num_heads == 0, f"{ch=} % {num_heads=} != 0"
        super().__init__()
        self.ch = ch
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.norm = normalization(ch)
        self.qkv = nn.Linear(ch, 3 * ch, bias=False)
        self.proj_out = _zero_init(nn.Linear(ch, ch, bias=False))

    def forward(self, x):
        N, C, H, W = x.shape
        norm_x = self.norm(x)
        norm_x = rearrange(norm_x, "n c h w -> n (h w) c")
        # this is linear layer, my ddpm implementation uses conv1d
        qkv = self.qkv(norm_x)
        qkv = rearrange(qkv, "n l (head c) -> n head l c", head=self.num_heads)
        qkv = qkv.contiguous()
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)
        h = rearrange(h, "n head l c -> n l (head c)")
        h = self.proj_out(h)
        h = rearrange(h, "n (h w) c -> n c h w", h=H, w=W)
        return x + h


class UpDownSample(nn.Module):
    def __init__(self, ch, use_conv=True):
        super().__init__()
        self.ch = ch
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        raise NotImplementedError


class Downsample(UpDownSample):
    def forward(self, x):
        h = F.avg_pool2d(x, 2)
        if self.use_conv:
            h = self.conv(h)
        return h


class Upsample(UpDownSample):
    def forward(self, x):
        h = x
        if self.use_conv:
            h = self.conv(h)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        return h


class Encoder(nn.Module):
    def __init__(
        self,
        in_ch,
        model_ch,
        num_resblock,
        z_dim,
        double_z=False,
        ch_mult=[1, 1, 1, 1],
        attn_ds=[4],
        num_heads=1,
        dropout_p=0.0,
        use_conv=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.model_ch = model_ch
        self.num_resblock = num_resblock
        self.ch_mult = ch_mult
        self.attn_ds = attn_ds
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.double_z = double_z
        if double_z:
            z_dim *= 2
        self.z_dim = z_dim

        self.in_conv = nn.Conv2d(in_ch, model_ch, 3, 1, 1)

        self.downs = nn.ModuleList()
        self.downsample = nn.ModuleList()

        prev_ch = model_ch
        ds = 1
        for level, mult in enumerate(ch_mult):
            # skip connections go from _block to _block
            _block = nn.ModuleList()
            for i in range(num_resblock):
                layers = [ResBlock(prev_ch, model_ch * mult, dropout_p=dropout_p)]
                prev_ch = model_ch * mult
                if ds in attn_ds:
                    layers.append(SelfAttention(prev_ch, num_heads, dropout_p))
                _block.append(TimestepSequential(*layers))
            self.downs.append(_block)
            self.downsample.append(Downsample(prev_ch, use_conv))
            ds *= 2

        self.bottleneck = TimestepSequential(
            ResBlock(prev_ch, prev_ch, dropout_p=dropout_p),
            SelfAttention(prev_ch, num_heads),
            ResBlock(prev_ch, prev_ch, dropout_p=dropout_p),
        )

        self.z_conv = nn.Sequential(
            normalization(prev_ch),
            nn.SiLU(),
            nn.Conv2d(prev_ch, z_dim, 3, 1, 1),
        )

    def forward(self, x):
        emb = None  # simple way to adapt code without much changes
        # !WARNING: be careful while debugging, emb is a wasted parameter passes

        h = self.in_conv(x)
        for block, downsample in zip(self.downs, self.downsample):
            for layer in block:
                h = layer(h, emb)
            h = downsample(h)

        h = self.bottleneck(h, emb)

        return self.z_conv(h)


class Decoder(nn.Module):
    def __init__(
        self,
        out_ch,
        model_ch,
        num_resblock,
        z_dim,
        ch_mult=[1, 1, 1, 1],
        attn_ds=[4],
        num_heads=1,
        dropout_p=0.0,
        use_conv=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.out_ch = out_ch
        self.model_ch = model_ch
        self.num_resblock = num_resblock
        self.ch_mult = ch_mult
        self.attn_ds = attn_ds
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.z_dim = z_dim

        prev_ch = ch_mult[-1] * model_ch
        self.in_conv = nn.Conv2d(z_dim, prev_ch, 3, 1, 1)

        self.bottleneck = TimestepSequential(
            ResBlock(prev_ch, prev_ch, dropout_p=dropout_p),
            SelfAttention(prev_ch, num_heads),
            ResBlock(prev_ch, prev_ch, dropout_p=dropout_p),
        )

        self.upsample = nn.ModuleList()
        self.ups = nn.ModuleList()

        prev_ch = model_ch
        ds = 2 ** len(ch_mult)
        for level, mult in enumerate(reversed(ch_mult)):
            self.upsample.append(Upsample(prev_ch, use_conv))
            _block = nn.ModuleList()
            for i in range(num_resblock):
                layers = [ResBlock(prev_ch, model_ch * mult, dropout_p=dropout_p)]
                prev_ch = model_ch * mult
                if ds in attn_ds:
                    layers.append(
                        SelfAttention(prev_ch, num_heads, dropout_p=dropout_p)
                    )

                _block.append(TimestepSequential(*layers))
            self.ups.append(_block)
            ds //= 2

        self.out_conv = nn.Sequential(
            normalization(prev_ch),
            nn.SiLU(),
            nn.Conv2d(prev_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x):
        emb = None  # simple way to adapt code without much changes

        h = self.in_conv(x)
        h = self.bottleneck(h, emb)

        for block, upsample in zip(self.ups, self.upsample):
            h = upsample(h)
            for layer in block:
                h = layer(h, emb)

        return self.out_conv(h)


# m = nn.Sequential(
#     Encoder(3, 3, 32, 2, 5, attn_ds=[2, 4, 8]),
#     Decoder(3, 3, 32, 2, 5, attn_ds=[2, 4, 8]),
# )

# x = torch.randn(4, 3, 128, 128)
# with torch.no_grad():
#     y = m(x)

# print(y.shape)


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        model_ch,
        embed_dim,
        num_resblock,
        z_dim,
        double_z=False,
        ch_mult=[1, 1, 1, 1],
        attn_ds=[4],
        num_heads=1,
        dropout_p=0.0,
        use_conv=True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.model_ch = model_ch
        self.embed_dim = embed_dim
        self.num_resblock = num_resblock
        self.z_dim = z_dim
        self.double_z = double_z
        self.ch_mult = ch_mult
        self.attn_ds = attn_ds
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.use_conv = use_conv

        config = dict(
            in_ch=in_ch,
            out_ch=out_ch,
            model_ch=model_ch,
            num_resblock=num_resblock,
            z_dim=z_dim,
            double_z=double_z,
            ch_mult=ch_mult,
            attn_ds=attn_ds,
            num_heads=num_heads,
            dropout_p=dropout_p,
            use_conv=use_conv,
        )
        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)
        assert double_z, "need double z-dim for AutoencoderKL"
        self.quant_conv = nn.Conv2d(2 * z_dim, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_dim, 1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
