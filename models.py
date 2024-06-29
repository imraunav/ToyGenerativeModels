"""
A common model across all generative models
"""

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math


def efficientconv(in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
    """
    Simple trick to make convolution more efficinet with minor performance tradeoff.
    Idea from EfficientNet
    """
    if kernel_size == 1:
        return nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                (kernel_size, 1),
                (stride, 1),
                (padding, 0),
                groups=in_ch,
                bias=bias,
            ),
            nn.Conv2d(
                in_ch,
                in_ch,
                (1, kernel_size),
                (1, stride),
                (0, padding),
                groups=in_ch,
                bias=bias,
            ),
            nn.Conv2d(in_ch, out_ch, 1, bias=bias),
        )


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Taken from https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/

    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, ResBlock) or isinstance(
                layer, TimestepEmbedSequential
            ):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        emb_dim,
        num_groups=8,
        model_ch=None,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        if model_ch is None:
            model_ch = out_ch
        self.model_ch = model_ch
        self.num_groups = num_groups
        self.emb_dim = emb_dim

        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_ch),
            nn.SiLU(),
            efficientconv(in_ch, model_ch, 3, 1, 1),
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups, model_ch),
            nn.SiLU(),
            efficientconv(model_ch, out_ch, 3, 1, 1),
        )

        self.mlp = nn.Sequential(
            nn.GroupNorm(num_groups, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, model_ch),
        )

        if in_ch == out_ch:
            self.residual = nn.Identity()
        else:
            self.residual = efficientconv(in_ch, out_ch, 1)

    def forward(self, x, emb=None):
        h = self.conv1(x)
        emb = self.mlp(emb)
        while len(h.shape) > len(emb.shape):
            emb = emb[..., None]
        h = h + emb
        h = self.conv2(h)

        return h + self.residual(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=1, num_groups=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_groups = num_groups

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = efficientconv(channels, channels * 3 * num_heads, 1)
        self.proj_out = efficientconv(channels * num_heads, channels, 1)

        # init
        self._weight_init()

    @torch.no_grad()
    def _weight_init(self):
        for p in self.proj_out.parameters():
            p.data.zero_()

    def forward(self, x):
        N, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # (N, (C + head), H, W)
        qkv = rearrange(
            qkv,
            "n (c head) h w -> n head (h w) c",
            head=self.num_heads,
        )  # (N, head, L, C)

        q, k, v = torch.split(qkv, C, dim=3)
        h = F.scaled_dot_product_attention(q, k, v)  # flash attention
        h = rearrange(
            h,
            "n head (h w) c -> n (head c) h w",
            h=H,
            w=W,
            head=self.num_heads,
        )
        h = self.proj_out(h)
        return x + h


class UNetDiffusion(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        emb_dim,
        model_ch=64,
        timesteps=None,
        depth=4,
        num_resblock=1,
        attn_res=[4],
        num_attn_heads=4,
        num_classes=None,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.emb_dim = emb_dim
        self.model_ch = model_ch
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.depth = depth
        self.num_attn_heads = num_attn_heads

        self.in_conv = efficientconv(in_ch, model_ch, 3, 1, 1)
        self.downs = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.up_samplers = nn.ModuleList()
        self.out_conv = efficientconv(model_ch + model_ch, out_ch, 3, 1, 1)

        ds = 1  # downsampling tracking
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.SiLU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, self.emb_dim)

        # encoder definition
        skip_ch = []
        for _ in range(depth):
            _resblocks = nn.ModuleList()
            in_ch = model_ch
            model_ch *= 2

            for _ in range(num_resblock):
                if ds in attn_res:
                    _block = TimestepEmbedSequential(
                        ResBlock(in_ch, model_ch, emb_dim),
                        SelfAttention(model_ch, num_attn_heads),
                    )
                else:
                    _block = ResBlock(in_ch, model_ch, emb_dim)
                _resblocks.append(_block)
                in_ch = model_ch

            self.downs.append(_resblocks)
            skip_ch.append(model_ch)

            ds *= 2
            self.down_samplers.append(
                nn.Sequential(
                    efficientconv(model_ch, model_ch // 4, 1),
                    nn.PixelUnshuffle(2),
                )
            )

        # bottleneck
        _resblocks = []
        for _ in range(num_resblock):
            _this_block = []
            _this_block.append(ResBlock(model_ch, model_ch, emb_dim))
            if ds in attn_res:
                _this_block.append(SelfAttention(model_ch, num_attn_heads))
            _this_block.append(ResBlock(model_ch, model_ch, emb_dim))
            _resblocks.append(TimestepEmbedSequential(*_this_block))
        self.bottleneck = TimestepEmbedSequential(*_resblocks)

        # decoder definition
        for _ in range(depth):
            ds //= 2
            self.up_samplers.append(
                nn.Sequential(
                    efficientconv(model_ch, model_ch * 4, 1),
                    nn.PixelShuffle(2),
                )
            )
            _resblocks = nn.ModuleList()
            skip = skip_ch.pop()
            in_ch = model_ch
            model_ch //= 2

            for _ in range(num_resblock):
                if ds in attn_res:
                    _block = TimestepEmbedSequential(
                        ResBlock(in_ch + skip, model_ch, emb_dim),
                        SelfAttention(model_ch, num_attn_heads),
                    )
                else:
                    _block = ResBlock(in_ch + skip, model_ch, emb_dim)
                _resblocks.append(_block)
                in_ch = model_ch
            self.ups.append(_resblocks)

    def forward(self, x, t, y=None):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # time embedding op
        emb = timestep_embedding(t, self.emb_dim)
        emb = self.mlp(emb)

        # class embedding op
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],), "batch size mismatch"
            emb = emb + self.label_emb(y)

        h = self.in_conv(x)

        # encoder path
        h_skip = [h]
        for m_list, m_down in zip(self.downs, self.down_samplers):
            for m in m_list:
                h = m(h, emb)
                h_skip.append(h)
            h = m_down(h)

        # bottleneck path
        h = self.bottleneck(h, emb)

        # decoder path
        for m_list, m_up in zip(self.ups, self.up_samplers):
            h = m_up(h)
            for m in m_list:
                h = torch.cat([h, h_skip.pop()], dim=1)
                h = m(h, emb)
        h = torch.cat([h, h_skip.pop()], dim=1)
        x = self.out_conv(h)
        return x


if __name__ == "__main__":
    # # ResBlock test
    # m = ResBlock(32, 64)
    # x = torch.randn(32, 32, 64, 64)
    # y = m(x)
    # print(y.shape)  # torch.Size([32, 64, 64, 64])

    # Unet test
    m = UNetDiffusion(
        3,
        3,
        emb_dim=128,
        timesteps=1000,
        depth=4,
        model_ch=32,
        num_resblock=2,
        attn_res=[4],
        num_attn_heads=4,
        num_classes=10,
    )
    torch.compile(m)
    num_par = 0
    for p in m.parameters():
        num_par += p.numel()
    print(f"No. parameters: {num_par/1_000_000:.3f}M")  # 9.903M
    x = torch.randn((4, 3, 64, 64))
    t = torch.randint(0, 1000, (4,))
    y = torch.randint(0, 10, (4,))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        x_ = m(x, t, y)
    print(x_.shape)  # torch.Size([4, 3, 64, 64])

    # # Attention test
    # m = SelfAttention(8, num_heads=4)
    # x = torch.randn((4, 8, 32, 32))
    # y = m(x)
    # print(y.shape)

    pass
