from abc import abstractmethod
import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange


def normalization(ch):
    return nn.GroupNorm(32, ch)


def broadcast_tensor_op(x, shape):
    while len(x.shape) < len(shape):
        x = x[..., None]
    return x


@torch.no_grad()
def _zero_init(m: nn.Module):
    """
    Initialize module with 0 parameters and return the same module
    """
    for p in m.parameters():
        nn.init.constant_(p.data, 0.0)
    return m


def timestep_embedding(timesteps, dim, max_period=10000):
    device = timesteps.device
    half = dim // 2
    i = torch.arange(0, half, dtype=torch.float32)
    freq = torch.exp(-math.log(max_period) * i / half).to(device)
    args = timesteps[..., None].float() * freq[None, ...]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 != 0:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepLayer(nn.Module):
    @abstractmethod
    def forward(self):
        pass


class TimestepSequential(nn.Sequential, TimestepLayer):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepLayer):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepLayer):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, dropout_p=0.3):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        assert time_emb_dim is None or isinstance(time_emb_dim, int)
        self.time_emb_dim = time_emb_dim

        self.conv1 = nn.Sequential(
            normalization(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
        )

        self.conv2 = nn.Sequential(
            normalization(out_ch),
            nn.SiLU(),
            _zero_init(nn.Conv2d(out_ch, out_ch, 3, 1, 1)),
        )

        if time_emb_dim is not None:
            self.mlp = nn.Sequential(
                normalization(time_emb_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p),
                nn.Linear(time_emb_dim, out_ch),
            )
        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x)
        if self.time_emb_dim is not None:
            assert (
                emb is not None
            ), "Need embedding for module initialized with timestep_emb support"
            emb = self.mlp(emb)
            emb = broadcast_tensor_op(emb, h.shape)
            h = emb + h
        h = self.conv2(h)
        x = self.residual(x)
        return h + x


class SelfAttention(nn.Module):
    def __init__(self, ch, num_heads):
        assert ch % num_heads == 0
        super().__init__()
        self.ch = ch
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv1d(ch, 3 * ch, 1)
        self.proj_out = _zero_init(nn.Conv1d(ch, ch, 1))

    def forward(self, x):
        N, C, H, W = x.shape
        x = rearrange(x, "n c h w -> n c (h w)")
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x)
        qkv = rearrange(qkv, "n (head c) l -> n head l c", head=self.num_heads)
        qkv = qkv.contiguous()
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "n head l c -> n (head c) l")
        h = self.proj_out(h)
        return rearrange(x + h, "n c (h w) -> n c h w", c=C, h=H, w=W)


class UnetDiffusion(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        model_ch,
        timesteps,
        time_emb_dim,
        num_resblock,
        ch_mult=[1, 1, 2, 2],
        attn_ds=[4],
        num_heads=1,
        dropout_p=0.3,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.model_ch = model_ch
        self.timesteps = timesteps
        self.time_emb_dim = time_emb_dim
        self.num_resblock = num_resblock
        self.ch_mult = ch_mult
        self.attn_ds = attn_ds
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.in_conv = nn.Conv2d(in_ch, model_ch, 3, 1, 1)
        skip_ch = [model_ch]

        self.downs = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.ups = nn.ModuleList()
        prev_ch = model_ch
        ds = 1
        for level, mult in enumerate(ch_mult):
            # skip connections go from _block to _block
            _block = nn.ModuleList()
            for i in range(num_resblock):
                layers = [ResBlock(prev_ch, model_ch * mult, time_emb_dim, dropout_p)]
                prev_ch = model_ch * mult
                if ds in attn_ds:
                    layers.append(SelfAttention(prev_ch, num_heads))

                _block.append(TimestepSequential(*layers))
                skip_ch.append(prev_ch)
            self.downs.append(_block)
            if level != len(ch_mult) - 1:
                self.downsample.append(nn.AvgPool2d(2))
                ds *= 2
            else:
                self.downsample.append(nn.Identity())

        self.bottleneck = TimestepSequential(
            ResBlock(prev_ch, prev_ch, time_emb_dim, dropout_p),
            SelfAttention(prev_ch, num_heads),
            ResBlock(prev_ch, prev_ch, time_emb_dim, dropout_p),
        )

        for level, mult in enumerate(reversed(ch_mult)):
            if level != 0:
                self.upsample.append(nn.UpsamplingNearest2d(scale_factor=2))
            else:
                self.upsample.append(nn.Identity())
            _block = nn.ModuleList()
            for i in range(num_resblock):
                layers = [
                    ResBlock(
                        prev_ch + skip_ch.pop(),
                        model_ch * mult,
                        time_emb_dim,
                        dropout_p,
                    )
                ]
                prev_ch = model_ch * mult
                if ds in attn_ds:
                    layers.append(SelfAttention(prev_ch, num_heads))

                _block.append(TimestepSequential(*layers))
            self.ups.append(_block)
            ds //= 2

        prev_ch += skip_ch.pop()
        self.out_conv = nn.Sequential(
            nn.GroupNorm(32, prev_ch),
            nn.SiLU(),
            _zero_init(nn.Conv2d(prev_ch, out_ch, 3, 1, 1)),
        )

        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 4 * time_emb_dim),
            nn.SiLU(),
            nn.Linear(4 * time_emb_dim, time_emb_dim),
        )

    def forward(self, x, t):
        emb = timestep_embedding(t, self.time_emb_dim)
        emb = self.mlp(emb)

        h = self.in_conv(x)
        h_skip = [h]
        for block, downsample in zip(self.downs, self.downsample):
            for layer in block:
                h = layer(h, emb)
                h_skip.append(h)
            h = downsample(h)

        h = self.bottleneck(h, emb)

        for block, upsample in zip(self.ups, self.upsample):
            h = upsample(h)
            for layer in block:
                h = torch.cat([h, h_skip.pop()], dim=1)
                h = layer(h, emb)
        h = torch.cat([h, h_skip.pop()], dim=1)
        return self.out_conv(h)


# m = UnetDiffusion(3, 3, 128, 1000, 512, 2, [1, 1, 1, 1], [2, 4, 8], num_heads=2)
# param_count = 0
# for p in m.parameters():
#     param_count += p.numel()
# print(f"{param_count=:,}")
# x = torch.randn(4, 3, 32, 32)
# t = torch.randint(0, 1000, (4,))
# with torch.no_grad():
#     with torch.autocast("cpu", torch.bfloat16):
#         y = m(x, t)
# print(y.shape)
