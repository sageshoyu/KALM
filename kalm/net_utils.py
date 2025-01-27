import math
from dataclasses import dataclass

import einops
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


@dataclass
class DiffusionHeadOutDict(object):
    position: torch.Tensor
    rotation_6d: torch.Tensor
    openess: torch.Tensor

    def get_10d_vector(self):
        return torch.cat((self.position, self.rotation_6d, self.openess), -1)


class KpEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim, n_kp=8):
        super().__init__()
        self.encode_posfirst = nn.Sequential(nn.Linear(3, emb_dim), nn.ReLU(), nn.Linear(emb_dim, in_dim))
        self.encode_pos = nn.Sequential(
            nn.Linear(in_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim // 2),
        )
        self.out = nn.Sequential(
            nn.Linear(emb_dim // 2 * n_kp, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.in_dim = in_dim
        self.n_kp = n_kp

    def forward(self, x_feat, x_pos):
        x_pos_emb = self.encode_posfirst(x_pos)
        concat_input = torch.cat((x_feat, x_pos_emb), dim=2)
        out = self.encode_pos(concat_input).view(x_feat.shape[0], -1)
        out = self.out(out)
        return out


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2, padding_mode="replicate"),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        return self.to_out(out)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size),
            ]
        )

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.attn = Residual(LinearAttention(out_channels))

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        out = self.attn(out[..., None])
        # out = self.global_mixing(out)
        out = out[..., 0]
        return out + self.residual_conv(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class TemporalUnet_diffuser(nn.Module):
    def __init__(self, traj_len, transition_dim, dim=128, dim_mults=(1, 2, 4, 8), n_kp=6):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        cond_dim = dim + 2048 + 3 * n_kp
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.encode_traj = nn.Sequential(nn.Linear(9, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.encode_scene_feat = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim * 8))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=cond_dim, horizon=traj_len),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=cond_dim, horizon=traj_len),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

            if not is_last:
                traj_len = traj_len // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=cond_dim, horizon=traj_len)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=cond_dim, horizon=traj_len)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=cond_dim, horizon=traj_len),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=cond_dim, horizon=traj_len),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

            if not is_last:
                traj_len = traj_len * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dims[1], dim, kernel_size=3),
            nn.Conv1d(dim, 10, 1),
        )
        self.self_attn = nn.MultiheadAttention(dim * 8, 8, dropout=0.0, batch_first=True)  # , kdim=dim, vdim=dim)

    def forward(self, x, time_step, input_dict):
        """
        x : [ batch x horizon x transition ]
        """
        bs = x.shape[0]
        scene_feats = self.encode_scene_feat(input_dict.keypoints_feats)  # B F
        scene_pcds = input_dict.keypoints_pos

        x = self.encode_traj(x)
        x = einops.rearrange(x, "b t emb -> b emb t")
        time_emb = self.time_mlp(time_step)

        time_emb = torch.cat((time_emb, scene_feats, scene_pcds.reshape(bs, -1)), dim=1)

        h = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, time_emb)
            x = resnet2(x, time_emb)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb)
        x = self.mid_block2(x, time_emb)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, time_emb)
            x = resnet2(x, time_emb)
            x = upsample(x)
        x = self.final_conv(x)

        x = einops.rearrange(x, "b emb t -> b t emb")
        out = DiffusionHeadOutDict(position=x[..., :3], rotation_6d=x[..., 3:9], openess=x[..., 9:])
        return out
