"""Member-attention temporal corrector trained from scratch."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MemberAttention(nn.Module):
    def __init__(self, features, width, heads, queries):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(features, width), nn.GELU(), nn.Linear(width, width)
        )
        self.queries = nn.Parameter(torch.randn(queries, width) * 0.02)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Linear(width * (queries + 1), width)

    def forward(self, members, context):
        # Members [M,H,W,F]; the forecast month is never an attention axis.
        count, h, w, _ = members.shape
        encoded = self.encode(members.permute(1, 2, 0, 3).reshape(h * w, count, -1))
        query = self.queries[None] + context.flatten(2).transpose(1, 2).reshape(
            h * w, 1, -1
        )
        normalized = self.norm(encoded)
        attended = self.attention(query, normalized, normalized, need_weights=False)[0]
        pooled = torch.cat([attended.flatten(1), encoded.mean(1)], dim=1)
        return self.output(pooled).T.reshape(1, -1, h, w)


class SpatialBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )

    def forward(self, x):
        return x + self.layers(x)


class Corrector(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["width"]
        inputs = config.get("input_channels", 104)
        self.stride = config["coarse_stride"]
        self.context = nn.Conv2d(inputs, width, 1)
        self.members = nn.ModuleDict(
            {
                group: MemberAttention(
                    channels, width, config["heads"], config["queries"]
                )
                for group, channels in (("seas5", 2), ("dwd", 2), ("gefs", 12))
            }
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(width * 4, width, 1),
            *[SpatialBlock(width, d) for d in (1, 2, 4, 8)],
        )
        self.fine = nn.Sequential(nn.Conv2d(inputs, 16, 1), nn.GELU())
        self.output = nn.Sequential(
            nn.Conv2d(width + 16, 32, 1), nn.GELU(), nn.Conv2d(32, 1, 1)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, context, members):
        coarse = self.context(context[:, :, :: self.stride, :: self.stride])
        features = [coarse] + [
            layer(members[group], coarse) for group, layer in self.members.items()
        ]
        spatial = self.spatial(torch.cat(features, dim=1))
        spatial = F.interpolate(
            spatial, size=context.shape[-2:], mode="bilinear", align_corners=True
        )
        return self.output(torch.cat([spatial, self.fine(context)], dim=1))[0, 0]


def runtime(config, seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for numerical execution")
    torch.set_num_threads(config["threads"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def encode_members(raw, reference, stride):
    base = reference[::stride, ::stride]
    result = {}
    for group in ("seas5", "dwd"):
        values = raw[group]
        result[group] = np.stack([np.log1p(values) / 2, (values - base) / 5], axis=-1)
    values = raw["gefs"].transpose(0, 2, 3, 1)
    coverage = np.broadcast_to(raw["coverage"], values.shape)
    anomaly = (values - base[None, :, :, None]) * coverage / 5
    result["gefs"] = np.concatenate([np.log1p(values) / 2, anomaly, coverage], axis=-1)
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in result.items()}
