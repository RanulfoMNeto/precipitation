"""U-Net used by the five-model forecast recipe."""

from __future__ import annotations


def unet_model():
    # Compact spatial correction network with a residual precipitation output.
    import torch
    from torch import nn
    from torch.nn import functional as f

    def block(inputs, outputs):
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(4, outputs),
            nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(4, outputs),
            nn.SiLU(),
        )

    class Corrector(nn.Module):
        def __init__(self):
            super().__init__()
            widths = (16, 32, 64, 128)
            self.down = nn.ModuleList(
                block(a, b) for a, b in zip((25, *widths[:-1]), widths, strict=True)
            )
            self.up = nn.ModuleList(
                block(a + b, b)
                for a, b in zip(widths[:0:-1], widths[-2::-1], strict=True)
            )
            self.head = nn.Conv2d(16, 1, 1)

        def forward(self, features, baseline):
            h, w = features.shape[-2:]
            x = f.pad(features, (0, -w % 8, 0, -h % 8), mode="replicate")
            skips = []
            for index, layer in enumerate(self.down):
                x = layer(f.avg_pool2d(x, 2) if index else x)
                skips.append(x)
            for layer, skip in zip(self.up, reversed(skips[:-1]), strict=True):
                x = f.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
                x = layer(torch.cat((x, skip), dim=1))
            return baseline.float() + self.head(x)[:, 0, :h, :w].float()

    return Corrector()
