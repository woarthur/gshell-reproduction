"""Learnable distant illumination, stored as equirectangular log radiance."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class EnvironmentMap(nn.Module):
    """Periodic longitude texture; +Z is north, +X is the central meridian.

    Texels contain linear RGB radiance in logarithmic parameterization. The
    forward pass bilinearly interpolates radiance, not log radiance. No exposure
    or display transfer function is applied by this module.
    """

    def __init__(self, height=128, width=256, initial_radiance=0.5):
        super().__init__()
        if height < 2 or width < 2 or initial_radiance <= 0:
            raise ValueError("Environment size must be >= 2 and radiance positive")
        self.log_radiance = nn.Parameter(
            torch.full((height, width, 3), math.log(initial_radiance))
        )

    @property
    def height(self):
        return self.log_radiance.shape[0]

    @property
    def width(self):
        return self.log_radiance.shape[1]

    def radiance(self):
        """Return the linear HDR image [H, W, 3] for export or inspection."""
        return self.log_radiance.exp()

    def forward(self, directions):
        directions = F.normalize(directions, dim=-1)
        x_dir, y_dir = directions[..., 0], directions[..., 1]
        at_pole = x_dir.square() + y_dir.square() < 1e-12
        # Longitude has no meaning at a pole; select a finite meridian there.
        longitude = torch.atan2(
            torch.where(at_pole, torch.zeros_like(y_dir), y_dir),
            torch.where(at_pole, torch.ones_like(x_dir), x_dir),
        )
        # Clamping avoids the singular derivative of acos exactly at a pole.
        latitude = torch.acos(directions[..., 2].clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        x = (longitude / (2.0 * math.pi) + 0.5) * self.width - 0.5
        y = (latitude / math.pi * self.height - 0.5).clamp(0, self.height - 1)
        x0_float, y0_float = torch.floor(x), torch.floor(y)
        wx, wy = (x - x0_float)[..., None], (y - y0_float)[..., None]
        x0 = x0_float.long().remainder(self.width)
        x1 = (x0 + 1).remainder(self.width)
        y0 = y0_float.long()
        y1 = (y0 + 1).clamp(max=self.height - 1)
        texture = self.radiance()
        top = texture[y0, x0] * (1 - wx) + texture[y0, x1] * wx
        bottom = texture[y1, x0] * (1 - wx) + texture[y1, x1] * wx
        return top * (1 - wy) + bottom * wy

    def total_variation(self):
        """Log-radiance TV with the longitude seam included in the prior."""
        longitude = self.log_radiance - torch.roll(self.log_radiance, 1, dims=1)
        latitude = self.log_radiance[1:] - self.log_radiance[:-1]
        return longitude.abs().mean() + latitude.abs().mean()

    def upsample(self, height, width):
        """Replace the texture parameter; rebuild its optimizer after this call.

        Explicitly sample the periodic texture at the new texel centers to keep
        interpolation consistent at the equirectangular seam. Interpolation is
        performed in log space for the coarse-to-fine initialization.
        """
        if height < 2 or width < 2:
            raise ValueError("Environment size must be >= 2")
        with torch.no_grad():
            old = self.log_radiance
            x = (torch.arange(width, device=old.device, dtype=old.dtype) + 0.5)
            x = x * (self.width / float(width)) - 0.5
            y = (torch.arange(height, device=old.device, dtype=old.dtype) + 0.5)
            y = (y * (self.height / float(height)) - 0.5).clamp(0, self.height - 1)
            x0f, y0f = x.floor(), y.floor()
            x0 = x0f.long().remainder(self.width)
            x1 = (x0 + 1).remainder(self.width)
            y0 = y0f.long()
            y1 = (y0 + 1).clamp(max=self.height - 1)
            wx, wy = (x - x0f)[None, :, None], (y - y0f)[:, None, None]
            top = old[y0[:, None], x0[None, :]] * (1 - wx)
            top = top + old[y0[:, None], x1[None, :]] * wx
            bottom = old[y1[:, None], x0[None, :]] * (1 - wx)
            bottom = bottom + old[y1[:, None], x1[None, :]] * wx
            values = top * (1 - wy) + bottom * wy
        self.log_radiance = nn.Parameter(values)
        return self.log_radiance
