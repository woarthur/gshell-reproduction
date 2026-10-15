"""Torch encodings used by the two shell fields and direction proxy."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.sh_utils import eval_sh


class HashGridEncoding(nn.Module):
    """Trilinear multi-resolution hash grid, without a custom CUDA extension.

    Coordinates are expected in [0, 1]. Small grids use dense indexing; larger
    grids hash integer vertices into at most 2**log2_hash_size entries per level.
    """

    def __init__(self, levels=16, features=2, base_resolution=16,
                 max_resolution=2048, log2_hash_size=19):
        super().__init__()
        if levels < 1 or base_resolution < 1 or max_resolution < base_resolution:
            raise ValueError("Invalid multi-resolution hash-grid dimensions")
        self.levels = levels
        self.features = features
        self.output_dim = levels * features
        growth = (max_resolution / float(base_resolution)) ** (1.0 / max(levels - 1, 1))
        self.resolutions = [int(math.floor(base_resolution * growth ** level))
                            for level in range(levels)]
        if levels > 1:
            self.resolutions[-1] = max_resolution
        self.table_sizes = [min(2 ** log2_hash_size, (resolution + 1) ** 3)
                            for resolution in self.resolutions]
        self.tables = nn.ParameterList([
            nn.Parameter(torch.empty(size, features)) for size in self.table_sizes
        ])
        for table in self.tables:
            nn.init.uniform_(table, -1e-4, 1e-4)
        self.register_buffer("corners", torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ], dtype=torch.long))

    def forward(self, coordinates):
        shape = coordinates.shape[:-1]
        coordinates = coordinates.reshape(-1, 3).clamp(0.0, 1.0)
        encoded = []
        for resolution, size, table in zip(self.resolutions, self.table_sizes, self.tables):
            scaled = coordinates * resolution
            lower = torch.floor(scaled).long().clamp(max=resolution - 1)
            fraction = scaled - lower.to(scaled.dtype)
            vertices = lower[:, None, :] + self.corners[None, :, :]
            if (resolution + 1) ** 3 <= size:
                stride = resolution + 1
                indices = vertices[..., 0] + stride * (vertices[..., 1] + stride * vertices[..., 2])
            else:
                indices = ((vertices[..., 0] * 1) ^
                           (vertices[..., 1] * 2654435761) ^
                           (vertices[..., 2] * 805459861)) % size
            weights = torch.where(self.corners[None, :, :].bool(),
                                  fraction[:, None, :], 1.0 - fraction[:, None, :])
            encoded.append((table[indices] * weights.prod(dim=-1, keepdim=True)).sum(dim=1))
        return torch.cat(encoded, dim=-1).reshape(shape + (self.output_dim,))


class SphericalHarmonicEncoding(nn.Module):
    """Real SH in the same ordering and normalization as upstream 3DGS."""

    def __init__(self, degree=4):
        super().__init__()
        if not 0 <= degree <= 4:
            raise ValueError("Supported SH degrees are 0 through 4")
        self.degree = degree
        self.output_dim = (degree + 1) ** 2
        self.register_buffer("basis", torch.eye(self.output_dim))

    def forward(self, directions):
        directions = F.normalize(directions, dim=-1)
        # Identity coefficients evaluate each basis function independently.
        basis = self.basis.to(dtype=directions.dtype)
        return eval_sh(self.degree, basis, directions)
