"""Independent neural shell interfaces and the disposable Stage-I proxy."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .encoding import HashGridEncoding, SphericalHarmonicEncoding


class ShellSDF(nn.Module):
    """Hash-grid SDF with an analytic sphere prior for initial ray intersections.

    The residual is unconstrained. A copied inner field therefore remains fully
    independent of the outer field after its initial inward offset.
    """

    def __init__(self, bound=1.5, radius=0.8, levels=16, features=2,
                 base_resolution=16, max_resolution=2048, log2_hash_size=19,
                 hidden_dim=64):
        super().__init__()
        self.bound = float(bound)
        self.register_buffer("radius", torch.tensor(float(radius)))
        self.sdf_offset = nn.Parameter(torch.zeros(()))
        self.encoding = HashGridEncoding(levels, features, base_resolution,
                                         max_resolution, log2_hash_size)
        self.network = nn.Sequential(
            nn.Linear(self.encoding.output_dim, hidden_dim), nn.Softplus(beta=100),
            nn.Linear(hidden_dim, hidden_dim), nn.Softplus(beta=100),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, points):
        encoded = self.encoding(points / (2.0 * self.bound) + 0.5)
        sphere = torch.sqrt(points.square().sum(dim=-1) + 1e-12) - self.radius
        return sphere + self.sdf_offset + self.network(encoded).squeeze(-1)

    def gradient(self, points, create_graph=True):
        # Surface normals are needed for rendering even under torch.no_grad().
        with torch.enable_grad():
            samples = points if points.requires_grad else points.detach().requires_grad_(True)
            values = self(samples)
            gradient = torch.autograd.grad(values.sum(), samples,
                                           create_graph=create_graph,
                                           retain_graph=create_graph)[0]
        return gradient if create_graph else gradient.detach()

    def normals(self, points, create_graph=True):
        return F.normalize(self.gradient(points, create_graph), dim=-1, eps=1e-8)

    def initialize_from_outer(self, outer, inset):
        """Copy weights once and shrink the initial zero level set by inset."""
        if inset <= 0:
            raise ValueError("The initial inner-shell inset must be positive")
        self.load_state_dict(outer.state_dict())
        self.bound = outer.bound
        with torch.no_grad():
            self.sdf_offset.add_(float(inset))
        return self

    def eikonal_loss(self, points):
        gradients = self.gradient(points, create_graph=True)
        return (gradients.norm(dim=-1) - 1.0).square().mean()

    def smoothness_loss(self, points, epsilon=0.01):
        neighbors = points + F.normalize(torch.randn_like(points), dim=-1) * epsilon
        normals = self.normals(points, create_graph=True)
        neighbor_normals = self.normals(neighbors, create_graph=True)
        return (normals - neighbor_normals).square().sum(dim=-1).mean()


class ShellMaterial(nn.Module):
    """Homogeneous shell with IOR above air and nonnegative RGB absorption."""

    def __init__(self, ior=1.3, absorption=0.01):
        super().__init__()
        if ior <= 1.0:
            raise ValueError("A dielectric shell must start with an IOR above 1")
        self.raw_ior = nn.Parameter(torch.tensor(math.log(math.expm1(ior - 1.0))))
        absorption = torch.as_tensor(absorption, dtype=torch.float32).expand(3).clone()
        if torch.any(absorption < 0):
            raise ValueError("Absorption must be nonnegative")
        self.raw_absorption = nn.Parameter(torch.log(torch.expm1(absorption.clamp_min(1e-8))))

    @property
    def ior(self):
        return 1.0 + F.softplus(self.raw_ior)

    @property
    def absorption(self):
        return F.softplus(self.raw_absorption)


class DirectionProxy(nn.Module):
    """Degree-4 SH and three 128-unit hidden layers; no positional input."""

    def __init__(self):
        super().__init__()
        self.encoding = SphericalHarmonicEncoding(degree=4)
        self.network = nn.Sequential(
            nn.Linear(25, 128), nn.ReLU(inplace=False),
            nn.Linear(128, 128), nn.ReLU(inplace=False),
            nn.Linear(128, 128), nn.ReLU(inplace=False),
            nn.Linear(128, 3), nn.Sigmoid(),
        )

    def forward(self, directions):
        return self.network(self.encoding(directions))
