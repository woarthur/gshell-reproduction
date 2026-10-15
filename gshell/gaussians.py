"""Explicit interior primitives with GShell's per-unit-length density (Eq. 9).

SH coefficients, logarithmic scales and wxyz quaternions follow the upstream
3DGS representation. Screen-space opacity is replaced by softplus density.
"""

import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.sh_utils import RGB2SH, eval_sh


def inverse_softplus(value):
    return value + torch.log(-torch.expm1(-value))


def quaternion_matrix(quaternions):
    """Device-independent equivalent of upstream build_rotation, in wxyz order."""
    w, x, y, z = F.normalize(quaternions, dim=-1).unbind(-1)
    entries = (
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    )
    return torch.stack(entries, dim=-1).reshape(-1, 3, 3)


class InteriorGaussians(nn.Module):
    PARAMETER_NAMES = (
        "_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_density"
    )

    def __init__(self, sh_degree=3):
        super().__init__()
        if not 0 <= sh_degree <= 4:
            raise ValueError("SH degree must be between 0 and 4")
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.percent_dense = 0.01
        self._xyz = nn.Parameter(torch.empty(0, 3))
        self._features_dc = nn.Parameter(torch.empty(0, 1, 3))
        self._features_rest = nn.Parameter(torch.empty(0, (sh_degree + 1) ** 2 - 1, 3))
        self._scaling = nn.Parameter(torch.empty(0, 3))
        self._rotation = nn.Parameter(torch.empty(0, 4))
        self._density = nn.Parameter(torch.empty(0, 1))
        self.register_buffer("gradient_accum", torch.empty(0))
        self.register_buffer("gradient_count", torch.empty(0))

    def initialize(self, points, colors, scale=0.02, density=1.0):
        """Initialize from matching [N,3] points and linear RGB in [0,1].

        Tensor inputs retain their device; array inputs use the module's device.
        Call before constructing the optimizer because parameters are replaced.
        """
        if scale <= 0 or density <= 0:
            raise ValueError("Initial scale and density must be positive")
        if not torch.is_tensor(points):
            points = torch.as_tensor(points, dtype=self._xyz.dtype, device=self._xyz.device)
        points = points.detach().to(dtype=self._xyz.dtype)
        colors = torch.as_tensor(colors, dtype=points.dtype, device=points.device)
        if points.ndim != 2 or points.shape[-1] != 3 or colors.shape != points.shape:
            raise ValueError("points and colors must both have shape [N,3]")
        count = points.shape[0]
        self._xyz = nn.Parameter(points.clone())
        self._features_dc = nn.Parameter(RGB2SH(colors).unsqueeze(1).clone())
        self._features_rest = nn.Parameter(points.new_zeros(count, (self.max_sh_degree + 1) ** 2 - 1, 3))
        self._scaling = nn.Parameter(points.new_full((count, 3), math.log(scale)))
        rotation = points.new_zeros(count, 4)
        rotation[:, 0] = 1
        self._rotation = nn.Parameter(rotation)
        self._density = nn.Parameter(inverse_softplus(points.new_full((count, 1), density)))
        self.gradient_accum = points.new_zeros(count)
        self.gradient_count = points.new_zeros(count)
        self.active_sh_degree = 0
        return self

    @property
    def xyz(self):
        return self._xyz

    @property
    def scales(self):
        return self._scaling.clamp(-12, 12).exp()

    @property
    def rotations(self):
        return F.normalize(self._rotation, dim=-1)

    @property
    def density(self):
        return F.softplus(self._density)

    @property
    def features(self):
        """SH tensor [N,(degree+1)^2,3], matching the upstream model."""
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    def rotation_matrices(self):
        return quaternion_matrix(self._rotation)

    def covariance_inverse(self):
        rotation = self.rotation_matrices()
        inverse_scale = self.scales.reciprocal()
        factor = rotation * inverse_scale.unsqueeze(-2)
        return factor @ factor.transpose(-1, -2)

    def colors(self, directions, indices=None):
        """Evaluate linear SH radiance using the ray's forward direction.

        With N selected primitives, directions may be [N,3], [R,N,3], or
        [R,1,3] for broadcasting one direction over each ray's primitives.
        For per-ray directions [R,3], the caller explicitly inserts [:,None,:].
        """
        if indices is None:
            coefficients = self.features
        else:
            coefficients = torch.cat((self._features_dc[indices], self._features_rest[indices]), dim=1)
        coefficients = coefficients.transpose(-1, -2)
        if directions.ndim == 3:
            coefficients = coefficients.unsqueeze(0)
        directions = F.normalize(directions, dim=-1)
        values = (eval_sh(self.active_sh_degree, coefficients, directions) + 0.5).clamp_min(0)
        if directions.ndim == 3:
            # Degree zero has no direction-dependent term to perform broadcasting.
            values = values.expand(directions.shape[0], coefficients.shape[1], 3)
        return values

    def oneupSHdegree(self):
        self.active_sh_degree = min(self.active_sh_degree + 1, self.max_sh_degree)

    def training_parameters(self, xyz_lr=0.00016, feature_lr=0.0025,
                            density_lr=0.05, scale_lr=0.005, rotation_lr=0.001):
        return [
            {"params": [self._xyz], "lr": xyz_lr, "name": "xyz"},
            {"params": [self._features_dc], "lr": feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": feature_lr / 20, "name": "f_rest"},
            {"params": [self._density], "lr": density_lr, "name": "density"},
            {"params": [self._scaling], "lr": scale_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": rotation_lr, "name": "rotation"},
        ]

    @torch.no_grad()
    def accumulate_statistics(self):
        """Call after backward and before zero_grad; uses world-position gradients."""
        if self._xyz.grad is None:
            return
        magnitudes = self._xyz.grad.norm(dim=-1)
        observed = torch.isfinite(magnitudes) & (magnitudes > 0)
        self.gradient_accum[observed] += magnitudes[observed]
        self.gradient_count[observed] += 1

    @torch.no_grad()
    def reset_statistics(self):
        self.gradient_accum.zero_()
        self.gradient_count.zero_()

    @torch.no_grad()
    def _replace_parameters(self, tensors, sources, retained_count, optimizer):
        """Remap optimizer moments for survivors and zero moments for new children."""
        for name in self.PARAMETER_NAMES:
            old = getattr(self, name)
            new = nn.Parameter(tensors[name], requires_grad=old.requires_grad)
            if optimizer is not None:
                for group in optimizer.param_groups:
                    group["params"] = [new if parameter is old else parameter
                                       for parameter in group["params"]]
                old_state = optimizer.state.pop(old, None)
                if old_state is not None:
                    state = {}
                    for key, value in old_state.items():
                        if torch.is_tensor(value) and value.shape == old.shape:
                            remapped = value[sources].clone()
                            remapped[retained_count:] = 0
                            state[key] = remapped
                        else:
                            state[key] = value.clone() if torch.is_tensor(value) else value
                    optimizer.state[new] = state
            setattr(self, name, new)
        for name in ("gradient_accum", "gradient_count"):
            remapped = getattr(self, name)[sources].clone()
            remapped[retained_count:] = 0
            setattr(self, name, remapped)

    @torch.no_grad()
    def densify_and_prune(self, optimizer, gradient_threshold=0.0002,
                          min_density=0.01, max_scale=None, scene_extent=1.0,
                          max_points=200000):
        """Clone small high-gradient points and replace large ones by two children.

        This adapts the upstream topology policy to world-position gradients.
        Clone density is shared; split density compensates for smaller volumes.
        Existing optimizer moments/statistics survive pruning, new rows start at
        zero. Retain the strongest finite-density primitive if all would be
        pruned, so interior regularizers and later densification stay defined.
        Call reset_statistics() to start the next accumulation window.
        max_scale is in scene units. This is a policy choice where GShell does
        not specify a ray-native densification rule.
        """
        count = self.xyz.shape[0]
        if count == 0:
            return {"before": 0, "after": 0, "cloned": 0, "split": 0, "pruned": 0}
        if scene_extent <= 0 or max_points < 1:
            raise ValueError("scene_extent and max_points must be positive")
        scales = self.scales
        density = self.density.squeeze(-1)
        keep = density >= min_density
        if max_scale is not None:
            keep &= scales.max(dim=-1).values <= max_scale
        if not keep.any():
            finite = torch.isfinite(density)
            if not finite.any():
                raise ValueError("Cannot prune a Gaussian field whose densities are all non-finite")
            strongest = torch.where(finite, density, torch.full_like(density, -float("inf"))).argmax()
            keep[strongest] = True
        valid = torch.nonzero(keep, as_tuple=False).flatten()
        if valid.numel() > max_points:
            valid = valid[density[valid].topk(max_points).indices]
            keep.zero_()
            keep[valid] = True
        score = self.gradient_accum / self.gradient_count.clamp_min(1)
        eligible = valid[(score[valid] >= gradient_threshold) & (self.gradient_count[valid] > 0)]
        capacity = max(0, max_points - valid.numel())
        if eligible.numel() > capacity:
            eligible = eligible[score[eligible].topk(capacity).indices]
        large = scales[eligible].max(dim=-1).values > self.percent_dense * scene_extent
        split = eligible[large]
        clone = eligible[~large]
        keep[split] = False
        retained = torch.nonzero(keep, as_tuple=False).flatten()
        sources = torch.cat((retained, clone, split.repeat_interleave(2)))
        tensors = {name: getattr(self, name)[sources].clone() for name in self.PARAMETER_NAMES}
        retained_count = retained.numel()
        clone_end = retained_count + clone.numel()
        if clone.numel():
            # Half the parent's density plus half in the nearby new primitive.
            cloned_parent = torch.zeros(count, dtype=torch.bool, device=self.xyz.device)
            cloned_parent[clone] = True
            clone_rows = cloned_parent[retained]
            tensors["_density"][:retained_count][clone_rows] = inverse_softplus(
                self.density[retained[clone_rows]] * 0.5
            )
            tensors["_density"][retained_count:clone_end] = inverse_softplus(self.density[clone] * 0.5)
            local_offset = torch.randn_like(scales[clone]) * scales[clone] * 0.1
            offset = (self.rotation_matrices()[clone] @ local_offset.unsqueeze(-1)).squeeze(-1)
            tensors["_xyz"][retained_count:clone_end] += offset
        if split.numel():
            split_sources = split.repeat_interleave(2)
            split_scale = scales[split_sources]
            local_offset = torch.randn_like(split_scale) * split_scale
            offset = (self.rotation_matrices()[split_sources] @ local_offset.unsqueeze(-1)).squeeze(-1)
            tensors["_xyz"][clone_end:] += offset
            shrink = 1.6
            tensors["_scaling"][clone_end:] = (split_scale / shrink).log()
            tensors["_density"][clone_end:] = inverse_softplus(self.density[split_sources] * shrink ** 3 / 2)
        self._replace_parameters(tensors, sources, retained_count, optimizer)
        return {"before": count, "after": sources.numel(), "cloned": clone.numel(),
                "split": split.numel(), "pruned": count - valid.numel()}

    def save_ply(self, path):
        """Export standard-style SH/scale/quaternion fields plus density metadata.

        density_raw is the trainable softplus argument; density is inverse-length
        extinction. opacity is only a viewer approximation derived from a central
        ray through the geometric-mean scale, so a stock 3DGS viewer cannot
        reproduce GShell's ray-dependent opacity. Checkpoints are authoritative.
        """
        from plyfile import PlyData, PlyElement

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        count = self.xyz.shape[0]
        dc = self._features_dc.detach().transpose(1, 2).reshape(count, 3)
        rest = self._features_rest.detach().transpose(1, 2).reshape(
            count, 3 * ((self.max_sh_degree + 1) ** 2 - 1)
        )
        reference_length = math.sqrt(2 * math.pi) * self.scales.prod(dim=-1, keepdim=True).pow(1 / 3)
        alpha = (-torch.expm1(-self.density * reference_length)).clamp(1e-6, 1 - 1e-6)
        opacity = torch.log(alpha) - torch.log1p(-alpha)
        attributes = torch.cat((self.xyz, torch.zeros_like(self.xyz), dc, rest,
                                opacity, self._scaling, self._rotation,
                                self.density, self._density), dim=-1).detach().cpu().numpy()
        names = ["x", "y", "z", "nx", "ny", "nz"]
        names += ["f_dc_{}".format(i) for i in range(3)]
        names += ["f_rest_{}".format(i) for i in range(rest.shape[-1])]
        names += ["opacity"] + ["scale_{}".format(i) for i in range(3)]
        names += ["rot_{}".format(i) for i in range(4)] + ["density", "density_raw"]
        vertices = np.empty(count, dtype=[(name, "f4") for name in names])
        for index, name in enumerate(names):
            vertices[name] = attributes[:, index]
        comments = ["GShell: density=softplus(density_raw), units inverse scene length",
                    "opacity is a reference-ray viewer approximation, not the learned parameter"]
        PlyData([PlyElement.describe(vertices, "vertex")], comments=comments).write(path)
