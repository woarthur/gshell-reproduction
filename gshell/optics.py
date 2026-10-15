"""Interface optics from GShell equations (1)-(5)."""

import torch
from torch.nn import functional as F


def incident_normal(direction, normal):
    """Orient a surface normal toward the incident medium."""
    normal = F.normalize(normal, dim=-1, eps=1e-8)
    return torch.where((direction * normal).sum(dim=-1, keepdim=True) > 0, -normal, normal)


def reflect(direction, normal):
    direction = F.normalize(direction, dim=-1, eps=1e-8)
    normal = F.normalize(normal, dim=-1, eps=1e-8)
    return F.normalize(direction - 2.0 * (direction * normal).sum(dim=-1, keepdim=True) * normal,
                       dim=-1, eps=1e-8)


def refract(direction, normal, eta_i, eta_t):
    """Return (transmitted direction, Schlick reflectance, TIR mask).

    Indices may be scalars or tensors with the ray batch shape. A TIR ray returns
    its reflection direction as a finite placeholder and reflectance exactly 1;
    callers terminate its transmission branch using the returned mask.
    """
    direction = F.normalize(direction, dim=-1, eps=1e-8)
    normal = incident_normal(direction, normal)
    eta_i = torch.as_tensor(eta_i, dtype=direction.dtype, device=direction.device)
    eta_t = torch.as_tensor(eta_t, dtype=direction.dtype, device=direction.device)
    ratio = eta_i / eta_t
    cos_i = -(direction * normal).sum(dim=-1).clamp(-1.0, 0.0)
    discriminant = 1.0 - ratio.square() * (1.0 - cos_i.square())
    tir = discriminant < 0.0
    cos_t = discriminant.clamp_min(1e-8).sqrt()
    transmitted = ratio[..., None] * direction + (ratio * cos_i - cos_t)[..., None] * normal
    transmitted = F.normalize(transmitted, dim=-1, eps=1e-8)
    transmitted = torch.where(tir[..., None], reflect(direction, normal), transmitted)
    r0 = ((eta_i - eta_t) / (eta_i + eta_t)).square()
    reflectance = r0 + (1.0 - r0) * (1.0 - cos_i).pow(5)
    reflectance = torch.where(tir, torch.ones_like(reflectance), reflectance)
    return transmitted, reflectance, tir


def beer_lambert(absorption, distance):
    """RGB attenuation for material path length distance, both differentiable."""
    return torch.exp(-distance.clamp_min(0.0)[..., None] * absorption)
