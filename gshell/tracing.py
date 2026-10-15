"""Sphere tracing with an implicit-root derivative at converged hits."""

from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class TraceConfig:
    bound: float = 1.5
    max_steps: int = 128
    epsilon: float = 1e-4
    max_distance: float = 100.0
    min_step: float = 1e-5
    step_scale: float = 0.8
    grazing_epsilon: float = 0.03


@dataclass
class RayHit:
    points: torch.Tensor
    normals: torch.Tensor
    distance: torch.Tensor
    hit: torch.Tensor
    grazing: torch.Tensor


def _ray_parameter(value, rays):
    return torch.as_tensor(value, dtype=rays.dtype, device=rays.device).expand(rays.shape[0])


def trace_sdf(sdf, origins, directions, config, t_min=0.0, t_max=None):
    """Trace [R,3] rays through a sphere that encloses the neural shell.

    Absolute SDF steps find both entries and exits. Iterations and visibility are
    detached; the final hit receives dt/dtheta = -df/dtheta / (grad(f) dot d).
    No gradient is propagated through hit/miss decisions.
    """
    if origins.ndim != 2 or origins.shape[-1] != 3 or origins.shape != directions.shape:
        raise ValueError("Ray origins and directions must both have shape [R, 3]")
    differentiable = torch.is_grad_enabled()
    directions = F.normalize(directions, dim=-1, eps=1e-8)
    with torch.no_grad():
        start = _ray_parameter(t_min, origins).clamp_min(config.epsilon * 4.0)
        stop = _ray_parameter(config.max_distance if t_max is None else t_max, origins)
        stop = stop.clamp_max(config.max_distance)
        projection = (origins * directions).sum(dim=-1)
        discriminant = projection.square() - origins.square().sum(dim=-1) + config.bound ** 2
        root = discriminant.clamp_min(0.0).sqrt()
        distance = torch.maximum(start, -projection - root)
        stop = torch.minimum(stop, -projection + root)
        active = (discriminant >= 0.0) & (distance <= stop)
        hit = torch.zeros_like(active)
        departed = torch.ones_like(active)
        initial_indices = active.nonzero(as_tuple=False).squeeze(-1)
        if initial_indices.numel():
            origin_values = sdf(origins[initial_indices])
            departed[initial_indices] = origin_values.abs() > config.epsilon * 4.0
        previous_distance = distance.clone()
        previous_value = torch.zeros_like(distance)
        had_previous = torch.zeros_like(active)

        for _ in range(config.max_steps):
            indices = active.nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                break
            current = distance[indices]
            values = sdf(origins[indices] + current[:, None] * directions[indices])
            # Near a grazing interface, 4*epsilon of forward travel can still be
            # within the same zero-set tolerance. Leave that band before re-hit.
            departed[indices] |= values.abs() > config.epsilon * 2.0
            converged = (values.abs() <= config.epsilon) & departed[indices]
            crossed = had_previous[indices] & (values * previous_value[indices] < 0.0) & ~converged
            if crossed.any():
                # A learned field can violate the exact SDF bound. Refine any
                # detected sign change without retaining the iteration graph.
                crossing_indices = indices[crossed]
                lo = previous_distance[crossing_indices].clone()
                hi = current[crossed].clone()
                lo_value = previous_value[crossing_indices].clone()
                for _ in range(16):
                    mid = (lo + hi) * 0.5
                    mid_value = sdf(origins[crossing_indices] + mid[:, None] * directions[crossing_indices])
                    same_side = lo_value * mid_value > 0.0
                    lo = torch.where(same_side, mid, lo)
                    lo_value = torch.where(same_side, mid_value, lo_value)
                    hi = torch.where(same_side, hi, mid)
                refined = (lo + hi) * 0.5
                refined_value = sdf(origins[crossing_indices] + refined[:, None] * directions[crossing_indices])
                distance[crossing_indices] = refined
                # A root is accepted only once its residual meets tolerance.
                crossing_hit = refined_value.abs() <= config.epsilon
                hit[crossing_indices] = crossing_hit
                active[crossing_indices] = False
            hit[indices[converged]] = True
            active[indices[converged]] = False
            continuing = ~converged & ~crossed
            advancing = indices[continuing]
            previous_distance[advancing] = current[continuing]
            previous_value[advancing] = values[continuing]
            had_previous[advancing] = departed[advancing]
            step = (values[continuing].abs() * config.step_scale).clamp_min(config.min_step)
            distance[advancing] = current[continuing] + step
            active[advancing] = distance[advancing] <= stop[advancing]
        # Misses carry finite placeholders; consumers select using the hit mask.
        distance = torch.where(hit, distance, torch.zeros_like(distance))

    points = origins + distance[:, None] * directions
    normals = torch.zeros_like(points)
    grazing = torch.zeros_like(hit)
    indices = hit.nonzero(as_tuple=False).squeeze(-1)
    if indices.numel():
        hit_points = points[indices]
        geometric_gradient = sdf.gradient(hit_points, create_graph=False)
        unit_normals = F.normalize(geometric_gradient, dim=-1, eps=1e-8)
        cosine = (unit_normals * directions[indices].detach()).sum(dim=-1)
        grazing[indices] = cosine.abs() < config.grazing_epsilon
        if differentiable:
            denominator = (geometric_gradient * directions[indices].detach()).sum(dim=-1)
            minimum = (config.grazing_epsilon * geometric_gradient.norm(dim=-1)).clamp_min(1e-8)
            sign = torch.where(denominator < 0.0, -torch.ones_like(denominator), torch.ones_like(denominator))
            denominator = sign * torch.maximum(denominator.abs(), minimum)
            value = sdf(hit_points)
            correction = -(value - value.detach()) / denominator.detach()
            distance = distance.index_add(0, indices, correction)
            points = origins + distance[:, None] * directions
            normals = normals.index_copy(0, indices, sdf.normals(points[indices], create_graph=True))
        else:
            normals[indices] = unit_normals
    return RayHit(points=points, normals=normals, distance=distance,
                  hit=hit, grazing=grazing)
