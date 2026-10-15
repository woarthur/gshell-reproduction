"""Finite-segment ray/Gaussian integration and front-to-back compositing.

Implements GShell Eqs. 6--9 and 11 in PyTorch. This readable reference backend
uses tensor AABB culling without a custom CUDA BVH. Work is chunked across rays and
Gaussians; global per-ray sorting still takes O(ray_chunk * candidate_count)
memory because primitives from different Gaussian chunks can overlap in depth.
"""

import math

import torch


class GaussianAcceleration:
    """Conservative covariance AABB broad phase using standard tensor operations.

    rebuild() stores the support bounds. candidates() refreshes them whenever
    the position, scale or rotation tensor changes, including optimizer updates
    and densification. Thus callers may schedule periodic rebuilds without
    silently using stale bounds to cull moved or expanded primitives.
    """

    def __init__(self, support_sigma=3.0):
        if support_sigma <= 0:
            raise ValueError("support_sigma must be positive")
        self.support_sigma = support_sigma
        self.lower = None
        self.upper = None
        self._signature = None

    def _model_signature(self, model):
        tensors = (model._xyz, model._scaling, model._rotation)
        return tuple((id(value), value._version, tuple(value.shape), value.device, value.dtype)
                     for value in tensors) + (self.support_sigma,)

    @torch.no_grad()
    def rebuild(self, model):
        factor = model.rotation_matrices() * model.scales.unsqueeze(-2)
        half_extent = self.support_sigma * factor.square().sum(dim=-1).sqrt()
        # Small outward padding keeps the AABB conservative at floating point edges.
        padding = torch.finfo(model.xyz.dtype).eps * 16 * (1 + model.xyz.abs() + half_extent)
        self.lower = model.xyz.detach() - half_extent - padding
        self.upper = model.xyz.detach() + half_extent + padding
        self._signature = self._model_signature(model)
        return self

    @torch.no_grad()
    def refresh(self, model):
        if self._signature != self._model_signature(model):
            self.rebuild(model)

    @torch.no_grad()
    def candidates(self, model, origins, directions, t_min, t_max, start, end):
        """Return [R,end-start] conservative finite-segment/AABB intersections."""
        self.refresh(model)
        lower = self.lower[start:end].unsqueeze(0) - origins[:, None, :]
        upper = self.upper[start:end].unsqueeze(0) - origins[:, None, :]
        direction = directions[:, None, :]
        parallel = direction == 0
        safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
        first = lower / safe_direction
        second = upper / safe_direction
        near = torch.minimum(first, second)
        far = torch.maximum(first, second)
        near = torch.where(parallel, torch.full_like(near, -float("inf")), near)
        far = torch.where(parallel, torch.full_like(far, float("inf")), far)
        outside_parallel = (parallel & ((lower > 0) | (upper < 0))).any(dim=-1)
        enter = torch.maximum(near.max(dim=-1).values, t_min[:, None])
        leave = torch.minimum(far.min(dim=-1).values, t_max[:, None])
        return (enter <= leave) & ~outside_parallel & (t_max[:, None] > t_min[:, None])


def _ray_bound(value, origins):
    value = torch.as_tensor(value, dtype=origins.dtype, device=origins.device)
    if value.ndim == 0:
        return value.expand(origins.shape[0])
    return value.reshape(origins.shape[0])


def _erf_difference(lower, upper):
    """Avoid subtracting two nearly-one erf values on the same side of zero."""
    positive = torch.erfc(lower) - torch.erfc(upper)
    negative = torch.erfc(-upper) - torch.erfc(-lower)
    crossing = torch.erf(upper) - torch.erf(lower)
    return torch.where(lower > 0, positive, torch.where(upper < 0, negative, crossing)).clamp_min(0)


def finite_segment_integral(origins, directions, means, rotations, scales,
                            t_min, t_max, support_sigma=3.0):
    """Analytic unit-direction integral over the segment AND support ellipsoid.

    Rays: origins/directions [R,3], metric bounds [R]. Primitives: means [N,3],
    rotation matrices [N,3,3], scales [N,3]. Returns (J, depth, valid), all [R,N].
    depth is the Mahalanobis closest-approach point clamped to the clipped
    segment; it supplies the explicit per-ray ordering for Eq. 11.
    """
    offset = origins[:, None, :] - means[None, :, :]
    local_origin = torch.einsum("rnj,njk->rnk", offset, rotations) / scales[None, :, :]
    local_direction = torch.einsum("rj,njk->rnk", directions, rotations) / scales[None, :, :]
    a = local_direction.square().sum(dim=-1).clamp_min(torch.finfo(origins.dtype).tiny)
    b = (local_origin * local_direction).sum(dim=-1)
    center = -b / a
    # Equivalent to c-b*b/a, without subtracting large almost equal values.
    perpendicular = local_origin + center.unsqueeze(-1) * local_direction
    q_min = perpendicular.square().sum(dim=-1)
    discriminant = support_sigma ** 2 - q_min
    # Tangent/outside supports are masked; never take sqrt(0) on those branches.
    positive_discriminant = torch.where(discriminant > 0, discriminant,
                                        torch.ones_like(discriminant))
    radius = (positive_discriminant / a).sqrt()
    lower = torch.maximum(t_min[:, None], center - radius)
    upper = torch.minimum(t_max[:, None], center + radius)
    valid = (discriminant > 0) & (upper > lower)
    # Empty intervals still evaluate finite expressions before masking gradients.
    upper = torch.maximum(upper, lower)
    factor = (a / 2).sqrt()
    u0 = factor * (lower - center)
    u1 = factor * (upper - center)
    integral = (math.pi / (2 * a)).sqrt() * (-0.5 * q_min).exp() * _erf_difference(u0, u1)
    integral = torch.where(valid, integral, torch.zeros_like(integral))
    depth = torch.minimum(torch.maximum(center, lower), upper)
    return integral, depth, valid


def render_gaussians(model, origins, directions, t_min, t_max, ray_chunk=256,
                     gaussian_chunk=2048, support_sigma=3.0, acceleration=None):
    """Composite the interior on arbitrary refracted finite ray segments.

    Bounds parameterize the supplied ray origins+t*directions. Nonunit directions
    are normalized, with bounds scaled by their norms to preserve endpoints and
    the physical path length in Eq. 8. Returned depth is physical distance from
    the ray origin, conditioned on a Gaussian interaction (zero when empty).

    Returns color [R,3], transmittance/opacity/depth [R], and integer
    num_intersections [R]. No environment is included in color. The caller uses
    residual transmittance for the back-wall/environment term in Eq. 12.
    Sorting and support membership are piecewise differentiable, as in splatting.
    """
    if origins.ndim != 2 or origins.shape[-1] != 3 or directions.shape != origins.shape:
        raise ValueError("origins and directions must have shape [R,3]")
    if ray_chunk < 1 or gaussian_chunk < 1 or support_sigma <= 0:
        raise ValueError("Chunk sizes and support_sigma must be positive")
    lengths = directions.norm(dim=-1)
    if (lengths <= 0).any():
        raise ValueError("Ray directions must be nonzero")
    unit_directions = directions / lengths[:, None]
    near = _ray_bound(t_min, origins) * lengths
    far = _ray_bound(t_max, origins) * lengths
    if acceleration is None:
        acceleration = GaussianAcceleration(support_sigma)
    elif acceleration.support_sigma != support_sigma:
        # The bound must enclose exactly the support used by the integral.
        acceleration.support_sigma = support_sigma
    acceleration.refresh(model)
    rotations = model.rotation_matrices()
    scales = model.scales
    densities = model.density.squeeze(-1)
    ray_count = origins.shape[0]
    point_count = model.xyz.shape[0]
    outputs = {"color": [], "transmittance": [], "opacity": [], "depth": [],
               "num_intersections": []}
    for ray_start in range(0, ray_count, ray_chunk):
        ray_end = min(ray_start + ray_chunk, ray_count)
        origin = origins[ray_start:ray_end]
        direction = unit_directions[ray_start:ray_end]
        lower = near[ray_start:ray_end]
        upper = far[ray_start:ray_end]
        taus, depths, colors, hits = [], [], [], []
        for point_start in range(0, point_count, gaussian_chunk):
            point_end = min(point_start + gaussian_chunk, point_count)
            candidates = acceleration.candidates(model, origin, direction, lower, upper,
                                                 point_start, point_end)
            selected = torch.nonzero(candidates.any(dim=0), as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            indices = selected + point_start
            integral, depth, valid = finite_segment_integral(
                origin, direction, model.xyz[indices], rotations[indices], scales[indices],
                lower, upper, support_sigma
            )
            valid = valid & candidates[:, selected]
            tau = torch.where(valid, integral * densities[indices][None, :],
                              torch.zeros_like(integral))
            taus.append(tau)
            depths.append(torch.where(valid, depth, torch.full_like(depth, float("inf"))))
            colors.append(model.colors(direction[:, None, :], indices=indices))
            hits.append(valid)
        count = ray_end - ray_start
        if not taus:
            # Retain a zero gradient path for empty scenes/rays during training.
            zero = (origin.sum() + direction.sum() + model.xyz.sum() + model._density.sum()) * 0
            color = origin.new_zeros(count, 3) + zero
            transmittance = origin.new_ones(count) + zero
            opacity = origin.new_zeros(count) + zero
            depth = origin.new_zeros(count) + zero
            intersections = torch.zeros(count, dtype=torch.long, device=origin.device)
        else:
            tau = torch.cat(taus, dim=1)
            depth_values = torch.cat(depths, dim=1)
            color_values = torch.cat(colors, dim=1)
            order = depth_values.argsort(dim=1)
            tau = tau.gather(1, order)
            depth_values = depth_values.gather(1, order)
            color_values = color_values.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))
            # Use optical-depth prefix sums instead of cumprod(1-alpha), so small
            # opacities and long sequences do not lose precision unnecessarily.
            cumulative = torch.cumsum(tau, dim=1)
            prefix = torch.cat((torch.zeros_like(tau[:, :1]), cumulative[:, :-1]), dim=1)
            weights = (-prefix).exp() * (-torch.expm1(-tau))
            color = (weights.unsqueeze(-1) * color_values).sum(dim=1)
            transmittance = (-cumulative[:, -1]).exp()
            opacity = -torch.expm1(-cumulative[:, -1])
            finite_depth = torch.where(torch.isfinite(depth_values), depth_values,
                                       torch.zeros_like(depth_values))
            depth = (weights * finite_depth).sum(dim=1) / opacity.clamp_min(1e-10)
            intersections = torch.cat(hits, dim=1).sum(dim=1)
        for name, value in (("color", color), ("transmittance", transmittance),
                            ("opacity", opacity), ("depth", depth),
                            ("num_intersections", intersections)):
            outputs[name].append(value)
    if ray_count == 0:
        return {"color": origins.new_empty(0, 3), "transmittance": origins.new_empty(0),
                "opacity": origins.new_empty(0), "depth": origins.new_empty(0),
                "num_intersections": torch.empty(0, dtype=torch.long, device=origins.device)}
    return {name: torch.cat(values, dim=0) for name, values in outputs.items()}
