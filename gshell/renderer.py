"""Four-interface shell transport and the direction-only bootstrap renderer.

Only first-surface reflection is evaluated, as in paper Eq. (12). Paths that
require internal reflection or fail to find a required boundary retain their
known radiance contribution but are excluded from the photometric mask.
"""

import torch
from torch.nn import functional as F

from .integrator import render_gaussians
from .optics import beer_lambert, reflect, refract
from .tracing import trace_sdf


def _indices(mask):
    return torch.nonzero(mask, as_tuple=False).flatten()


def _put(result, key, indices, values):
    # Out-of-place index_copy preserves all optical-path gradient connections.
    result[key] = result[key].index_copy(0, indices, values)


class GShellRenderer:
    """Compose scene modules without registering their parameters a second time.

    Input rays are compact [R, 3] tensors in normalized scene coordinates. The
    returned ``valid`` mask excludes grazing and incomplete transport paths; it
    is inferred from geometry and requires no foreground segmentation mask.

    Stage I does not specify a path length for absorption in the paper. A fixed
    nominal length is used only for that proxy stage and does not constrain the
    distance between the learned inner and outer surfaces in Stage III.
    """

    def __init__(self, outer, inner, material, environment, proxy, gaussians,
                 trace_config, ray_chunk=256, gaussian_chunk=2048,
                 support_sigma=3.0, bootstrap_absorption_length=0.05,
                 opaque_threshold=1e-4):
        self.outer = outer
        self.inner = inner
        self.material = material
        self.environment = environment
        self.proxy = proxy
        self.gaussians = gaussians
        self.trace_config = trace_config
        self.ray_chunk = ray_chunk
        self.gaussian_chunk = gaussian_chunk
        self.support_sigma = support_sigma
        self.bootstrap_absorption_length = bootstrap_absorption_length
        self.opaque_threshold = opaque_threshold

    def outer_interaction(self, origins, directions):
        """Return first-interface geometry and optics for all three stages."""
        directions = F.normalize(directions, dim=-1)
        traced = trace_sdf(self.outer, origins, directions, self.trace_config)
        count = origins.shape[0]
        result = {
            "hit": traced.hit,
            "valid": ~traced.hit | ~traced.grazing,
            "points": traced.points,
            "normals": traced.normals,
            "distance": traced.distance,
            "reflected": directions.clone(),
            "direction": directions.clone(),
            "reflectance": directions.new_zeros(count),
            "tir": torch.zeros(count, device=origins.device, dtype=torch.bool),
            "grazing": traced.grazing,
        }
        indices = _indices(traced.hit)
        if indices.numel():
            incident, normals = directions[indices], traced.normals[indices]
            transmitted, reflectance, tir = refract(
                incident, normals, 1.0, self.material.ior
            )
            _put(result, "direction", indices, transmitted)
            _put(result, "reflected", indices, reflect(incident, normals))
            _put(result, "reflectance", indices, reflectance)
            _put(result, "tir", indices, tir)
        return result

    def render_bootstrap(self, origins, directions):
        """Stage I: a shared direction-only proxy handles hits and misses."""
        interaction = self.outer_interaction(origins, directions)
        rgb = self.proxy(F.normalize(directions, dim=-1))
        indices = _indices(interaction["hit"])
        if indices.numel():
            reflectance = interaction["reflectance"][indices, None]
            reflected = self.proxy(interaction["reflected"][indices])
            transmitted = self.proxy(interaction["direction"][indices])
            length = rgb.new_full((indices.numel(),), self.bootstrap_absorption_length)
            attenuation = beer_lambert(self.material.absorption, length)
            hit_rgb = reflectance * reflected + (1 - reflectance) * attenuation * transmitted
            rgb = rgb.index_copy(0, indices, hit_rgb)
        interaction["rgb"] = rgb
        return interaction

    def _trace_next(self, sdf, points, directions):
        return trace_sdf(
            sdf, points, directions, self.trace_config,
            t_min=4.0 * self.trace_config.epsilon,
        )

    def render(self, origins, directions, acceleration=None):
        """Evaluate Eq. (12), plus shell-only rays that never enter the cavity.

        A nearly opaque Gaussian segment (Tres <= opaque_threshold) terminates
        transport before tracing the back wall. The finite cavity segment still
        needs its back-inner boundary to evaluate the analytical integral.
        """
        directions = F.normalize(directions, dim=-1)
        outer = self.outer_interaction(origins, directions)
        count = origins.shape[0]
        scalar = directions.new_zeros(count)
        vector = directions.new_zeros((count, 3))
        boolean = torch.zeros(count, dtype=torch.bool, device=origins.device)
        result = {
            "rgb": self.environment(directions),
            "valid": ~outer["hit"],
            "hit": outer["hit"],
            "grazing": outer["grazing"].clone(),
            "tir": outer["tir"].clone(),
            "front_inner_hit": boolean.clone(),
            "back_inner_hit": boolean.clone(),
            "back_outer_hit": boolean.clone(),
            "shell_only": boolean.clone(),
            "opaque": boolean.clone(),
            "front_length": scalar.clone(),
            "back_length": scalar.clone(),
            "residual_transmittance": torch.ones_like(scalar),
            "interior_color": vector.clone(),
            "point1": outer["points"],
            "point2": vector.clone(),
            "point3": vector.clone(),
            "point4": vector.clone(),
            "direction1": outer["direction"],
            "direction2": vector.clone(),
            "direction3": vector.clone(),
            "direction4": vector.clone(),
            "reflectance1": outer["reflectance"],
            "reflectance2": scalar.clone(),
            "reflectance3": scalar.clone(),
            "reflectance4": scalar.clone(),
        }
        hit_indices = _indices(outer["hit"])
        if not hit_indices.numel():
            return result
        reflection = outer["reflectance"][hit_indices, None] * self.environment(
            outer["reflected"][hit_indices]
        )
        _put(result, "rgb", hit_indices, reflection)
        # Outer-interface TIR is entirely described by first-surface reflection.
        _put(result, "valid", hit_indices,
             outer["tir"][hit_indices] & ~outer["grazing"][hit_indices])

        indices = _indices(outer["hit"] & ~outer["tir"])
        if not indices.numel():
            return result
        p1, d1 = outer["points"][indices], outer["direction"][indices]
        with torch.no_grad():
            starts_in_shell = self.inner(p1) > self.trace_config.epsilon
        indices, p1, d1 = indices[starts_in_shell], p1[starts_in_shell], d1[starts_in_shell]
        if not indices.numel():
            return result
        entry = self._trace_next(self.inner, p1, d1)
        direct_exit = self._trace_next(self.outer, p1, d1)
        # Establish which interface occurs first; a shell silhouette ray may
        # legitimately leave the outer surface without ever reaching the cavity.
        reaches_cavity = entry.hit & direct_exit.hit & (
            entry.distance < direct_exit.distance - self.trace_config.epsilon
        )
        shell_only = direct_exit.hit & (
            ~entry.hit | (direct_exit.distance < entry.distance - self.trace_config.epsilon)
        )
        if shell_only.any():
            self._render_shell_only(result, outer, indices[shell_only],
                                    d1[shell_only], direct_exit, shell_only)
        if not reaches_cavity.any():
            return result

        indices = indices[reaches_cavity]
        p2 = entry.points[reaches_cavity]
        d2, r2, tir2 = refract(
            d1[reaches_cavity], entry.normals[reaches_cavity], self.material.ior, 1.0
        )
        front_length = (p2 - outer["points"][indices]).norm(dim=-1)
        front_weight = (
            (1 - outer["reflectance"][indices, None])
            * beer_lambert(self.material.absorption, front_length)
            * (1 - r2[:, None])
        )
        _put(result, "front_inner_hit", indices, torch.ones_like(tir2))
        _put(result, "point2", indices, p2)
        _put(result, "direction2", indices, d2)
        _put(result, "front_length", indices, front_length)
        _put(result, "reflectance2", indices, r2)
        _put(result, "tir", indices, tir2)
        _put(result, "grazing", indices,
             result["grazing"][indices] | entry.grazing[reaches_cavity])
        continuing = ~tir2
        if not continuing.any():
            return result
        indices, p2, d2, front_weight = (
            indices[continuing], p2[continuing], d2[continuing], front_weight[continuing]
        )

        back_inner = self._trace_next(self.inner, p2, d2)
        # Independent fields are not guaranteed to remain nested while learning.
        # Compare the nearest boundaries: an endpoint test alone would miss an
        # early outer crossing followed by reentry into a nonconvex outer shape.
        with torch.no_grad():
            cavity_outer_exit = self._trace_next(self.outer, p2, d2)
            nested = self.outer(back_inner.points) <= self.trace_config.epsilon
            inner_is_first = cavity_outer_exit.hit & (
                back_inner.distance < cavity_outer_exit.distance - self.trace_config.epsilon
            )
        cavity_valid = back_inner.hit & nested & inner_is_first
        if not cavity_valid.any():
            return result
        indices = indices[cavity_valid]
        p2, d2, front_weight = p2[cavity_valid], d2[cavity_valid], front_weight[cavity_valid]
        p3, n3 = back_inner.points[cavity_valid], back_inner.normals[cavity_valid]
        cavity_length = back_inner.distance[cavity_valid]
        interior = render_gaussians(
            self.gaussians, p2, d2, 0.0, cavity_length,
            ray_chunk=self.ray_chunk, gaussian_chunk=self.gaussian_chunk,
            support_sigma=self.support_sigma, acceleration=acceleration,
        )
        color, residual = interior["color"], interior["transmittance"]
        _put(result, "rgb", indices, result["rgb"][indices] + front_weight * color)
        _put(result, "interior_color", indices, color)
        _put(result, "residual_transmittance", indices, residual)
        _put(result, "back_inner_hit", indices, torch.ones_like(residual, dtype=torch.bool))
        _put(result, "point3", indices, p3)
        _put(result, "grazing", indices,
             result["grazing"][indices] | back_inner.grazing[cavity_valid])
        opaque = residual <= self.opaque_threshold
        _put(result, "opaque", indices, opaque)
        _put(result, "valid", indices, opaque & ~result["grazing"][indices])
        continuing = ~opaque
        if not continuing.any():
            return result

        indices = indices[continuing]
        p3, n3, d2 = p3[continuing], n3[continuing], d2[continuing]
        residual, front_weight = residual[continuing], front_weight[continuing]
        d3, r3, tir3 = refract(d2, n3, 1.0, self.material.ior)
        _put(result, "direction3", indices, d3)
        _put(result, "reflectance3", indices, r3)
        _put(result, "tir", indices, tir3)
        continuing = ~tir3
        if not continuing.any():
            return result
        indices, p3, d3, r3 = (
            indices[continuing], p3[continuing], d3[continuing], r3[continuing]
        )
        residual, front_weight = residual[continuing], front_weight[continuing]
        back_outer = self._trace_next(self.outer, p3, d3)
        with torch.no_grad():
            inner_reentry = self._trace_next(self.inner, p3, d3)
            outer_is_first = ~inner_reentry.hit | (
                back_outer.distance < inner_reentry.distance - self.trace_config.epsilon
            )
        found = back_outer.hit & outer_is_first
        if not found.any():
            return result
        indices = indices[found]
        p4 = back_outer.points[found]
        d4, r4, tir4 = refract(d3[found], back_outer.normals[found], self.material.ior, 1.0)
        back_length = (p4 - p3[found]).norm(dim=-1)
        back_weight = (
            residual[found, None] * (1 - r3[found, None])
            * beer_lambert(self.material.absorption, back_length) * (1 - r4[:, None])
        )
        outgoing = front_weight[found] * back_weight * self.environment(d4)
        _put(result, "rgb", indices, result["rgb"][indices] + outgoing)
        _put(result, "back_outer_hit", indices, torch.ones_like(tir4))
        _put(result, "back_length", indices, back_length)
        _put(result, "point4", indices, p4)
        _put(result, "direction4", indices, d4)
        _put(result, "reflectance4", indices, r4)
        _put(result, "tir", indices, tir4)
        _put(result, "grazing", indices, result["grazing"][indices] | back_outer.grazing[found])
        _put(result, "valid", indices, ~tir4 & ~result["grazing"][indices])
        return result

    def _render_shell_only(self, result, outer, indices, d1, direct_exit, mask):
        """Two outer crossings for rays whose nearest boundary is the outer exit."""
        p4 = direct_exit.points[mask]
        d4, r4, tir4 = refract(d1, direct_exit.normals[mask], self.material.ior, 1.0)
        length = (p4 - outer["points"][indices]).norm(dim=-1)
        transmitted = (
            (1 - outer["reflectance"][indices, None])
            * beer_lambert(self.material.absorption, length)
            * (1 - r4[:, None]) * self.environment(d4)
        )
        _put(result, "rgb", indices, result["rgb"][indices] + transmitted)
        _put(result, "front_length", indices, length)
        _put(result, "point4", indices, p4)
        _put(result, "direction4", indices, d4)
        _put(result, "reflectance4", indices, r4)
        _put(result, "shell_only", indices, torch.ones_like(tir4))
        _put(result, "back_outer_hit", indices, torch.ones_like(tir4))
        _put(result, "tir", indices, tir4)
        _put(result, "grazing", indices, outer["grazing"][indices] | direct_exit.grazing[mask])
        _put(result, "valid", indices, ~tir4 & ~result["grazing"][indices])
