"""Small mathematical contracts; no dataset, CUDA extension, or training run.

Run explicitly with: python -m unittest discover -s tests -p test_gshell_math.py
These tests require the project's existing PyTorch and NumPy dependencies.
"""

import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from gshell.gaussians import InteriorGaussians, inverse_softplus
from gshell.integrator import GaussianAcceleration, finite_segment_integral, render_gaussians
from gshell.optics import reflect, refract
from gshell.tracing import TraceConfig, trace_sdf


DTYPE = torch.float64


def tensor(value, requires_grad=False):
    return torch.tensor(value, dtype=DTYPE, requires_grad=requires_grad)


class FrozenSphere(nn.Module):
    """An exact SDF without hash grids; its geometry is deliberately frozen."""

    def __init__(self, radius=1.0):
        super().__init__()
        self.radius = nn.Parameter(tensor(radius), requires_grad=False)

    def forward(self, points):
        return points.norm(dim=-1) - self.radius

    def gradient(self, points, create_graph=True):
        gradient = F.normalize(points, dim=-1)
        return gradient if create_graph else gradient.detach()

    def normals(self, points, create_graph=True):
        return self.gradient(points, create_graph=create_graph)


class GaussianIntegralTests(unittest.TestCase):
    def integral(self, origin, scale, near, far, support_sigma=3.0):
        return finite_segment_integral(
            tensor([origin]), tensor([[1.0, 0.0, 0.0]]), tensor([[0.0, 0.0, 0.0]]),
            torch.eye(3, dtype=DTYPE).unsqueeze(0), tensor([[scale, scale, scale]]),
            near if torch.is_tensor(near) else tensor([near]),
            far if torch.is_tensor(far) else tensor([far]), support_sigma=support_sigma
        )

    def test_integral_matches_independent_numerical_quadrature(self):
        origin = [-0.6, 0.17, 0.0]
        scale, near, far = 0.45, 0.1, 0.8
        integral, _, valid = self.integral(origin, scale, near, far)
        samples = torch.linspace(near, far, 8193, dtype=DTYPE)
        points = tensor(origin)[None, :] + samples[:, None] * tensor([1.0, 0.0, 0.0])
        density = torch.exp(-0.5 * points.square().sum(dim=-1) / scale ** 2)
        numerical = torch.trapz(density, samples)
        self.assertTrue(valid.item())
        self.assertAlmostEqual(integral.item(), numerical.item(), delta=1e-8)

    def test_finite_segment_and_ellipsoid_support_both_clip_integral(self):
        full, _, valid = self.integral([-5.0, 0.0, 0.0], 1.0, 0.0, 10.0, support_sigma=2.0)
        half, _, _ = self.integral([-5.0, 0.0, 0.0], 1.0, 5.0, 10.0, support_sigma=2.0)
        missed, _, miss_valid = self.integral([-5.0, 0.0, 0.0], 1.0, 0.0, 1.0, support_sigma=2.0)
        off_axis, _, off_axis_valid = self.integral([-5.0, 2.1, 0.0], 1.0, 0.0, 10.0, support_sigma=2.0)
        expected = math.sqrt(2 * math.pi) * math.erf(2 / math.sqrt(2))
        self.assertTrue(valid.item())
        self.assertAlmostEqual(full.item(), expected, delta=1e-12)
        self.assertAlmostEqual(half.item(), expected / 2, delta=1e-12)
        self.assertFalse(miss_valid.item())
        self.assertFalse(off_axis_valid.item())
        self.assertEqual(missed.item(), 0.0)
        self.assertEqual(off_axis.item(), 0.0)

    def test_endpoint_gradients_match_fundamental_theorem_of_calculus(self):
        origin, scale = [-0.3, 0.2, 0.0], 0.7
        near, far = tensor([0.1], requires_grad=True), tensor([0.8], requires_grad=True)
        integral, _, _ = self.integral(origin, scale, near, far)
        d_near, d_far = torch.autograd.grad(integral.sum(), (near, far))
        density_near = math.exp(-0.5 * ((origin[0] + near.item()) ** 2 + origin[1] ** 2) / scale ** 2)
        density_far = math.exp(-0.5 * ((origin[0] + far.item()) ** 2 + origin[1] ** 2) / scale ** 2)
        self.assertAlmostEqual(d_near.item(), -density_near, delta=1e-12)
        self.assertAlmostEqual(d_far.item(), density_far, delta=1e-12)

    def test_composition_sorts_across_chunks_and_preserves_nonunit_ray_endpoints(self):
        # Store the farther red point first, so chunk-local compositing is wrong.
        scale, density = 0.1, 2.0
        model = InteriorGaussians(sh_degree=0).double().initialize(
            tensor([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), scale=scale, density=density
        )
        result = render_gaussians(
            model, tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), 0.0, tensor([3.0, 1.5]),
            ray_chunk=2, gaussian_chunk=1
        )
        integral = scale * math.sqrt(2 * math.pi) * math.erf(3 / math.sqrt(2))
        transmission = math.exp(-density * integral)
        alpha = 1 - transmission
        expected_color = tensor([[transmission * alpha, alpha, 0.0]]).expand(2, 3)
        self.assertTrue(torch.allclose(result["color"], expected_color, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(result["transmittance"], tensor([transmission ** 2] * 2)))
        self.assertTrue(torch.equal(result["num_intersections"], torch.tensor([2, 2])))
        expected_depth = (alpha + 2 * transmission * alpha) / (1 - transmission ** 2)
        self.assertAlmostEqual(result["depth"][0].item(), expected_depth, delta=1e-12)

    def test_cached_bounds_refresh_after_primitive_moves_into_ray(self):
        model = InteriorGaussians(sh_degree=0).double().initialize(
            tensor([[1.0, 5.0, 0.0]]), tensor([[1.0, 1.0, 1.0]]), scale=0.1
        )
        acceleration = GaussianAcceleration().rebuild(model)
        with torch.no_grad():
            model.xyz[:, 1] = 0.0
        result = render_gaussians(model, tensor([[0.0, 0.0, 0.0]]),
                                  tensor([[1.0, 0.0, 0.0]]), 0.0, 2.0,
                                  acceleration=acceleration)
        self.assertEqual(result["num_intersections"].item(), 1)
        self.assertGreater(result["opacity"].item(), 0.0)


class GaussianTopologyTests(unittest.TestCase):
    def test_pruning_preserves_strongest_point_when_all_fall_below_threshold(self):
        model = InteriorGaussians(sh_degree=0).double().initialize(
            tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]), density=0.001
        )
        with torch.no_grad():
            model._density[1] = inverse_softplus(tensor([0.002]))
        summary = model.densify_and_prune(None, min_density=0.01, max_points=10)
        self.assertEqual(summary["after"], 1)
        self.assertEqual(summary["pruned"], 1)
        self.assertTrue(torch.equal(model.xyz, tensor([[2.0, 0.0, 0.0]])))
        self.assertTrue(torch.isfinite(model.scales.mean()).item())

    def test_clone_split_prune_preserve_adam_survivors_and_reset_child_moments(self):
        model = InteriorGaussians(sh_degree=1).double().initialize(
            tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]),
            tensor([[0.8, 0.3, 0.2], [0.4, 0.7, 0.2], [0.2, 0.5, 0.8]]), scale=0.005
        )
        with torch.no_grad():
            model._scaling[1] = math.log(0.05)
            model._density[2] = inverse_softplus(tensor([0.0001]))
        optimizer = torch.optim.Adam(model.training_parameters())
        sum(parameter.square().sum() for parameter in model.parameters()).backward()
        optimizer.step()
        optimizer.zero_grad()
        old_xyz = model.xyz
        old_moment = optimizer.state[old_xyz]["exp_avg"].clone()
        old_step = optimizer.state[old_xyz]["step"]
        with torch.no_grad():
            model.gradient_accum[:] = tensor([1.0, 1.0, 0.0])
            model.gradient_count[:] = 1
        kept_mass = (model.density[:2, 0] * model.scales[:2].prod(dim=-1)).sum().item()
        summary = model.densify_and_prune(optimizer, gradient_threshold=0.5,
                                          min_density=0.01, scene_extent=1.0, max_points=10)
        self.assertEqual(summary, {"before": 3, "after": 4, "cloned": 1, "split": 1, "pruned": 1})
        self.assertNotIn(old_xyz, optimizer.state)
        new_state = optimizer.state[model.xyz]
        self.assertTrue(torch.equal(new_state["exp_avg"][0], old_moment[0]))
        self.assertEqual(new_state["exp_avg"][1:].abs().sum().item(), 0.0)
        self.assertEqual(new_state["exp_avg_sq"][1:].abs().sum().item(), 0.0)
        self.assertEqual(float(new_state["step"]), float(old_step))
        self.assertTrue(torch.equal(model.gradient_accum, tensor([1.0, 0.0, 0.0, 0.0])))
        mass = (model.density[:, 0] * model.scales.prod(dim=-1)).sum().item()
        self.assertAlmostEqual(mass, kept_mass, delta=1e-12)
        # A subsequent optimizer update must use the replacement tensors and shapes.
        sum(parameter.square().sum() for parameter in model.parameters()).backward()
        optimizer.step()
        for group in optimizer.param_groups:
            parameter = group["params"][0]
            self.assertEqual(parameter.shape[0], 4)
            self.assertEqual(optimizer.state[parameter]["exp_avg"].shape, parameter.shape)


class InterfaceOpticsTests(unittest.TestCase):
    def test_normal_incidence_preserves_direction_and_fresnel_reflectance(self):
        direction = tensor([[0.0, 0.0, -1.0]])
        transmitted, reflectance, tir = refract(direction, tensor([[0.0, 0.0, 1.0]]), 1.0, 1.5)
        self.assertTrue(torch.allclose(transmitted, direction, atol=1e-12, rtol=0))
        self.assertAlmostEqual(reflectance.item(), 0.04, delta=1e-12)
        self.assertFalse(tir.item())

    def test_total_internal_reflection_disables_transmission(self):
        direction = tensor([[math.sqrt(3) / 2, 0.0, -0.5]])
        normal = tensor([[0.0, 0.0, 1.0]])
        transmitted, reflectance, tir = refract(direction, normal, 1.5, 1.0)
        self.assertTrue(tir.item())
        self.assertEqual(reflectance.item(), 1.0)
        self.assertTrue(torch.allclose(transmitted, reflect(direction, normal)))
        self.assertTrue(torch.isfinite(transmitted).all().item())


class ImplicitIntersectionTests(unittest.TestCase):
    def test_frozen_sphere_keeps_origin_and_normal_gradients(self):
        sphere = FrozenSphere()
        origins = tensor([[-2.0, 0.3, 0.0]], requires_grad=True)
        config = TraceConfig(bound=1.5, epsilon=1e-9, min_step=1e-10, max_steps=256)
        hit = trace_sdf(sphere, origins, tensor([[1.0, 0.0, 0.0]]), config)
        self.assertTrue(hit.hit.item())
        root = math.sqrt(1 - 0.3 ** 2)
        self.assertAlmostEqual(hit.distance.item(), 2 - root, delta=2e-9)
        distance_gradient = torch.autograd.grad(hit.distance.sum(), origins, retain_graph=True)[0]
        normal_gradient = torch.autograd.grad(hit.normals[:, 0].sum(), origins)[0]
        expected_distance_gradient = tensor([[-1.0, 0.3 / root, 0.0]])
        expected_normal_gradient = tensor([[0.0, 0.3 / root, 0.0]])
        self.assertTrue(torch.allclose(distance_gradient, expected_distance_gradient,
                                       atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(normal_gradient, expected_normal_gradient,
                                       atol=1e-7, rtol=1e-7))
        self.assertIsNone(sphere.radius.grad)


if __name__ == "__main__":
    unittest.main()
