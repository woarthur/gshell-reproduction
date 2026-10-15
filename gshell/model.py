"""Scene ownership, explicit stage freezing, and portable state dictionaries."""

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .config import GShellConfig
from .environment import EnvironmentMap
from .fields import DirectionProxy, ShellMaterial, ShellSDF
from .gaussians import InteriorGaussians
from .renderer import GShellRenderer
from .tracing import TraceConfig


class GShellModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        field_args = dict(bound=config.bound, radius=config.outer_radius,
                          levels=config.hash_levels, features=config.hash_features,
                          base_resolution=config.hash_base_resolution,
                          max_resolution=config.hash_max_resolution,
                          log2_hash_size=config.log2_hash_size, hidden_dim=config.sdf_hidden_dim)
        self.outer = ShellSDF(**field_args)
        self.inner = ShellSDF(**field_args)
        self.material = ShellMaterial(config.ior, config.absorption)
        self.proxy = DirectionProxy()
        self.environment = EnvironmentMap(config.env_height, config.env_width)
        self.gaussians = InteriorGaussians(config.sh_degree)
        self.stage = 1

    def set_stage(self, stage):
        self.stage = stage
        for module in (self.outer, self.inner, self.material, self.environment, self.gaussians):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if stage > 1:
            self.proxy = None
        active = {1: (self.outer, self.material, self.proxy),
                  2: (self.environment,), 3: (self.inner, self.gaussians)}[stage]
        for module in active:
            if module is None:
                raise ValueError("A discarded bootstrap proxy cannot restart Stage I")
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def renderer(self):
        cfg = self.config
        trace = TraceConfig(bound=cfg.bound, max_steps=cfg.trace_steps, epsilon=cfg.trace_epsilon,
                            max_distance=cfg.trace_max_distance, grazing_epsilon=cfg.grazing_epsilon)
        return GShellRenderer(self.outer, self.inner, self.material, self.environment,
                             self.proxy, self.gaussians, trace, ray_chunk=cfg.ray_chunk,
                             gaussian_chunk=cfg.gaussian_chunk, support_sigma=cfg.support_sigma,
                             bootstrap_absorption_length=cfg.bootstrap_absorption_length)

    @torch.no_grad()
    def initialize_interior(self, dataset):
        cfg = self.config
        self.inner.initialize_from_outer(self.outer, cfg.inner_inset)
        device = next(self.outer.parameters()).device
        points, colors = [], []
        if dataset.points is not None:
            candidates = torch.as_tensor(dataset.points, dtype=torch.float32, device=device)
            candidate_colors = torch.as_tensor(dataset.point_colors, dtype=torch.float32, device=device)
            keep = (candidates.norm(dim=-1) < cfg.bound) & (self.inner(candidates) < -cfg.minimum_wall)
            selected = keep.nonzero(as_tuple=False).flatten()
            selected = selected[torch.randperm(selected.numel(), device=device)[:cfg.initial_gaussians]]
            points.append(candidates[selected])
            colors.append(candidate_colors[selected])
        count = sum(part.shape[0] for part in points)
        # Synthetic scenes have no reliable SfM points inside refractive shells.
        for _ in range(100):
            if count >= cfg.initial_gaussians:
                break
            candidates = (torch.rand(max(4096, 2 * (cfg.initial_gaussians - count)), 3, device=device) * 2 - 1) * cfg.bound
            candidates = candidates[self.inner(candidates) < -cfg.minimum_wall]
            candidates = candidates[:cfg.initial_gaussians - count]
            points.append(candidates)
            colors.append(torch.full_like(candidates, 0.5))
            count += candidates.shape[0]
        if count == 0:
            raise ValueError("Inner SDF encloses no initialization samples; check normalization and shell geometry")
        self.gaussians.initialize(torch.cat(points), torch.cat(colors),
                                  cfg.initial_gaussian_scale, cfg.initial_density)

    def optimizer(self, stage):
        cfg = self.config
        if stage == 1:
            groups = [{"params": self.outer.parameters(), "lr": cfg.sdf_lr, "name": "outer"},
                      {"params": self.proxy.parameters(), "lr": cfg.proxy_lr, "name": "proxy"},
                      {"params": self.material.parameters(), "lr": cfg.material_lr, "name": "material"}]
        elif stage == 2:
            groups = [{"params": self.environment.parameters(), "lr": cfg.environment_lr, "name": "environment"}]
        else:
            groups = [{"params": self.inner.parameters(), "lr": cfg.sdf_lr, "name": "inner"}]
            groups += self.gaussians.training_parameters(
                xyz_lr=cfg.gaussian_xyz_lr, feature_lr=cfg.gaussian_feature_lr,
                density_lr=cfg.gaussian_density_lr, scale_lr=cfg.gaussian_scale_lr,
                rotation_lr=cfg.gaussian_rotation_lr)
        for group in groups:
            group["initial_lr"] = group["lr"]
        return torch.optim.Adam(groups, eps=1e-8)


def save_checkpoint(path, model, optimizer, step, source):
    state = {"format": "gshell-v1", "config": asdict(model.config), "stage": model.stage,
             "step": step, "source": str(Path(source).resolve()), "model": model.state_dict(),
             "active_sh_degree": model.gaussians.active_sh_degree,
             "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def load_checkpoint(path, device=None):
    # All saved values are tensors or plain containers, compatible with weights-only loading.
    state = torch.load(path, map_location="cpu")
    if state.get("format") != "gshell-v1":
        raise ValueError("Expected a gshell-v1 checkpoint")
    cfg = GShellConfig(**state["config"])
    if device is not None:
        cfg.device = device
    model = GShellModel(cfg)
    height, width = state["model"]["environment.log_radiance"].shape[:2]
    model.environment = EnvironmentMap(height, width)
    xyz = state["model"]["gaussians._xyz"]
    model.gaussians.initialize(xyz, torch.full_like(xyz, 0.5))
    model.set_stage(state["stage"])
    model.load_state_dict(state["model"])
    model.gaussians.active_sh_degree = state["active_sh_degree"]
    model.to(cfg.device)
    return model, state
