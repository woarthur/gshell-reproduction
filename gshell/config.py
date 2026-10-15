"""Paper defaults and explicitly configurable reconstruction assumptions."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class GShellConfig:
    stage1_steps: int = 20000
    stage2_steps: int = 10000
    stage3_steps: int = 40000
    seed: int = 1235
    device: str = "cuda"
    image_size: int = 800
    color_space: str = "srgb"
    holdout_every: int = 5
    scene_center: tuple = (0.0, 0.0, 0.0)
    scene_scale: float = 1.0
    bound: float = 1.5
    outer_radius: float = 0.8
    inner_inset: float = 0.05
    ior: float = 1.3
    absorption: float = 0.01
    bootstrap_absorption_length: float = 0.05
    reflection_threshold: float = 0.5
    hash_levels: int = 16
    hash_features: int = 2
    hash_base_resolution: int = 16
    hash_max_resolution: int = 2048
    log2_hash_size: int = 19
    sdf_hidden_dim: int = 64
    env_height: int = 128
    env_width: int = 256
    env_final_height: int = 512
    env_final_width: int = 1024
    sh_degree: int = 3
    initial_gaussians: int = 10000
    initial_gaussian_scale: float = 0.02
    initial_density: float = 1.0
    max_gaussians: int = 100000
    batch_rays: int = 1024
    patch_size: int = 16
    patches_per_batch: int = 4
    ray_chunk: int = 128
    gaussian_chunk: int = 2048
    support_sigma: float = 3.0
    trace_steps: int = 128
    trace_epsilon: float = 1e-4
    trace_max_distance: float = 20.0
    grazing_epsilon: float = 0.05
    silhouette_band: int = 1
    regularization_samples: int = 1024
    smoothness_distance: float = 0.01
    lambda_eikonal_outer: float = 0.1
    lambda_smoothness: float = 0.01
    lambda_reflection: float = 1.0
    lambda_environment_tv: float = 0.01
    lambda_ssim: float = 0.2
    lambda_eikonal_inner: float = 0.1
    lambda_inside: float = 0.1
    lambda_scale: float = 0.01
    # Containment supplements Eq. 16 to keep independently learned interfaces nested.
    lambda_nesting: float = 0.1
    minimum_wall: float = 0.001
    scale_target: float = 0.05
    sdf_lr: float = 0.001
    proxy_lr: float = 0.001
    material_lr: float = 0.0001
    environment_lr: float = 0.01
    gaussian_xyz_lr: float = 0.00016
    gaussian_feature_lr: float = 0.0025
    gaussian_density_lr: float = 0.01
    gaussian_scale_lr: float = 0.005
    gaussian_rotation_lr: float = 0.001
    final_lr_ratio: float = 0.1
    gradient_clip: float = 1.0
    densify_from: int = 500
    densify_until: int = 30000
    densify_interval: int = 100
    densify_gradient: float = 0.0002
    prune_density: float = 0.001
    prune_max_scale: float = 0.3
    acceleration_interval: int = 100
    checkpoint_interval: int = 2000
    log_interval: int = 100

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError("Unknown configuration keys: " + ", ".join(sorted(unknown)))
        return cls(**values)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    def validate(self):
        if not 0 < self.inner_inset < self.outer_radius < self.bound:
            raise ValueError("Require 0 < inner_inset < outer_radius < bound")
        if self.scene_scale <= 0 or self.color_space not in ("srgb", "linear"):
            raise ValueError("scene_scale must be positive; color_space is srgb or linear")
        if self.patch_size < 11 or self.patches_per_batch < 1:
            raise ValueError("SSIM requires patch_size >= 11 and at least one patch")
        if self.stage1_steps < 1 or self.stage2_steps < 1 or self.stage3_steps < 1:
            raise ValueError("Each stage must have at least one step")
        if not 1 <= self.initial_gaussians <= self.max_gaussians:
            raise ValueError("Require 1 <= initial_gaussians <= max_gaussians")
        if self.env_final_height < self.env_height or self.env_final_width < self.env_width:
            raise ValueError("Final environment resolution must be at least its starting resolution")
