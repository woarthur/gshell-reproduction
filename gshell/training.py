"""Three sequential optimization stages; no benchmark/evaluation pipeline."""

import json
from pathlib import Path

import numpy as np
import torch

from .data import PosedDataset
from .integrator import GaussianAcceleration
from .losses import (interior_regularizers, masked_l1, patch_ssim_loss,
                     photometric_mask, regularization_points)
from .model import GShellModel, load_checkpoint, save_checkpoint


def stage_objective(model, renderer, batch, config, acceleration):
    origins, directions, target = batch["origins"], batch["directions"], batch["target"]
    if model.stage == 1:
        result = renderer.render_bootstrap(origins, directions)
        valid = photometric_mask(result)
        points = regularization_points(config, origins.device)
        rgb = masked_l1(result["rgb"], target, valid)
        eikonal = model.outer.eikonal_loss(points)
        smoothness = model.outer.smoothness_loss(points, config.smoothness_distance)
        loss = rgb + config.lambda_eikonal_outer * eikonal + config.lambda_smoothness * smoothness
        terms = {"rgb": rgb, "eikonal_outer": eikonal, "normal_smoothness": smoothness}
    elif model.stage == 2:
        # Only the environment is trainable. Discard the Stage-I proxy entirely.
        with torch.no_grad():
            result = renderer.outer_interaction(origins, directions)
        misses = ~result["hit"]
        reflections = (result["hit"] & result["valid"] & ~result["grazing"]
                       & (result["reflectance"] >= config.reflection_threshold))
        direct = model.environment(directions)
        reflected = result["reflectance"][:, None] * model.environment(result["reflected"])
        miss_loss = masked_l1(direct, target, misses)
        # RGB alone does not isolate reflection. Use only reflection-dominated
        # hits; README_GSHELL documents this approximation to the unspecified target.
        reflection_loss = masked_l1(reflected, target, reflections)
        tv = model.environment.total_variation()
        loss = miss_loss + config.lambda_reflection * reflection_loss + config.lambda_environment_tv * tv
        valid = misses | reflections
        terms = {"miss_rgb": miss_loss, "reflection_rgb": reflection_loss, "environment_tv": tv,
                 "reflection_fraction": reflections.float().mean()}
    else:
        result = renderer.render(origins, directions, acceleration=acceleration)
        valid = photometric_mask(result, batch["patch_shape"], config.silhouette_band)
        rgb = masked_l1(result["rgb"], target, valid)
        ssim = patch_ssim_loss(result["rgb"], target, valid, batch["patch_shape"])
        points = regularization_points(config, origins.device)
        terms = interior_regularizers(model, points, config)
        loss = (rgb + config.lambda_ssim * ssim + config.lambda_eikonal_inner * terms["eikonal"]
                + config.lambda_inside * terms["inside"] + config.lambda_scale * terms["scale"]
                + config.lambda_nesting * terms["nesting"])
        terms.update(rgb=rgb, ssim=ssim)
    terms["valid_fraction"] = valid.float().mean()
    return loss, terms


def _environment_resolution(config, step):
    fraction = (step - 1) / max(1, config.stage2_steps)
    if step == config.stage2_steps or fraction >= 2 / 3:
        return config.env_final_height, config.env_final_width
    exponent = 0 if fraction < 1 / 3 else 1
    return (min(config.env_final_height, config.env_height * 2 ** exponent),
            min(config.env_final_width, config.env_width * 2 ** exponent))


def train(source, output, config, resume=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if resume is None and (checkpoint_dir / "latest.pt").exists():
        raise ValueError("Output already has a checkpoint; use --resume or choose a new output directory")
    torch.manual_seed(config.seed)
    if resume:
        model, saved = load_checkpoint(resume, device=config.device)
        config = model.config
        torch.set_rng_state(saved["rng"].cpu())
        if "cuda_rng" in saved and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
        starting_stage, completed = saved["stage"], saved["step"]
    else:
        model = GShellModel(config).to(config.device)
        saved, starting_stage, completed = None, 1, 0
    config.validate()
    dataset = PosedDataset(source, config)
    config.save(output / "config.json")
    (output / "dataset.json").write_text(json.dumps({
        "source": str(Path(source).resolve()),
        "train": [view.name for view in dataset.train_views],
        "test": [view.name for view in dataset.test_views],
        "scene_center": config.scene_center, "scene_scale": config.scene_scale,
    }, indent=2), encoding="utf-8")
    stage_steps = (config.stage1_steps, config.stage2_steps, config.stage3_steps)
    acceleration = GaussianAcceleration(config.support_sigma)
    with (output / "training.jsonl").open("a", encoding="utf-8") as log:
        for stage in range(starting_stage, 4):
            continuing = saved is not None and stage == starting_stage
            if stage == 3 and not continuing:
                model.initialize_interior(dataset)
            model.set_stage(stage)
            optimizer = model.optimizer(stage)
            if continuing:
                optimizer.load_state_dict(saved["optimizer"])
            renderer = model.renderer()
            first_step = completed + 1 if continuing else 1
            total = stage_steps[stage - 1]
            print("Stage {}: {}..{}".format(stage, first_step, total))
            for step in range(first_step, total + 1):
                if stage == 2:
                    height, width = _environment_resolution(config, step)
                    if (model.environment.height, model.environment.width) != (height, width):
                        model.environment.upsample(height, width)
                        optimizer = model.optimizer(stage)
                if stage == 3:
                    if step % 1000 == 0:
                        model.gaussians.oneupSHdegree()
                    if step == 1 or step % config.acceleration_interval == 0:
                        acceleration.rebuild(model.gaussians)
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * config.final_lr_ratio ** ((step - 1) / max(1, total - 1))
                batch = dataset.sample(config.device, patches=(stage == 3))
                optimizer.zero_grad(set_to_none=True)
                loss, terms = stage_objective(model, renderer, batch, config, acceleration)
                loss.backward()
                if stage == 3:
                    model.gaussians.accumulate_statistics()
                torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters()
                                               if parameter.requires_grad), config.gradient_clip)
                optimizer.step()
                if (stage == 3 and config.densify_from <= step <= config.densify_until
                        and step % config.densify_interval == 0):
                    model.gaussians.densify_and_prune(
                        optimizer, gradient_threshold=config.densify_gradient,
                        min_density=config.prune_density, max_scale=config.prune_max_scale,
                        scene_extent=config.bound, max_points=config.max_gaussians)
                    model.gaussians.reset_statistics()
                    acceleration.rebuild(model.gaussians)
                if step == 1 or step % config.log_interval == 0 or step == total:
                    record = {"stage": stage, "step": step, "loss": float(loss.detach()),
                              "gaussians": model.gaussians.xyz.shape[0]}
                    record.update({name: float(value.detach()) for name, value in terms.items()})
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print("Stage {} {:6d}/{} loss={:.6f} valid={:.3f}".format(
                        stage, step, total, record["loss"], record["valid_fraction"]))
                if step % config.checkpoint_interval == 0 or step == total:
                    save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, step, source)
            save_checkpoint(checkpoint_dir / ("stage{}.pt".format(stage)), model, optimizer, total, source)
    model.gaussians.save_ply(str(output / "interior_gaussians.ply"))
    np.save(str(output / "environment_linear.npy"), model.environment.radiance().detach().cpu().numpy())
    return model
