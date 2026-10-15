"""Render views with GShell's four-interface optical model."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gshell.data import PosedDataset, linear_to_srgb
from gshell.integrator import GaussianAcceleration
from gshell.model import load_checkpoint


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", help="Posed images; defaults to checkpoint dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--environment", help="Replace with linear HDR .npy array [H,W,3]")
    parser.add_argument("--interior-only", action="store_true", help="Render explicit Gaussians on straight rays")
    parser.add_argument("--shell-only", action="store_true", help="Remove interior Gaussian density")
    args = parser.parse_args()
    if args.chunk < 1 or (args.interior_only and args.shell_only):
        parser.error("Require chunk > 0 and only one component isolation option")
    model, state = load_checkpoint(args.checkpoint, args.device)
    if model.stage != 3:
        parser.error("Full-scene rendering needs a Stage III checkpoint")
    model.eval()
    if args.environment:
        radiance = torch.as_tensor(np.load(args.environment), dtype=torch.float32, device=args.device)
        if radiance.ndim != 3 or radiance.shape[-1] != 3 or (radiance < 0).any():
            parser.error("Environment must be a nonnegative [H,W,3] linear RGB array")
        model.environment.log_radiance = torch.nn.Parameter(radiance.clamp_min(1e-8).log(), requires_grad=False)
    if args.shell_only:
        with torch.no_grad():
            model.gaussians._density.fill_(-100.0)
    dataset = PosedDataset(args.source or state["source"], model.config)
    views = {"train": dataset.train_views, "test": dataset.test_views, "all": dataset.views}[args.split]
    if not views:
        parser.error("Requested split has no views")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    renderer = model.renderer()
    acceleration = GaussianAcceleration(model.config.support_sigma).rebuild(model.gaussians)
    with torch.no_grad():
        for index, view in enumerate(views):
            origins, directions = dataset.full_rays(view, args.device)
            colors, masks = [], []
            for start in range(0, origins.shape[0], args.chunk):
                origin, direction = origins[start:start + args.chunk], directions[start:start + args.chunk]
                if args.interior_only:
                    from gshell.integrator import render_gaussians
                    result = render_gaussians(model.gaussians, origin, direction, 0.0,
                                              model.config.trace_max_distance,
                                              ray_chunk=model.config.ray_chunk,
                                              gaussian_chunk=model.config.gaussian_chunk,
                                              support_sigma=model.config.support_sigma, acceleration=acceleration)
                    rgb = result["color"] + result["transmittance"][:, None] * model.environment(direction)
                    valid = torch.ones(origin.shape[0], dtype=torch.bool, device=origin.device)
                else:
                    result = renderer.render(origin, direction, acceleration)
                    rgb, valid = result["rgb"], result["valid"]
                colors.append(rgb.cpu())
                masks.append(valid.cpu())
            height, width = dataset.image(view).shape[:2]
            linear = torch.cat(colors).reshape(height, width, 3)
            preview = (linear_to_srgb(linear).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            stem = "{:04d}_{}".format(index, Path(view.name).stem)
            Image.fromarray(preview).save(output / (stem + ".png"))
            np.save(str(output / (stem + "_linear.npy")), linear.numpy())
            mask = torch.cat(masks).reshape(height, width).numpy().astype(np.uint8) * 255
            Image.fromarray(mask).save(output / (stem + "_valid.png"))
            print("Rendered {}/{}: {}".format(index + 1, len(views), view.name))


if __name__ == "__main__":
    main()
