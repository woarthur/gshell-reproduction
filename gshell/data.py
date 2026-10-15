"""Posed RGB images, pinhole rays and contiguous patches without foreground masks."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.read_write_model import read_model


def srgb_to_linear(rgb):
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(rgb):
    rgb = rgb.clamp_min(0.0)
    return torch.where(rgb <= 0.0031308, 12.92 * rgb,
                       1.055 * rgb.clamp_min(0.0031308).pow(1.0 / 2.4) - 0.055)


@dataclass
class View:
    name: str
    image_path: Path
    c2w: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    split: str = "train"


class PosedDataset:
    def __init__(self, source, config):
        self.source = Path(source)
        self.config = config
        self.points = None
        self.point_colors = None
        self._images = {}
        if (self.source / "transforms_train.json").exists():
            self.views = self._read_transforms("transforms_train.json", "train")
            if (self.source / "transforms_test.json").exists():
                self.views += self._read_transforms("transforms_test.json", "test")
            elif config.holdout_every > 0:
                self._split_views()
        elif (self.source / "transforms.json").exists():
            self.views = self._read_transforms("transforms.json", "train")
            self._split_views()
        elif (self.source / "sparse").exists():
            self.views = self._read_colmap()
            self._split_views()
        else:
            raise ValueError("Expected transforms[_train].json or a COLMAP sparse directory")
        center = np.asarray(config.scene_center, dtype=np.float32)
        for view in self.views:
            view.c2w[:3, 3] = (view.c2w[:3, 3] - center) / config.scene_scale
        if self.points is not None:
            self.points = (self.points - center) / config.scene_scale
        self.train_views = [v for v in self.views if v.split == "train"]
        self.test_views = [v for v in self.views if v.split == "test"]
        if not self.train_views:
            raise ValueError("Dataset contains no training views")

    def _split_views(self):
        if self.config.holdout_every > 0 and len(self.views) > 1:
            for index, view in enumerate(self.views):
                view.split = "test" if (index + 1) % self.config.holdout_every == 0 else "train"

    def _read_transforms(self, filename, split):
        meta = json.loads((self.source / filename).read_text(encoding="utf-8"))
        views = []
        for frame in meta["frames"]:
            path = self.source / frame["file_path"]
            if not path.suffix:
                path = path.with_suffix(".png")
            with Image.open(path) as image:
                width, height = image.size
            # Intrinsics may refer to a different resolution than the stored image.
            w = frame.get("w", meta.get("w", width))
            h = frame.get("h", meta.get("h", height))
            fx = frame.get("fl_x", meta.get("fl_x"))
            fy = frame.get("fl_y", meta.get("fl_y"))
            if fx is None:
                angle_x = frame.get("camera_angle_x", meta.get("camera_angle_x"))
                if angle_x is None:
                    raise ValueError("Transforms must provide fl_x or camera_angle_x")
                fx = 0.5 * w / math.tan(0.5 * angle_x)
            if fy is None:
                angle_y = frame.get("camera_angle_y", meta.get("camera_angle_y"))
                fy = 0.5 * h / math.tan(0.5 * angle_y) if angle_y is not None else fx
            cx = frame.get("cx", meta.get("cx", w / 2.0))
            cy = frame.get("cy", meta.get("cy", h / 2.0))
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            c2w[:3, 1:3] *= -1  # Blender/OpenGL -> x right, y down, z forward.
            views.append(View(path.stem, path, c2w, fx * width / w, fy * height / h,
                              cx * width / w, cy * height / h, width, height, split))
        return views

    def _read_colmap(self):
        sparse = self.source / "sparse" / "0"
        if not sparse.exists():
            sparse = self.source / "sparse"
        extension = ".bin" if (sparse / "cameras.bin").exists() else ".txt"
        cameras, images, points = read_model(str(sparse), ext=extension)
        views = []
        for image in sorted(images.values(), key=lambda v: v.name):
            camera = cameras[image.camera_id]
            if camera.model == "PINHOLE":
                fx, fy, cx, cy = camera.params
            elif camera.model == "SIMPLE_PINHOLE":
                fx, cx, cy = camera.params
                fy = fx
            else:
                raise ValueError("Use undistorted PINHOLE/SIMPLE_PINHOLE COLMAP images")
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = image.qvec2rotmat().T
            c2w[:3, 3] = -c2w[:3, :3] @ image.tvec
            views.append(View(image.name, self.source / "images" / image.name, c2w,
                              fx, fy, cx, cy, camera.width, camera.height))
        if points:
            self.points = np.stack([point.xyz for point in points.values()]).astype(np.float32)
            self.point_colors = np.stack([point.rgb for point in points.values()]).astype(np.float32) / 255.0
            if self.config.color_space == "srgb":
                self.point_colors = srgb_to_linear(torch.from_numpy(self.point_colors)).numpy()
        return views

    def image(self, view):
        key = str(view.image_path)
        if key not in self._images:
            with Image.open(view.image_path) as source:
                image = source.convert("RGB")  # Alpha is not a shell/foreground mask.
                factor = min(1.0, self.config.image_size / max(image.size))
                if factor < 1.0:
                    image = image.resize((round(image.width * factor), round(image.height * factor)),
                                         Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
                rgb = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
            self._images[key] = srgb_to_linear(rgb) if self.config.color_space == "srgb" else rgb
        return self._images[key]

    def rays(self, view, xy, device):
        image = self.image(view)
        height, width = image.shape[:2]
        sx, sy = width / view.width, height / view.height
        xy = xy.to(device=device, dtype=torch.float32)
        directions = torch.stack(((xy[:, 0] + 0.5 - view.cx * sx) / (view.fx * sx),
                                  (xy[:, 1] + 0.5 - view.cy * sy) / (view.fy * sy),
                                  torch.ones_like(xy[:, 0])), dim=-1)
        c2w = torch.as_tensor(view.c2w, device=device)
        directions = F.normalize(directions @ c2w[:3, :3].T, dim=-1)
        return c2w[:3, 3].expand_as(directions), directions

    def sample(self, device, patches=False):
        view = self.train_views[int(torch.randint(len(self.train_views), (1,)))]
        image = self.image(view)
        height, width = image.shape[:2]
        if patches:
            size = self.config.patch_size
            if min(height, width) < size:
                raise ValueError("Image smaller than patch_size")
            yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
            corners_x = torch.randint(width - size + 1, (self.config.patches_per_batch, 1))
            corners_y = torch.randint(height - size + 1, (self.config.patches_per_batch, 1))
            x = (xx.reshape(1, -1) + corners_x).flatten()
            y = (yy.reshape(1, -1) + corners_y).flatten()
            patch_shape = (self.config.patches_per_batch, size, size)
        else:
            x = torch.randint(width, (self.config.batch_rays,))
            y = torch.randint(height, (self.config.batch_rays,))
            patch_shape = None
        xy = torch.stack((x, y), dim=-1)
        origins, directions = self.rays(view, xy, device)
        return {"origins": origins, "directions": directions,
                "target": image[y, x].to(device), "patch_shape": patch_shape, "view": view}

    def full_rays(self, view, device):
        height, width = self.image(view).shape[:2]
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        return self.rays(view, torch.stack((xx.flatten(), yy.flatten()), dim=-1), device)
