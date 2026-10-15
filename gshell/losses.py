"""Stage objectives from Eqs. 14-16 and spatially valid patch supervision."""

import torch
import torch.nn.functional as F


def masked_l1(prediction, target, mask):
    weight = mask.to(prediction.dtype)
    return ((prediction - target).abs().mean(dim=-1) * weight).sum() / weight.sum().clamp_min(1)


def photometric_mask(result, patch_shape=None, band=1):
    valid = result["valid"] & ~result["grazing"]
    if patch_shape is not None and band > 0:
        count, height, width = patch_shape
        hit = result["hit"].float().reshape(count, 1, height, width)
        kernel = 2 * band + 1
        dilated = F.max_pool2d(hit, kernel, stride=1, padding=band)
        eroded = -F.max_pool2d(-hit, kernel, stride=1, padding=band)
        boundary = (dilated - eroded).flatten() > 0
        valid = valid & ~boundary
    return valid


def patch_ssim_loss(prediction, target, valid, patch_shape):
    """SSIM on actual image patches; discard every window touching invalid rays."""
    count, height, width = patch_shape
    x = prediction.reshape(count, height, width, 3).permute(0, 3, 1, 2)
    y = target.reshape(count, height, width, 3).permute(0, 3, 1, 2)
    # Paper does not specify the patch estimator. Use the standard 11x11, sigma=1.5 window.
    axis = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    weights = torch.exp(-axis.square() / (2 * 1.5 ** 2))
    weights = weights / weights.sum()
    window = (weights[:, None] * weights[None, :]).expand(3, 1, 11, 11).contiguous()
    mean_x = F.conv2d(x, window, groups=3)
    mean_y = F.conv2d(y, window, groups=3)
    var_x = F.conv2d(x * x, window, groups=3) - mean_x.square()
    var_y = F.conv2d(y * y, window, groups=3) - mean_y.square()
    covariance = F.conv2d(x * y, window, groups=3) - mean_x * mean_y
    score = ((2 * mean_x * mean_y + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)) / (
        (mean_x.square() + mean_y.square() + 0.01 ** 2) * (var_x + var_y + 0.03 ** 2))
    invalid = (~valid).float().reshape(count, 1, height, width)
    keep = 1 - F.max_pool2d(invalid, 11, stride=1)
    return ((1 - score.mean(dim=1, keepdim=True)) * keep).sum() / keep.sum().clamp_min(1)


def regularization_points(config, device):
    return (torch.rand(config.regularization_samples, 3, device=device) * 2 - 1) * config.bound


def interior_regularizers(model, points, config):
    centers = model.gaussians.xyz
    if centers.shape[0] > config.regularization_samples:
        indices = torch.randperm(centers.shape[0], device=centers.device)[:config.regularization_samples]
        centers = centers[indices]
    inside = F.relu(model.inner(centers)).square().mean()
    scale = F.relu(model.gaussians.scales / config.scale_target - 1).square().mean()
    eikonal = model.inner.eikonal_loss(points)
    # Project nearby samples to the inner surface, then require an outer margin.
    inner_values = model.inner(points)
    normals = model.inner.normals(points, create_graph=True)
    surface = points - inner_values[:, None] * normals
    nesting = F.relu(model.outer(surface) + config.minimum_wall).square().mean()
    return {"inside": inside, "scale": scale, "eikonal": eikonal, "nesting": nesting}
