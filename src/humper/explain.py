"""What each metric looks at, for the failure-case figures (B-27).

Every function returns numbers that add up to the metric's own score, so a figure built from them
explains the score instead of illustrating something else:
  - luma/chroma |diff|: the YIQ split that SSIM / MS-SSIM use (they only see Y)
  - ssim_map: SSIM's local map; its mean is the SSIM score
  - msssim_map: MS-SSIM's per-scale maps upsampled and combined with the MS-SSIM weights
  - lpips_map / lpips_layers: LPIPS-VGG's spatial map; per-layer values sum to the score
  - dists_stages: DISTS per VGG stage, split into texture (mean) and structure (correlation) terms;
    they sum to the score. DISTS compares global statistics, so it has no spatial map.
"""

from functools import cache

import numpy as np
import torch
import torch.nn.functional as F

from humper.ssim_variants import C1, C2, luma255, ms_ssim_scales

STAGES = ["pixels", "relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]  # DISTS; LPIPS uses the last 5


def yiq(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) RGB [0,1] -> (3,H,W) YIQ on a 0-255 scale."""
    m = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
    return np.einsum("ij,jhw->ihw", m, img.double().numpy() * 255)


def luma_chroma_diff(ref: torch.Tensor, dist: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """|dY| (what SSIM/MS-SSIM see) and sqrt(dI^2 + dQ^2) (color change they ignore), same units."""
    d = yiq(dist) - yiq(ref)
    return np.abs(d[0]), np.hypot(d[1], d[2])


def ssim_map(ref: torch.Tensor, dist: torch.Tensor) -> np.ndarray:
    """pyiqa SSIM map (valid region); mean = SSIM score."""
    from scipy.ndimage import gaussian_filter

    x, y = luma255(ref), luma255(dist)
    f = lambda a: gaussian_filter(a, 1.5, truncate=3.5)[5:-5, 5:-5]  # noqa: E731
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx**2, f(y * y) - my**2, f(x * y) - mx * my
    cs = np.maximum((2 * sxy + C2) / (sxx + syy + C2), 0)
    return cs * (2 * mx * my + C1) / (mx**2 + my**2 + C1)


def msssim_map(ref: torch.Tensor, dist: torch.Tensor) -> np.ndarray:
    """Per-scale MS-SSIM maps upsampled to full size and combined as prod(map_k ** w_k).
    A visual summary (the score is the product of per-scale *means*, see ms_ssim_scales)."""
    scales = ms_ssim_scales(luma255(ref), luma255(dist))
    size = scales[0]["map"].shape
    out = np.ones(size)
    for d in scales:
        up = F.interpolate(torch.from_numpy(d["map"])[None, None], size=size, mode="bilinear", align_corners=False)
        out *= np.clip(up[0, 0].numpy(), 1e-6, None) ** d["weight"]
    return out


@cache
def _model(name: str, **kw):
    import pyiqa

    return pyiqa.create_metric(name, device="cpu", **kw)


def lpips_map(ref: torch.Tensor, dist: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return _model("lpips-vgg", spatial=True)(dist[None], ref[None])[0, 0].numpy()


def lpips_layers(ref: torch.Tensor, dist: torch.Tensor) -> np.ndarray:
    """LPIPS-VGG per layer (relu1_2 .. relu5_3); sums to the LPIPS score."""
    with torch.no_grad():
        _, per = _model("lpips-vgg").net(dist[None], ref[None], retPerLayer=True)
    return np.array([float(p) for p in per])


def dists_stages(ref: torch.Tensor, dist: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """DISTS = sum_k alpha_k (1 - S1_k) + beta_k (1 - S2_k) over 6 stages (alpha, beta sum to 1).
    Returns (texture, structure) per stage: S1 compares global feature means, S2 global correlations."""
    net = _model("dists").net
    with torch.no_grad():
        f0, f1 = net.forward_once(dist[None]), net.forward_once(ref[None])
        w = net.alpha.sum() + net.beta.sum()
        alpha, beta = torch.split(net.alpha / w, net.chns, dim=1), torch.split(net.beta / w, net.chns, dim=1)
        tex, struct = [], []
        for k in range(len(net.chns)):
            xm, ym = f0[k].mean([2, 3], keepdim=True), f1[k].mean([2, 3], keepdim=True)
            s1 = (2 * xm * ym + 1e-6) / (xm**2 + ym**2 + 1e-6)
            xv, yv = ((f0[k] - xm) ** 2).mean([2, 3], keepdim=True), ((f1[k] - ym) ** 2).mean([2, 3], keepdim=True)
            cov = (f0[k] * f1[k]).mean([2, 3], keepdim=True) - xm * ym
            s2 = (2 * cov + 1e-6) / (xv + yv + 1e-6)
            tex.append(float((alpha[k] * (1 - s1)).sum()))
            struct.append(float((beta[k] * (1 - s2)).sum()))
    return np.array(tex), np.array(struct)
