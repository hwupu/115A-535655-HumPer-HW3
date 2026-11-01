"""SSIM variants for B-22 / D-20: does pyiqa's choice of SSIM explain our low SRCC (0.62)?

pyiqa SSIM (our main score) = Wang et al. 2004 on YIQ luma, *no* downsampling, contrast-structure
term clamped at 0 (ReLU). The variants change one thing at a time:
  - clamp_cs=False: textbook formula (matches skimage, tests/test_sanity.py)
  - downsample=True: Wang's reference MATLAB code first averages f x f blocks and subsamples by
    f = max(1, round(min(H, W) / 256)), i.e. f = 2 for KADID's 512 x 384 images. This mimics a
    typical viewing distance, so fine pixel-level detail counts less.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, uniform_filter

C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2


def luma255(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) RGB in [0,1] -> YIQ Y channel on [0,255], rounded like pyiqa (float64)."""
    r, g, b = (img.double() * 255).numpy()
    return np.round(0.299 * r + 0.587 * g + 0.114 * b)


def downsample(y: np.ndarray) -> np.ndarray:
    f = max(1, round(min(y.shape) / 256))
    return y if f == 1 else uniform_filter(y, f, mode="nearest", origin=-(f // 2) if f % 2 == 0 else 0)[::f, ::f]


def ssim(x: np.ndarray, y: np.ndarray, clamp_cs: bool = True, down: bool = False) -> float:
    """Wang et al. 2004: 11x11 Gaussian (sigma 1.5), K = (0.01, 0.03), L = 255, 'valid' region."""
    if down:
        x, y = downsample(x), downsample(y)
    f = lambda a: gaussian_filter(a, 1.5, truncate=3.5)[5:-5, 5:-5]  # noqa: E731
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx**2, f(y * y) - my**2, f(x * y) - mx * my
    cs = (2 * sxy + C2) / (sxx + syy + C2)
    if clamp_cs:
        cs = np.maximum(cs, 0)
    return float((((2 * mx * my + C1) / (mx**2 + my**2 + C1)) * cs).mean())


VARIANTS = {
    "ssim_noclamp": dict(clamp_cs=False, down=False),
    "ssim_ds": dict(clamp_cs=True, down=True),
    "ssim_ds_noclamp": dict(clamp_cs=False, down=True),
}
