"""Figure helpers for failure cases: images, error maps, and where each metric sees damage."""

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from humper.data import load_image
from humper.metrics import METRICS

LABELS = {"psnr": "PSNR", "ssim": "SSIM", "ms_ssim": "MS-SSIM", "lpips_vgg": "LPIPS-VGG", "dists": "DISTS"}


@cache
def _lpips_spatial():
    import pyiqa

    return pyiqa.create_metric("lpips-vgg", spatial=True, device="cpu")  # small batches: CPU is fine


def maps(ref: torch.Tensor, dist: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-pixel views for one pair (tensors (3,H,W) in [0,1]). Higher = more damage in every map."""
    from pyiqa.archs.ssim_arch import ssim
    from pyiqa.utils.color_util import to_y_channel

    y = lambda t: to_y_channel(t[None], 255, "yiq").double()  # noqa: E731  same input as pyiqa SSIM
    ssim_map = ssim(y(dist), y(ref), data_range=255, get_ssim_map=True)[0, 0].numpy()
    with torch.no_grad():
        lpips_map = _lpips_spatial()(dist[None], ref[None])[0, 0].numpy()
    return {
        "|diff|": (dist - ref).abs().mean(0).numpy(),
        "1 - SSIM map": 1 - ssim_map,
        "LPIPS-VGG map": lpips_map,
    }


def _score_line(row: pd.Series, ranks: pd.Series | None) -> str:
    parts = []
    for m in METRICS:
        txt = f"{LABELS[m.name]} {row[m.name]:.3g}"
        if ranks is not None:
            txt += f" (r={ranks[m.name]:.2f})"
        parts.append(txt)
    mos = f"MOS {row['mos']:.2f}" + (f" (r={ranks['mos']:.2f})" if ranks is not None else "")
    return "  ".join(parts) + "\n" + mos


def case_figure(row: pd.Series, ranks: pd.Series | None = None, title: str = "") -> plt.Figure:
    """One distorted image: reference | distorted | |diff| | 1-SSIM map | LPIPS map, with all scores.
    `ranks` (percentile ranks incl. 'mos') adds r= values, which are what the selection used."""
    ref, dist = load_image(row["ref_path"]), load_image(row["dist_path"])
    panels = {"reference": ref.permute(1, 2, 0).numpy(), f"{row['id']} ({row['dist_name']}, L{row['level']})": dist.permute(1, 2, 0).numpy()}
    panels |= maps(ref, dist)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.1))
    for ax, (name, img) in zip(axes, panels.items()):
        ax.imshow(img, cmap=None if img.ndim == 3 else "magma", vmin=0, vmax=None if img.ndim == 3 else max(img.max(), 1e-6))
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle((title + "\n" if title else "") + _score_line(row, ranks), fontsize=8.5, y=1.02)
    fig.tight_layout()
    return fig


def pair_figure(row_i: pd.Series, row_j: pd.Series, note: str = "") -> plt.Figure:
    """Case C: two distorted versions of the same reference, with scores, to show the ranking conflict."""
    ref = load_image(row_i["ref_path"])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    for ax, img, name in [(axes[0], ref, "reference"), (axes[1], load_image(row_i["dist_path"]), "i"), (axes[2], load_image(row_j["dist_path"]), "j")]:
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis("off")
        if name == "reference":
            ax.set_title(f"reference I{row_i['ref_id']:02d}", fontsize=9)
            continue
        r = row_i if name == "i" else row_j
        ax.set_title(f"{name}: {r['id']} ({r['dist_name']}, L{r['level']})", fontsize=9)
        scores = "\n".join(f"{LABELS[m.name]} {r[m.name]:.3g}" for m in METRICS) + f"\nMOS {r['mos']:.2f}"
        ax.text(0.5, -0.03, scores, transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
    if note:
        fig.suptitle(note, fontsize=9)
    fig.tight_layout()
    return fig
