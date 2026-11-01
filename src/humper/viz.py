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


LEVEL_LABEL = "distortion level: L1 = mildest … L5 = strongest"
LEVEL_FOOTNOTE = ("L1–L5 = KADID-10k distortion levels (1 = mildest, 5 = strongest). Exceptions: mean_shift and contrast_change "
                  "levels go in both directions around the reference (mean_shift L3 = no change), so they are not a severity order.")


def grid_heatmap(ax: plt.Axes, g: pd.DataFrame, title: str, lim: float, highlight=(), cbar_label: str = "") -> None:
    """One distortion-type x level grid on a diverging blue-red scale centred at 0.
    `highlight` = [(dist_name, level), ...] cells outlined in black (the selected images).
    Gray cells = no data (only pixel-identical pairs there, removed by D-18)."""
    ax.set_facecolor("#c8c8c8")
    im = ax.imshow(g.to_numpy(float), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(g.shape[1]), [f"L{c}" for c in g.columns], fontsize=7)
    ax.set_yticks(range(g.shape[0]), g.index, fontsize=6.5)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel(LEVEL_LABEL, fontsize=7)
    for name, level in highlight:
        y, x = list(g.index).index(name), list(g.columns).index(level)
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, lw=2, ec="black"))
    ax.figure.colorbar(im, ax=ax, fraction=0.06, pad=0.03).set_label(cbar_label, fontsize=7)


def case_heatmaps(g_all: pd.DataFrame, g_ref: pd.DataFrame, ref_id: int, highlight, label: str, title: str) -> plt.Figure:
    """Left: mean of a signed per-image measure over all references. Right: the selected image's reference."""
    lim = float(np.nanmax(np.abs(np.r_[g_all.to_numpy(float).ravel(), g_ref.to_numpy(float).ravel()]))) or 1.0
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 6.2))
    grid_heatmap(axes[0], g_all, "all references (mean)", lim, highlight, label)
    grid_heatmap(axes[1], g_ref, f"reference I{ref_id:02d} only", lim, highlight, label)
    axes[1].set_yticklabels([])
    fig.suptitle(title, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.01, 0.005, LEVEL_FOOTNOTE + "\nGray = no data (only pixel-identical pairs in that cell).", fontsize=6.5, wrap=True)
    return fig


def srcc_grid_figure(tables: dict[str, pd.DataFrame]) -> plt.Figure:
    """SRCC per distortion type x level, one panel per metric (diverging: blue = agrees with humans, red = inverted)."""
    fig, axes = plt.subplots(1, len(tables), figsize=(2.3 * len(tables) + 1.5, 6.2), sharey=True)
    for ax, (name, g) in zip(axes, tables.items()):
        ax.set_facecolor("#c8c8c8")  # no data: the cell only held pixel-identical pairs
        im = ax.imshow(g.to_numpy(float), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(g.shape[1]), [f"L{c}" for c in g.columns], fontsize=7)
        ax.set_yticks(range(g.shape[0]), g.index, fontsize=6.5)
        ax.set_title(LABELS[name], fontsize=9)
        ax.set_xlabel("level (L1 mild → L5 strong)", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01).set_label("SRCC within cell (81 references)", fontsize=7)
    fig.text(0.01, -0.03, LEVEL_FOOTNOTE + "\nGray = no data (cell held only pixel-identical pairs).", fontsize=7)
    return fig


# Categorical palette (fixed order, validated for CVD separation; dataviz reference palette), one per KADID group.
GROUP_COLORS = dict(zip(["blur", "color", "compression", "noise", "brightness", "spatial", "sharpness_contrast"],
                        ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))


def scatter_figure(scores: pd.DataFrame) -> plt.Figure:
    """B-09: raw metric value (native direction, no flipping) vs MOS, coloured by distortion group,
    with the VQEG logistic used for PLCC drawn on top. Pixel-identical pairs are left out."""
    from humper.evaluate import fit_logistic
    from humper.metrics import aligned

    s = scores[scores["psnr"] < 80]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.1 * len(METRICS), 3.2), sharey=True)
    for ax, m in zip(axes, METRICS):
        for group, color in GROUP_COLORS.items():
            g = s[s["dist_group"] == group]
            ax.scatter(g[m.name], g["mos"], s=3, alpha=0.35, color=color, label=group, linewidths=0, rasterized=True)
        f = fit_logistic(aligned(s, m.name), s["mos"])
        if f is not None:
            order = np.argsort(s[m.name].to_numpy())
            ax.plot(s[m.name].to_numpy()[order], f[order], color="black", lw=1.5)
        ax.set_title(f"{LABELS[m.name]} ({'higher' if m.higher_is_better else 'lower'} = better)", fontsize=9)
        ax.set_xlabel(m.name, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("MOS (higher = better)", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return fig


def type_heatmap(srcc_by_type: pd.DataFrame) -> plt.Figure:
    """B-09: SRCC per distortion type (rows) x metric (columns), values printed in each cell."""
    g = srcc_by_type.rename(columns=LABELS)
    fig, ax = plt.subplots(figsize=(5.2, 7.2))
    im = ax.imshow(g.to_numpy(float), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(g.shape[1]), g.columns, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(g.shape[0]), g.index, fontsize=7.5)
    for y in range(g.shape[0]):
        for x in range(g.shape[1]):
            v = g.iat[y, x]
            ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02).set_label("SRCC within distortion type (405 images)", fontsize=7)
    fig.tight_layout()
    return fig


def msssim_scales_figure(row: pd.Series, title: str = "") -> plt.Figure:
    """MS-SSIM scale by scale. Row 1: the original (reference) image at each scale (2x2 average pooling
    between scales). Row 2: the map MS-SSIM uses at that scale for the distorted image, shown as damage
    (1 - cs; 1 - SSIM at scale 5), with its mean, weight and factor mean**weight.
    MS-SSIM = product of the factors."""
    import torch.nn.functional as F

    from humper.ssim_variants import luma255, ms_ssim_scales

    ref, dist = load_image(row["ref_path"]), load_image(row["dist_path"])
    scales = ms_ssim_scales(luma255(ref), luma255(dist))
    vmax = max(float((1 - d["map"]).max()) for d in scales) or 1.0
    fig, axes = plt.subplots(2, len(scales), figsize=(3.0 * len(scales), 5.0))
    rgb = ref[None]
    for k, d in enumerate(scales):
        h, w = d["x"].shape
        axes[0, k].imshow(rgb[0].permute(1, 2, 0).numpy(), interpolation="nearest")
        axes[0, k].set_title(f"scale {d['scale']}: {w}×{h} px" + (" (SSIM's only scale)" if k == 0 else ""), fontsize=8)
        im = axes[1, k].imshow(1 - d["map"], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        term = "cs" if k < len(scales) - 1 else "l·cs"
        axes[1, k].set_title(f"mean {term} = {d['mean']:.3f}, weight w = {d['weight']:.3f}\nfactor ({term})^w = {d['mean'] ** d['weight']:.3f}", fontsize=7.5)
        rgb = F.avg_pool2d(rgb, 2)
        for ax in axes[:, k]:
            ax.axis("off")
    axes[0, 0].text(-0.08, 0.5, "original (reference)", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    axes[1, 0].text(-0.08, 0.5, "MS-SSIM damage map (1 − cs)", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.colorbar(im, ax=axes[1, :], fraction=0.015, pad=0.01).set_label("damage (0 = identical)", fontsize=7)
    head = f"{row['id']} ({row['dist_name']}, L{row['level']}): MS-SSIM {row['ms_ssim']:.3f} = product of factors; SSIM {row['ssim']:.3f}; MOS {row['mos']:.2f}"
    fig.suptitle((title + "\n" if title else "") + head, fontsize=9.5)
    return fig


# --- B-27: case-specific explanation figures (show only what the compared metrics look at) ---------

PANEL_INFO = {
    "rgb_diff": ("|RGB diff| — what PSNR averages", "rgb"),
    "luma": ("|ΔY| luma — all SSIM/MS-SSIM see", "yiq"),
    "chroma": ("|ΔIQ| chroma — invisible to SSIM/MS-SSIM", "yiq"),
    "ssim": ("1 − SSIM map (11×11 window)", "ssim"),
    "msssim": ("1 − MS-SSIM map (5 scales, weighted)", "ssim"),
    "lpips": ("LPIPS-VGG map (per-pixel distance)", "lpips"),
}


def _panel_data(kind: str, ref: torch.Tensor, dist: torch.Tensor) -> np.ndarray:
    from humper import explain as X

    if kind == "rgb_diff":
        return (dist - ref).abs().mean(0).numpy() * 255
    if kind in ("luma", "chroma"):
        return X.luma_chroma_diff(ref, dist)[0 if kind == "luma" else 1]
    if kind == "ssim":
        return 1 - X.ssim_map(ref, dist)
    if kind == "msssim":
        return 1 - X.msssim_map(ref, dist)
    if kind == "lpips":
        return X.lpips_map(ref, dist)
    raise ValueError(kind)


def _stage_bars(ax: plt.Axes, ref: torch.Tensor, dist: torch.Tensor, which=("lpips", "dists")) -> None:
    """Per-VGG-stage contributions; each metric's bars sum to its score. DISTS split into texture
    (global feature means) and structure (global feature correlations)."""
    from humper import explain as X

    x = np.arange(len(X.STAGES))
    wbar = 0.38
    if "lpips" in which:
        lp = np.r_[np.nan, X.lpips_layers(ref, dist)]
        ax.bar(x - wbar / 2 if "dists" in which else x, lp, wbar, color="#2a78d6", label=f"LPIPS-VGG (sum {np.nansum(lp):.3f})")
    if "dists" in which:
        tex, st = X.dists_stages(ref, dist)
        xs = x + wbar / 2 if "lpips" in which else x
        ax.bar(xs, tex, wbar, color="#eb6834", label=f"DISTS texture (sum {tex.sum():.3f})")
        ax.bar(xs, st, wbar, bottom=tex, color="#1baf7a", label=f"DISTS structure (sum {st.sum():.3f})")
    ax.set_xticks(x, X.STAGES, fontsize=7, rotation=30, ha="right")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("contribution to score", fontsize=7)
    ax.set_title("VGG16 stage contributions (bars sum to the score)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(axis="y", alpha=0.2)


def explain_figure(row: pd.Series, panels: list[str], title: str = "", ranks: pd.Series | None = None) -> plt.Figure:
    """reference | distorted | the maps named in `panels` (+ "stages" = per-stage bar chart).
    Maps in the same group (e.g. luma/chroma, or SSIM/MS-SSIM) share one colour scale, so they compare."""
    ref, dist = load_image(row["ref_path"]), load_image(row["dist_path"])
    maps_ = {k: _panel_data(k, ref, dist) for k in panels if k != "stages"}
    vmax = {}
    for k, m in maps_.items():
        g = PANEL_INFO[k][1]
        vmax[g] = max(vmax.get(g, 0), float(np.percentile(m, 99.5)) or 1e-6)
    n = 2 + len(panels)
    widths = [1] * (n - 1) + [1.35 if "stages" in panels else 1]
    fig, axes = plt.subplots(1, n, figsize=(3.0 * sum(widths), 3.3), gridspec_kw={"width_ratios": widths})
    axes[0].imshow(ref.permute(1, 2, 0).numpy()); axes[0].set_title("reference", fontsize=8.5)
    axes[1].imshow(dist.permute(1, 2, 0).numpy()); axes[1].set_title(f"{row['id']} ({row['dist_name']}, L{row['level']})", fontsize=8.5)
    for ax, k in zip(axes[2:], panels):
        if k == "stages":
            _stage_bars(ax, ref, dist)
            continue
        im = ax.imshow(maps_[k], cmap="magma", vmin=0, vmax=vmax[PANEL_INFO[k][1]])
        ax.set_title(f"{PANEL_INFO[k][0]}\nmean {maps_[k].mean():.3g}", fontsize=7.5)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).ax.tick_params(labelsize=6)
    for ax in axes[: n - (1 if "stages" in panels else 0)]:
        ax.axis("off")
    fig.suptitle((title + "\n" if title else "") + _score_line(row, ranks), fontsize=8.5, y=1.04)
    fig.tight_layout()
    return fig


def explain_pair_figure(row_i: pd.Series, row_j: pd.Series, panels: list[str], note: str = "") -> plt.Figure:
    """Case C: one row per distorted image (i, j of the same reference): distorted | maps.
    Both rows share colour scales, so the two images compare directly."""
    ref = load_image(row_i["ref_path"])
    rows = [(r, load_image(r["dist_path"])) for r in (row_i, row_j)]
    data = [{k: _panel_data(k, ref, d) for k in panels} for _, d in rows]
    vmax = {}
    for dct in data:
        for k, m in dct.items():
            g = PANEL_INFO[k][1]
            vmax[g] = max(vmax.get(g, 0), float(np.percentile(m, 99.5)) or 1e-6)
    fig, axes = plt.subplots(2, 1 + len(panels), figsize=(3.0 * (1 + len(panels)), 5.0))
    for r_idx, ((r, d), dct) in enumerate(zip(rows, data)):
        axes[r_idx, 0].imshow(d.permute(1, 2, 0).numpy())
        name = "i" if r_idx == 0 else "j"
        scores = "  ".join(f"{LABELS[m.name]} {r[m.name]:.3g}" for m in METRICS)
        axes[r_idx, 0].set_title(f"{name}: {r['id']} ({r['dist_name']}, L{r['level']})\nMOS {r['mos']:.2f}", fontsize=8)
        axes[r_idx, 0].text(0, -0.04, scores, transform=axes[r_idx, 0].transAxes, fontsize=6.5, va="top")
        for c_idx, k in enumerate(panels, start=1):
            im = axes[r_idx, c_idx].imshow(dct[k], cmap="magma", vmin=0, vmax=vmax[PANEL_INFO[k][1]])
            axes[r_idx, c_idx].set_title(f"{PANEL_INFO[k][0]}\nmean {dct[k].mean():.3g}", fontsize=7.5)
            fig.colorbar(im, ax=axes[r_idx, c_idx], fraction=0.035, pad=0.02).ax.tick_params(labelsize=6)
    for ax in axes.ravel():
        ax.axis("off")
    if note:
        fig.suptitle(note, fontsize=9)
    fig.tight_layout()
    return fig
