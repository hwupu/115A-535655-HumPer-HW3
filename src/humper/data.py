"""KADID-10k loading: dmos.csv -> tidy DataFrame of (reference, distorted, MOS) pairs."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
KADID_ROOT = REPO_ROOT / "data" / "kadid10k"

# Lin et al., "KADID-10k: A Large-scale Artificially Distorted IQA Database", QoMEX 2019.
DIST_NAMES = {
    1: "gaussian_blur", 2: "lens_blur", 3: "motion_blur",
    4: "color_diffusion", 5: "color_shift", 6: "color_quantization",
    7: "color_saturation_1", 8: "color_saturation_2",
    9: "jpeg2000", 10: "jpeg",
    11: "white_noise", 12: "white_noise_color", 13: "impulse_noise",
    14: "multiplicative_noise", 15: "denoise",
    16: "brighten", 17: "darken", 18: "mean_shift",
    19: "jitter", 20: "non_eccentricity_patch", 21: "pixelate",
    22: "quantization", 23: "color_block",
    24: "high_sharpen", 25: "contrast_change",
}
DIST_GROUPS = {
    "blur": [1, 2, 3],
    "color": [4, 5, 6, 7, 8],
    "compression": [9, 10],
    "noise": [11, 12, 13, 14, 15],
    "brightness": [16, 17, 18],
    "spatial": [19, 20, 21, 22, 23],
    "sharpness_contrast": [24, 25],
}
_TYPE_TO_GROUP = {t: g for g, ts in DIST_GROUPS.items() for t in ts}


def load_pairs(root: Path = KADID_ROOT, n_refs: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Return one row per distorted image.

    Columns: id, ref_path, dist_path, ref_id, dist_type, dist_name, dist_group, level, mos, mos_var.
    Paths are POSIX strings relative to the repo root (portable across macOS/WSL).
    `mos` is KADID's `dmos` column, which despite its name is higher = better (1..5).
    `n_refs` samples whole reference images, so every distortion type/level stays balanced.
    """
    root = Path(root)
    df = pd.read_csv(root / "dmos.csv")
    # dist_img looks like I01_02_03.png -> ref 01, distortion type 02, level 03
    parts = df["dist_img"].str.extract(r"I(\d+)_(\d+)_(\d+)\.png").astype(int)
    rel = root.resolve().relative_to(REPO_ROOT) / "images"
    out = pd.DataFrame({
        "id": df["dist_img"].str.removesuffix(".png"),
        "ref_path": [(rel / f).as_posix() for f in df["ref_img"]],
        "dist_path": [(rel / f).as_posix() for f in df["dist_img"]],
        "ref_id": parts[0],
        "dist_type": parts[1],
        "level": parts[2],
        "mos": df["dmos"].astype(float),
        "mos_var": df["var"].astype(float),  # variance of the individual ratings (rater disagreement)
    })
    out["dist_name"] = out["dist_type"].map(DIST_NAMES)
    out["dist_group"] = out["dist_type"].map(_TYPE_TO_GROUP)
    if n_refs is not None:
        refs = np.random.default_rng(seed).choice(sorted(out["ref_id"].unique()), n_refs, replace=False)
        out = out[out["ref_id"].isin(refs)]
    cols = ["id", "ref_path", "dist_path", "ref_id", "dist_type", "dist_name", "dist_group", "level", "mos", "mos_var"]
    return out[cols].sort_values("id").reset_index(drop=True)


def load_image(rel_path: str) -> torch.Tensor:
    """Read an RGB image as a float tensor (3, H, W) in [0, 1]."""
    img = Image.open(REPO_ROOT / rel_path).convert("RGB")
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
