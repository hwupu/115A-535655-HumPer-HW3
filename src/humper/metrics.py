"""Metric registry: one place that says what each metric is and which direction is "better"."""

from dataclasses import dataclass

import pandas as pd
import torch


@dataclass(frozen=True)
class MetricSpec:
    name: str              # column name in scores.csv
    pyiqa_name: str        # id passed to pyiqa.create_metric
    higher_is_better: bool
    family: str            # "traditional" | "learned"
    note: str


METRICS = [
    MetricSpec("psnr", "psnr", True, "traditional", "pixel MSE on RGB, capped at 80 dB"),
    MetricSpec("ssim", "ssim", True, "traditional", "luminance/contrast/structure on Y channel"),
    MetricSpec("ms_ssim", "ms_ssim", True, "traditional", "SSIM over 5 scales on Y channel"),
    MetricSpec("lpips", "lpips", False, "learned", "AlexNet feature distance, v0.1 linear weights"),
    MetricSpec("dists", "dists", False, "learned", "VGG structure + texture (mean/var) similarity"),
]
BY_NAME = {m.name: m for m in METRICS}


def device_for(spec: MetricSpec, device: torch.device) -> torch.device:
    """Traditional metrics always run on CPU: pyiqa's SSIM needs float64 (unsupported on MPS),
    they are cheap, and CPU makes them bit-identical on macOS and WSL. Learned metrics use `device`."""
    return torch.device("cpu") if spec.family == "traditional" else device


def create(spec: MetricSpec, device: torch.device) -> torch.nn.Module:
    """Build the pyiqa model and check that our direction matches pyiqa's."""
    import pyiqa  # lazy: importing pyiqa is slow, and evaluate/failures don't need it

    model = pyiqa.create_metric(spec.pyiqa_name, device=device_for(spec, device))
    assert model.lower_better == (not spec.higher_is_better), f"direction mismatch for {spec.name}"
    return model


@torch.no_grad()
def score(model: torch.nn.Module, dist: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Batch (N,3,H,W) in [0,1] -> raw scores (N,). pyiqa order is (distorted, reference)."""
    return model(dist, ref).reshape(-1).float().cpu()  # pyiqa moves inputs to the model's device


def aligned(scores: pd.DataFrame, name: str) -> pd.Series:
    """Raw score with sign flipped if needed so that higher = better quality (same as MOS)."""
    s = scores[name]
    return s if BY_NAME[name].higher_is_better else -s
