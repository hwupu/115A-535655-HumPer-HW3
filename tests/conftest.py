import pandas as pd
import pytest
import torch

from humper.data import KADID_ROOT, REPO_ROOT, load_image
from humper.metrics import ALL_METRICS, create

SCORES_CSV = REPO_ROOT / "results" / "scores.csv"

needs_data = pytest.mark.skipif(not (KADID_ROOT / "dmos.csv").exists(), reason="KADID-10k not in data/ (see README)")
needs_scores = pytest.mark.skipif(not SCORES_CSV.exists(), reason="results/scores.csv missing")

# One pair per distortion family, plus a pixel-identical pair (I01_01_01).
SAMPLE_IDS = ["I01_01_01", "I05_03_04", "I12_10_05", "I20_11_03", "I33_19_05", "I47_21_04", "I60_25_02"]


@pytest.fixture(scope="session")
def scores() -> pd.DataFrame:
    return pd.read_csv(SCORES_CSV)


@pytest.fixture(scope="session")
def sample(scores) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """(rows, dist, ref): a small fixed batch of real KADID pairs, tensors on CPU."""
    rows = scores.set_index("id").loc[SAMPLE_IDS].reset_index()
    dist = torch.stack([load_image(p) for p in rows["dist_path"]])
    ref = torch.stack([load_image(p) for p in rows["ref_path"]])
    return rows, dist, ref


@pytest.fixture(scope="session")
def cpu_models() -> dict:
    return {m.name: create(m, torch.device("cpu")) for m in ALL_METRICS}
