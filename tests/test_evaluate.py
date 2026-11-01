import numpy as np
import pandas as pd
import pytest

from humper.evaluate import plcc, srcc, table


def test_perfect_monotonic_relation():
    x = np.linspace(0, 1, 200)
    y = 1 + 4 / (1 + np.exp(-10 * (x - 0.5)))  # nonlinear but monotonic
    assert srcc(x, y) > 0.999
    assert plcc(x, y) > plcc(x, y, fit=None)  # logistic fit removes the nonlinearity
    assert plcc(x, y) > 0.999


def test_lower_is_better_metric_is_flipped():
    rng = np.random.default_rng(0)
    mos = rng.uniform(1, 5, 300)
    scores = pd.DataFrame({"mos": mos, "psnr": 10 * mos, "dists": 1 / mos})
    t = table(scores).set_index("metric")
    assert t.loc["psnr", "srcc"] > 0.999
    assert t.loc["dists", "srcc"] > 0.999  # positive after direction alignment


def test_monotonicity_detects_non_monotone_metric():
    from humper.evaluate import monotonicity

    rows = []
    for ref in range(4):
        for level in range(1, 6):
            mos = 5 - level + 0.1 * ref
            rows.append({"ref_id": ref, "dist_name": "blur", "level": level, "mos": mos,
                         "psnr": 50 - 5 * level, "dists": 0.1 * (level == 3)})  # DISTS: spike at level 3 only
    m = monotonicity(pd.DataFrame(rows)).set_index("metric")
    assert m.loc["psnr", "share"] == 1 and m.loc["psnr", "rho"] == pytest.approx(-1)
    assert m.loc["mos", "share"] == 1
    assert m.loc["dists", "share"] == 0  # aligned score dips at level 3 and recovers
