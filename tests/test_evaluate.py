import numpy as np
import pandas as pd

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
