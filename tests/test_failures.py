import numpy as np
import pandas as pd

from humper.failures import family_pairs, metric_pairs, select_diverse, vs_human


def _toy(n_refs=3, n_per_ref=6, seed=0) -> pd.DataFrame:
    """All metrics agree perfectly with MOS (distances inverted), unless a test plants a disagreement."""
    rng = np.random.default_rng(seed)
    rows = []
    for ref in range(1, n_refs + 1):
        for k in range(n_per_ref):
            mos = rng.uniform(1, 5)
            rows.append({
                "id": f"I{ref:02d}_{k:02d}", "ref_id": ref, "dist_name": f"type{k}", "dist_group": "g", "level": 1,
                "mos": mos, "mos_var": 0.5,
                "psnr": 10 + 5 * mos, "ssim": mos / 5, "ms_ssim": mos / 5, "lpips_vgg": 1 / mos, "dists": 1 / mos,
            })
    return pd.DataFrame(rows)


def test_identical_pairs_are_excluded():
    s = _toy()
    s.loc[0, "psnr"] = 80.0
    assert s.loc[0, "id"] not in set(vs_human(s)["id"])


def test_vs_human_finds_planted_optimistic_metric():
    s = _toy()
    worst = s["mos"].idxmin()
    s.loc[worst, "psnr"] = 79  # PSNR calls the worst image the best one (kept below the 80 dB identical cap)
    top = vs_human(s).iloc[0]
    assert (top["id"], top["metric"], top["direction"]) == (s.loc[worst, "id"], "psnr", "too_optimistic")


def test_metric_pairs_gap_is_symmetric_and_flags_cross_family():
    s = _toy()
    worst = s["mos"].idxmin()
    s.loc[worst, "dists"] = 1e-3  # DISTS says near-perfect, everything else says worst
    top = metric_pairs(s).iloc[0]
    assert top["id"] == s.loc[worst, "id"] and "dists" in top["pair"]
    assert top["gap"] > 0.8


def test_family_pairs_detects_conflict_within_reference():
    s = _toy(n_refs=1, n_per_ref=6)
    lo, hi = s["mos"].idxmin(), s["mos"].idxmax()
    # traditional metrics swap the best and worst image; learned metrics and MOS do not
    for m in ["psnr", "ssim", "ms_ssim"]:
        s.loc[lo, m], s.loc[hi, m] = s.loc[hi, m], s.loc[lo, m]
    top = family_pairs(s, min_mos_diff=0.1).iloc[0]
    assert {top["id_i"], top["id_j"]} == {s.loc[lo, "id"], s.loc[hi, "id"]}
    assert top["human_sides_with"] == "learned"


def test_select_diverse_respects_caps():
    c = pd.DataFrame({"gap": [0.9, 0.8, 0.7, 0.6], "dist_name": ["a", "a", "b", "c"], "ref_id": [1, 2, 1, 3]})
    picked = select_diverse(c, k=3, max_per={"dist_name": 1, "ref_id": 1})
    assert list(picked["gap"]) == [0.9, 0.6]  # 0.8 repeats type a, 0.7 repeats ref 1
