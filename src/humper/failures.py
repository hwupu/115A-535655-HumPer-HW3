"""Failure-case mining (D-24). Quantitative first, so the visual inspection can't cherry-pick.

All comparisons use direction-aligned percentile ranks in [0, 1] (higher = better), never raw values,
because the metrics live on different scales (dB, [0,1] similarities, feature distances).

Three candidate lists, matching the HW3 spec:
  A. metric_pairs(): two metrics disagree on the same image        gap = |r_a - r_b|
  B. vs_human():     a metric disagrees with MOS                    gap = r_m - r_mos (signed)
  C. family_pairs(): traditional vs learned *ranking* of two images of the same reference
"""

from itertools import combinations

import numpy as np
import pandas as pd

from humper.metrics import METRICS, aligned

TRADITIONAL = [m.name for m in METRICS if m.family == "traditional"]
LEARNED = [m.name for m in METRICS if m.family == "learned"]
INFO = ["id", "ref_id", "dist_name", "dist_group", "level", "mos", "mos_var"]


def eval_set(scores: pd.DataFrame) -> pd.DataFrame:
    """Drop the 320 pixel-identical pairs (D-18): every metric scores them perfect, so they can't disagree."""
    return scores[scores["psnr"] < 80].reset_index(drop=True)


def ranks(scores: pd.DataFrame, metrics: list[str] | None = None, by: str | None = None) -> pd.DataFrame:
    """Percentile rank of MOS and each direction-aligned metric; `by="ref_id"` ranks within each reference."""
    names = metrics or [m.name for m in METRICS]
    vals = pd.DataFrame({n: aligned(scores, n) for n in names})
    vals["mos"] = scores["mos"]
    if by is None:
        return vals.rank(pct=True)
    return vals.groupby(scores[by]).rank(pct=True)


def reliable(scores: pd.DataFrame, var_quantile: float = 0.9) -> pd.Series:
    """True where raters agreed reasonably (mos_var below the given quantile). We don't build a case
    on a MOS that humans themselves disagree about."""
    return scores["mos_var"] <= scores["mos_var"].quantile(var_quantile)


def metric_pairs(scores: pd.DataFrame, var_quantile: float = 0.9) -> pd.DataFrame:
    """Case A: one row per (image, metric pair), sorted by rank gap."""
    s = eval_set(scores)
    r = ranks(s)
    keep = reliable(s, var_quantile)
    rows = []
    for a, b in combinations([m.name for m in METRICS], 2):
        rows.append(s.loc[keep, INFO].assign(
            pair=f"{a}~{b}", metric_a=a, metric_b=b,
            rank_a=r.loc[keep, a], rank_b=r.loc[keep, b], rank_mos=r.loc[keep, "mos"],
            cross_family=(a in TRADITIONAL) != (b in TRADITIONAL),
            gap=(r.loc[keep, a] - r.loc[keep, b]).abs(),
        ))
    return pd.concat(rows, ignore_index=True).sort_values("gap", ascending=False, ignore_index=True)


def vs_human(scores: pd.DataFrame, var_quantile: float = 0.9) -> pd.DataFrame:
    """Case B: one row per (image, metric). gap > 0: the metric is too optimistic (says good, humans say bad);
    gap < 0: too pessimistic (penalises something humans hardly notice). Sorted by |gap|."""
    s = eval_set(scores)
    r = ranks(s)
    keep = reliable(s, var_quantile)
    rows = [
        s.loc[keep, INFO].assign(metric=m.name, rank_metric=r.loc[keep, m.name], rank_mos=r.loc[keep, "mos"],
                                 gap=r.loc[keep, m.name] - r.loc[keep, "mos"])
        for m in METRICS
    ]
    out = pd.concat(rows, ignore_index=True)
    out["direction"] = np.where(out["gap"] > 0, "too_optimistic", "too_pessimistic")
    return out.reindex(out["gap"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def family_pairs(scores: pd.DataFrame, var_quantile: float = 0.9, min_mos_diff: float = 0.5) -> pd.DataFrame:
    """Case C: pairs (i, j) of distorted versions of the *same* reference where the traditional family
    ranks i above j and the learned family ranks j above i. Family score = mean within-reference rank of
    its metrics. strength = min(|d_trad|, |d_learned|), so both families must be confident.
    `human_sides_with` uses MOS; pairs with |MOS_i - MOS_j| < min_mos_diff count as "tie"."""
    s = eval_set(scores)
    r = ranks(s, by="ref_id")
    s = s.assign(trad=r[TRADITIONAL].mean(axis=1), learned=r[LEARNED].mean(axis=1), ok=reliable(s, var_quantile))
    out = []
    for ref_id, g in s[s["ok"]].groupby("ref_id"):
        i, j = np.triu_indices(len(g), k=1)
        gi, gj = g.iloc[i].reset_index(drop=True), g.iloc[j].reset_index(drop=True)
        d_trad, d_learned = gi["trad"] - gj["trad"], gi["learned"] - gj["learned"]
        opposite = np.sign(d_trad) * np.sign(d_learned) < 0
        if not opposite.any():
            continue
        d_mos = gi["mos"] - gj["mos"]
        side = np.where(d_mos.abs() < min_mos_diff, "tie",
                        np.where(np.sign(d_mos) == np.sign(d_trad), "traditional", "learned"))
        out.append(pd.DataFrame({
            "ref_id": ref_id,
            "id_i": gi["id"], "id_j": gj["id"],
            "dist_i": gi["dist_name"], "dist_j": gj["dist_name"],
            "level_i": gi["level"], "level_j": gj["level"],
            "mos_i": gi["mos"], "mos_j": gj["mos"],
            "d_trad": d_trad, "d_learned": d_learned, "d_mos": d_mos,
            "strength": np.minimum(d_trad.abs(), d_learned.abs()),
            "human_sides_with": side,
        })[opposite.to_numpy()])
    res = pd.concat(out, ignore_index=True)
    res["types"] = [" / ".join(sorted(t)) for t in zip(res["dist_i"], res["dist_j"])]
    return res.sort_values("strength", ascending=False, ignore_index=True)


def family_agreement(scores: pd.DataFrame) -> pd.DataFrame:
    """Kendall tau between traditional and learned family scores within each reference image:
    which *content* makes the two families disagree most."""
    s = eval_set(scores)
    r = ranks(s, by="ref_id")
    fam = pd.DataFrame({"ref_id": s["ref_id"], "trad": r[TRADITIONAL].mean(axis=1), "learned": r[LEARNED].mean(axis=1)})
    tau = fam.groupby("ref_id").apply(lambda g: g["trad"].corr(g["learned"], method="kendall"), include_groups=False)
    return tau.rename("kendall_tau").sort_values().reset_index()


def select_diverse(cands: pd.DataFrame, k: int = 5, score: str = "gap", max_per: dict | None = None) -> pd.DataFrame:
    """Greedy top-k by |score| with caps, e.g. max_per={"dist_name": 1, "ref_id": 1}.
    The candidate list is already sorted; the caps keep the final set from being one distortion type."""
    max_per = max_per or {}
    counts = {col: {} for col in max_per}
    picked = []
    for idx in cands[score].abs().sort_values(ascending=False).index:
        row = cands.loc[idx]
        if any(counts[c].get(row[c], 0) >= n for c, n in max_per.items()):
            continue
        for c in max_per:
            counts[c][row[c]] = counts[c].get(row[c], 0) + 1
        picked.append(idx)
        if len(picked) == k:
            break
    return cands.loc[picked].reset_index(drop=True)


def systematic(scores: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Share of images per distortion type where |r_metric - r_mos| > threshold.
    High share = systematic failure of the metric on that type; low share = isolated, content-specific."""
    s = eval_set(scores)
    r = ranks(s)
    gap = pd.DataFrame({m.name: (r[m.name] - r["mos"]).abs() > threshold for m in METRICS})
    return gap.groupby(s["dist_name"]).mean().round(3)
