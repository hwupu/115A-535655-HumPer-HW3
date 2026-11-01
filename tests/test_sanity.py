"""B-05: metric sanity checks (ARCHITECTURE.md §5). Rubric: "correct metric implementation"."""

import numpy as np
import pytest
import torch
from conftest import needs_data, needs_scores
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from humper import get_device
from humper.metrics import ALL_METRICS, BY_NAME, METRICS, create, score

# --- configuration we report in the Setup section (D-15) ---------------------------------


def test_reported_pyiqa_configs():
    """If pyiqa changes a default, this fails and the report's Setup section must be updated."""
    from pyiqa.default_model_configs import DEFAULT_CONFIGS as C

    assert C["psnr"]["metric_opts"]["test_y_channel"] is False   # PSNR on RGB
    assert C["ssim"]["metric_opts"]["test_y_channel"] is True    # SSIM on luma
    assert C["ssim"]["metric_opts"]["downsample"] is False       # no MATLAB auto-downsampling (see B-22)
    assert C["ms_ssim"]["metric_opts"]["test_y_channel"] is True
    assert C["lpips-vgg"]["metric_opts"]["net"] == "vgg"            # main LPIPS: same backbone as DISTS (D-23)
    assert C["lpips-vgg"]["metric_opts"]["version"] == "0.1"
    assert C["lpips"]["metric_opts"]["net"] == "alex"               # supplementary lpips_alex


def test_direction_matches_pyiqa(cpu_models):
    """`create` asserts this too; listed here so it shows up in the test report."""
    for m in ALL_METRICS:
        assert cpu_models[m.name].lower_better == (not m.higher_is_better)


# --- check 1: identical input gives the "perfect" value ----------------------------------


@needs_data
@needs_scores
def test_identical_pair_gives_perfect_score(cpu_models, sample):
    _, _, ref = sample
    perfect = {"psnr": 80.0, "ssim": 1.0, "ms_ssim": 1.0, "lpips_vgg": 0.0, "lpips_alex": 0.0, "dists": 0.0}
    for name, expected in perfect.items():
        got = score(cpu_models[name], ref, ref)
        np.testing.assert_allclose(got, expected, atol=1e-5, err_msg=name)


# --- check 2: independent re-implementation (scikit-image) -------------------------------


def _yiq_luma_255(x: torch.Tensor) -> np.ndarray:
    """Same luma as pyiqa's SSIM: YIQ Y = .299R + .587G + .114B, scaled to [0, 255] and rounded."""
    y = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    return (y.double() * 255).round().numpy()


@needs_data
@needs_scores
def test_psnr_matches_skimage(cpu_models, sample):
    rows, dist, ref = sample
    ours = score(cpu_models["psnr"], dist, ref).numpy()
    for i in range(len(rows)):
        if ours[i] >= 80:  # identical pair: skimage returns inf, pyiqa caps at 80
            continue
        sk = peak_signal_noise_ratio(ref[i].numpy(), dist[i].numpy(), data_range=1.0)
        assert ours[i] == pytest.approx(sk, abs=1e-3), rows.id[i]


def _reference_ssim(x: np.ndarray, y: np.ndarray, clamp_cs: bool) -> float:
    """Wang et al. 2004 SSIM: 11x11 Gaussian (sigma 1.5), K=(0.01, 0.03), L=255, 'valid' region.
    clamp_cs=True reproduces pyiqa, which applies ReLU to the contrast-structure term."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    f = lambda a: gaussian_filter(a, 1.5, truncate=3.5)[5:-5, 5:-5]  # noqa: E731
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx**2, f(y * y) - my**2, f(x * y) - mx * my
    cs = (2 * sxy + c2) / (sxx + syy + c2)
    if clamp_cs:
        cs = np.maximum(cs, 0)
    return float((((2 * mx * my + c1) / (mx**2 + my**2 + c1)) * cs).mean())


@needs_data
@needs_scores
def test_ssim_matches_reference(cpu_models, sample):
    """pyiqa SSIM == textbook SSIM except for one documented deviation (D-20):
    pyiqa clamps negative contrast-structure to 0. skimage (unclamped) validates our reference."""
    rows, dist, ref = sample
    ours = score(cpu_models["ssim"], dist, ref).numpy()
    for i in range(len(rows)):
        x, y = _yiq_luma_255(ref[i]), _yiq_luma_255(dist[i])
        sk = structural_similarity(x, y, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        assert _reference_ssim(x, y, clamp_cs=False) == pytest.approx(sk, abs=1e-6), rows.id[i]
        assert ours[i] == pytest.approx(_reference_ssim(x, y, clamp_cs=True), abs=1e-5), rows.id[i]


# --- check 5: same numbers on CPU and accelerator, and the committed scores reproduce ----


@needs_data
@needs_scores
@pytest.mark.skipif(get_device().type == "cpu", reason="no MPS/CUDA on this machine")
def test_learned_metrics_cpu_vs_accelerator(cpu_models, sample):
    _, dist, ref = sample
    device = get_device()
    for name in ["lpips_vgg", "lpips_alex", "dists"]:
        accel = score(create(BY_NAME[name], device), dist, ref)
        cpu = score(cpu_models[name], dist, ref)
        np.testing.assert_allclose(accel, cpu, rtol=1e-3, atol=1e-5, err_msg=f"{name} on {device}")


@needs_data
@needs_scores
def test_committed_scores_reproduce(cpu_models, sample):
    """Recompute a few rows of results/scores.csv on this machine (CPU). Run on WSL to cross-check D-12."""
    rows, dist, ref = sample
    for m in ALL_METRICS:
        got = score(cpu_models[m.name], dist, ref).numpy()
        tol = dict(rtol=1e-5) if m.family == "traditional" else dict(rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(got, rows[m.name].to_numpy(), **tol, err_msg=m.name)


# --- checks on the committed scores file (no images needed) ------------------------------


@needs_scores
def test_scores_complete(scores):
    assert len(scores) == 10125
    assert scores[[m.name for m in ALL_METRICS]].notna().all().all()


@needs_scores
def test_raw_correlation_sign_matches_direction(scores):
    """Check 4: before alignment, distances correlate negatively with MOS, similarities positively."""
    for m in ALL_METRICS:
        rho = scores[m.name].corr(scores["mos"], method="spearman")
        assert (rho > 0) == m.higher_is_better, (m.name, rho)


@needs_scores
def test_identical_pairs_are_the_known_320(scores):
    """D-18: 4 identical pairs per reference image (80 refs; I18 has none)."""
    ident = scores[scores.psnr >= 80]
    assert len(ident) == 320
    assert set(ident.dist_name) == {"gaussian_blur", "motion_blur", "color_saturation_2", "mean_shift"}
    np.testing.assert_allclose(ident[["ssim", "ms_ssim"]], 1.0, atol=1e-6)
    np.testing.assert_allclose(ident[["lpips_vgg", "lpips_alex", "dists"]], 0.0, atol=1e-6)


@needs_data
@needs_scores
def test_ssim_variants_match_pyiqa(sample):
    """B-22: our variant code reproduces pyiqa SSIM (clamped, no downsampling) and pyiqa's downsample=True."""
    from pyiqa.archs.ssim_arch import ssim as pyiqa_ssim
    from pyiqa.utils.color_util import to_y_channel

    from humper.ssim_variants import luma255, ssim

    rows, dist, ref = sample
    for i in range(1, len(rows)):  # skip the identical pair
        x, y = luma255(ref[i]), luma255(dist[i])
        assert ssim(x, y) == pytest.approx(rows.ssim[i], abs=1e-5), rows.id[i]
        yd, yr = (to_y_channel(t[None], 255, "yiq").double() for t in (dist[i], ref[i]))
        want = float(pyiqa_ssim(yd, yr, data_range=255, downsample=True))
        assert ssim(x, y, down=True) == pytest.approx(want, abs=1e-4), rows.id[i]


@needs_data
@needs_scores
def test_ms_ssim_scales_reproduce_pyiqa(sample):
    """The scale-by-scale MS-SSIM used for figures multiplies back to the committed pyiqa MS-SSIM."""
    from humper.ssim_variants import luma255, ms_ssim_scales

    rows, dist, ref = sample
    for i in range(1, len(rows)):
        sc = ms_ssim_scales(luma255(ref[i]), luma255(dist[i]))
        assert np.prod([d["mean"] ** d["weight"] for d in sc]) == pytest.approx(rows.ms_ssim[i], abs=1e-5), rows.id[i]


@needs_data
@needs_scores
def test_explanation_views_add_up_to_scores(sample):
    """B-27: the case-figure views are decompositions of the metrics, not illustrations."""
    from humper.explain import dists_stages, lpips_layers, ssim_map

    rows, dist, ref = sample
    for i in [1, 4]:
        tex, st = dists_stages(ref[i], dist[i])
        assert tex.sum() + st.sum() == pytest.approx(rows.dists[i], rel=1e-3), rows.id[i]
        assert lpips_layers(ref[i], dist[i]).sum() == pytest.approx(rows.lpips_vgg[i], rel=1e-3), rows.id[i]
        assert ssim_map(ref[i], dist[i]).mean() == pytest.approx(rows.ssim[i], abs=1e-5), rows.id[i]
