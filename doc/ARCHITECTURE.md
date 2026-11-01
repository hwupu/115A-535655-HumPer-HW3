# Architecture

Small analysis project. Design goal: **compute expensive scores once, analyse many times.**

## 1. Repo layout

```
115A-535655-HumPer-HW3/
├── README.md                 # setup (macOS + WSL) & run commands
├── AGENTS.md                 # AI agent context (canonical)
├── CLAUDE.md                 # `@AGENTS.md` import only
├── .gitattributes            # LF everywhere (macOS ↔ WSL)
├── pyproject.toml / uv.lock  # uv-managed env (Python 3.11, universal lock)
├── src/humper/               # reusable logic — notebooks import from here
│   ├── device.py             # get_device(): cuda > mps > cpu, $HUMPER_DEVICE override
│   ├── data.py               # dataset → tidy DataFrame
│   ├── metrics.py            # metric registry (name, pyiqa id, direction)
│   ├── evaluate.py           # SRCC / PLCC (+ logistic fit), per-group tables
│   ├── failures.py           # rank-disagreement mining
│   └── viz.py                # figure helpers (triptychs, heatmaps)
├── scripts/
│   └── compute_scores.py     # long-running job → results/scores.csv
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_correlation.ipynb
│   └── 03_failure_cases.ipynb
├── tests/test_sanity.py      # metric sanity checks (§5)
├── data/                     # gitignored — raw dataset
├── results/                  # scores.csv, tables/, figures/
├── report/                   # report source + final PDF
└── doc/                      # process, decisions, prompt log, scrum
```

## 2. Data flow

```
data/kadid10k/ ──data.py──▶ pairs DataFrame
                              │ (ref_path, dist_path, mos, dist_type, level)
                              ▼
               scripts/compute_scores.py  (metrics.py, device=mps|cpu)
                              ▼
                     results/scores.csv   ◀── single source of truth
                     ┌────────┴─────────┐
                     ▼                  ▼
          evaluate.py (02_*)      failures.py (03_*)
          tables/*.csv, heatmap   figures/failure_*.png
                     └────────┬─────────┘
                              ▼
                         report/ (PDF)
```

## 3. Key interfaces

**`device.py`**: `get_device(prefer=None) -> torch.device`.

**`data.py`**: `load_pairs(root, n_refs=None, seed=0) -> DataFrame`, `load_image(rel_path) -> (3,H,W) in [0,1]`
Columns: `id, ref_path, dist_path, ref_id, dist_type, dist_name, dist_group, level, mos`. The `n_refs` subset samples whole reference images.
`mos` is normalised so that **higher = better quality**.

**`metrics.py`**: `MetricSpec(name, pyiqa_name, higher_is_better, family, note)`; `create()` asserts our direction == pyiqa's `lower_better`; `aligned()` flips sign.
- `PSNR, SSIM, MS-SSIM`: higher = better
- `lpips_vgg, dists` (main, both VGG16, D-23) and `lpips_alex` (`EXTRA_METRICS`, supplementary): lower = better → negated by `aligned()` before correlating
- `METRICS` = main comparison (tables, failure analysis); `ALL_METRICS` = also computed and tested

**`scores.csv`**: one row per distorted image. Columns are the pair columns plus one raw-score column per metric (stored raw, not sign-flipped).

**`evaluate.py`**
- `srcc(x, y)`
- `plcc(x, y, fit="logistic5")`: fits the VQEG 5-parameter logistic (on z-scored x), then computes Pearson
- `krcc(x, y)`: Kendall, as an extra
- `table(scores, by=None | "dist_type")`

**`failures.py`**: converts each metric and the MOS to percentile ranks (direction-aligned), then computes
- `gap_human[m] = |rank_m − rank_mos|`
- `gap_pair[a,b] = |rank_a − rank_b|` (traditional vs learned)

It returns the top-k cases, capped at N per distortion type so the selection stays diverse (no cherry-picking).

## 4. Runtime (cross-platform)

| | Member A | Member B |
|---|---|---|
| OS | macOS (Apple Silicon M2 Pro) | Windows + WSL2 Ubuntu |
| Torch wheel (from `uv.lock`) | macOS arm64 (MPS) | Linux x86_64, CUDA-enabled PyPI build |
| Device from `get_device()` | `mps` | `cuda` if NVIDIA GPU + Windows driver, else `cpu` |
| Jupyter | `uv run jupyter lab` | `uv run jupyter lab --no-browser`, open URL in Windows browser (or VS Code Remote-WSL) |

- **Traditional metrics (PSNR/SSIM/MS-SSIM) always run on CPU** (`metrics.device_for`). pyiqa's SSIM uses float64, which MPS doesn't support. This also makes them bit-identical across machines. Only LPIPS/DISTS use the accelerator.
- Code never hard-codes a device. It calls `humper.get_device()`, which also sets `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- All paths are `pathlib.Path` relative to the repo root. That way the same code runs on both OSes, and `scores.csv` stores **relative** paths.
- **Device-independence:** MPS, CUDA and CPU can differ in the last decimals. `scores.csv` records the `device` used. The final numbers come from **one** full run on one machine, so the report table is internally consistent.
- `compute_scores.py` is resumable: it skips IDs already present in `scores.csv`.
- `--limit N` gives a fast smoke run.
- Seeds, devices and package versions are written to `results/scores_run_info.json`.

## 5. Correctness checks (rubric: "correct implementation", 15 pts)

1. Identical image pair: PSNR = 80 (the `pyiqa` cap; verified 2026-10-30), SSIM = 1, LPIPS ≈ 0, DISTS ≈ 0. ✔ tested
2. `pyiqa` PSNR/SSIM match `skimage.metrics` on a few images. Record whether metrics use the Y channel or RGB.
   ✔ `tests/test_sanity.py` (7 real pairs): PSNR = skimage within 5e-5 dB (RGB). SSIM uses YIQ luma (0.299R+0.587G+0.114B, ×255, rounded).
   **Deviation (D-20):** pyiqa clamps the contrast-structure term at 0 (`F.relu(cs_map)`), unlike Wang et al. 2004 / skimage.
   This changes SSIM only where local structure is anti-correlated (jitter: −1.9e-3, JPEG: −2.4e-4 on the samples).
3. Within each (ref, dist_type), metric scores move monotonically with distortion level (check with Spearman vs level).
4. The sign/direction check is applied before PLCC and printed in the notebook. ✔ tested (raw SRCC sign vs `higher_is_better`)
5. Cross-device: on 20 images, CPU vs MPS (A) and CPU vs CUDA (B) agree to within 1e-3 relative error.
   ✔ A, 2026-10-30, 40 pairs: traditional metrics identical; LPIPS-Alex max 2.5e-6, DISTS max 7.1e-6.
   ✔ `test_learned_metrics_cpu_vs_accelerator` (skipped on CPU-only machines) and `test_committed_scores_reproduce`
     (recomputes 7 rows of `scores.csv` on CPU; run on WSL to cross-check D-12).
