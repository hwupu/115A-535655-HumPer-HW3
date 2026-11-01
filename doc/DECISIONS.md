# Decision Log

Lightweight ADRs. Newest at the bottom. Status: `Proposed` → `Accepted` / `Rejected` / `Superseded by D-xx`.
A `Proposed` decision needs both members to agree at the next planning/review.

| ID | Date | Decision | Why | Status | Owner |
|---|---|---|---|---|---|
| D-01 | 2026-10-30 | Run locally (Mac M2 Pro + WSL2 PC), not Google Cloud | Inference only; LPIPS/DISTS are light; avoids setup and cost | Proposed | A |
| D-02 | 2026-10-30 | Use `uv` for env + lockfile, Python 3.11 | Reproducible (`uv.lock`), fast | Proposed | A |
| D-03 | 2026-10-30 | Dataset: KADID-10k (fallback: TID2013) | Public, no request form, 25 distortion types × 5 levels → rich per-type analysis | Proposed | A |
| D-04 | 2026-10-30 | Metrics: PSNR, SSIM, MS-SSIM, LPIPS-alex, DISTS (all via `pyiqa`) | Meets the "traditional + learned + extra" requirement; one consistent API | Superseded by D-23 | A |
| D-05 | 2026-10-30 | PLCC after 5-param logistic fit; also report raw PLCC | Standard VQEG practice; transparent | Proposed | B |
| D-06 | 2026-10-30 | Failure cases = top rank-gap, max 1–2 per distortion type, then visual inspection | Rubric forbids cherry-picking; quantitative first | Proposed | B |
| D-07 | 2026-10-30 | Process: Scrum-lite with markdown backlog/board, 3 short sprints | 2-person team, ~2.5 weeks | Proposed | B |
| D-08 | — | Report tool (LaTeX/Overleaf vs Typst vs Markdown→PDF) | — | Open | B |
| D-09 | 2026-10-30 | Support macOS + WSL2: `get_device()` (cuda>mps>cpu), pathlib relative paths, LF via `.gitattributes`, clone inside WSL fs | Member B uses WSL; same code on both machines | Proposed | A |
| D-10 | 2026-10-30 | `AGENTS.md` is canonical agent context; `CLAUDE.md` = `@AGENTS.md` | Vendor-neutral and works on any checkout (no symlinks, which break on Windows-side clones) | Proposed | A |
| D-11 | 2026-10-30 | Default PyPI torch wheels (no custom index); one universal `uv.lock` | Linux wheel already has CUDA and falls back to CPU; simplest config. Cost: large download on WSL | Proposed | A |
| D-12 | 2026-10-30 | Final reported numbers come from one full run on one machine; other device only cross-checks | Avoid mixing MPS/CUDA/CPU numerical differences in one table | Proposed | B |
| D-13 | 2026-10-30 | Public GitHub repo `hwupu/115A-535655-HumPer-HW3`; HW3.pdf committed; dataset never committed | Owner's choice; easy sharing with B. Dataset excluded for licensing/size | Accepted | A |
| D-14 | 2026-10-30 | Get KADID-10k from the pyiqa HuggingFace mirror (`chaofengc/IQA-PyTorch-Datasets/kadid10k.tgz`) | Official server (datasets.vqa.mmsp-kn.de) timed out on 2026-10-30. Same files; checked: 10,125 pairs, MOS range | Proposed | A |
| D-15 | 2026-10-30 | Use `pyiqa` default configs: PSNR on RGB; SSIM & MS-SSIM on Y (luma); LPIPS v0.1 AlexNet; DISTS | Standard settings used in the literature; reported in Setup section | Proposed | A |
| D-16 | 2026-10-30 | Commit `results/scores.csv` (~1 MB) | B can analyse on any machine without recomputing; final numbers come from one run (D-12) | Proposed | A |
| D-17 | 2026-10-30 | Traditional metrics on CPU, learned metrics on accelerator (`metrics.device_for`) | pyiqa SSIM casts to float64, which MPS rejects; CPU also gives identical results on macOS/WSL | Proposed | A |
| D-18 | 2026-10-31 | Keep the 320 pixel-identical pairs (4 per ref: gaussian/motion blur L1, saturation-2 L1, mean-shift L3) in main results; report "excluding identical" as a sensitivity check | They are part of KADID's official set. Excluding them lowers SRCC by about 0.01–0.04 and doesn't change the ranking | Proposed | A |
| D-19 | 2026-10-31 | If the logistic fit doesn't converge, PLCC falls back to raw (flag `fit_ok=False`) | 2 of 125 per-type fits fail (color_block/psnr, color_saturation_1/ms_ssim); report should footnote this | Proposed | A |
| D-20 | 2026-11-02 | Keep pyiqa's SSIM as-is (contrast-structure clamped at 0), and say so in the Setup section | Found by B-05 cross-check vs a textbook SSIM; differs ≤2e-3 per image on samples. Impact on SRCC to be measured in B-22 together with downsampling | Proposed | A |
| D-21 | 2026-11-02 | Branch + PR per ticket; @ALLEN07056 is code reviewer (CODEOWNERS); @hwupu implements code tickets with the AI agent | Requested by @hwupu. Gives a human review gate on AI-written code (DoD) | Proposed | A |
| D-22 | 2026-11-02 | Treat DISTS as possibly *in-domain* on KADID-10k: verify in Ding et al. (TPAMI 2022) whether its weights were fitted on KADID-10k; if so, footnote it in Results | A metric tuned on the test set has an unfair advantage; affects interpretation, not code | Proposed | A |
| D-23 | 2026-11-02 | Main LPIPS = **LPIPS-VGG** (`lpips-vgg`, v0.1), same VGG16 backbone as DISTS. LPIPS-AlexNet kept as supplementary column `lpips_alex` | HW is about comparing *formulas*; a shared backbone removes the backbone confound between LPIPS and DISTS. Backbones change raw values (Zhang et al. 2018), but all comparisons are rank-based anyway | Accepted (A); B to confirm | A |
| D-24 | 2026-11-02 | Failure mining: percentile ranks (not raw values); drop identical pairs and images with `mos_var` > 90th pct; cases A (metric pair), B (metric vs MOS, signed), C (traditional vs learned within one reference); greedy top-5 per case with caps (1 per type and reference; A ≤ 2 per metric pair; B ≤ 2 per metric; C skips human ties); humans pick the final 6–8 (≥ 2 per case) | Plan approved by @hwupu 2026-11-02. Ranks remove scale and backbone effects; the caps and the full candidate CSVs make the selection auditable (spec §7) | Accepted (A); B to confirm | A |

## Template

```
| D-xx | YYYY-MM-DD | <what we chose> | <why; alternatives considered> | Proposed | <A/B> |
```
