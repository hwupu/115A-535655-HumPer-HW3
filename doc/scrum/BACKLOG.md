# Product Backlog

Owner: Member A (PO). Ordered by priority, so the top item is the next one to do.
Code tickets: implemented by A (+AI), reviewed by B via PR (D-21). `A→B` = A implements, B reviews.
Size: **S** ≤ 2h, **M** ≈ half day, **L** ≈ 1 day. Rubric points in brackets.

| ID | Item | Size | Sprint | Owner | Rubric |
|---|---|---|---|---|---|
| **Epic 1 — Setup** |||||
| B-01 | `git init`, `.gitignore`/`.gitattributes`, uv project (`pyproject`, `uv.lock`), folder layout, README, AGENTS.md | S | S0 | A | Repro [15] |
| B-01b | Push to public GitHub repo `hwupu/115A-535655-HumPer-HW3`; add B as collaborator | S | S0 | A | Repro [15] |
| B-01c | WSL setup on B's machine: clone in `~/`, `uv sync`, `get_device()`, Jupyter from Windows browser | S | S0 | B | Repro [15] |
| B-02 | Confirm the course GenAI policy; agree to D-01..D-07 | S | S0 | B | — |
| B-03 | Download KADID-10k to `data/`; write `data.py` + `01_data_check.ipynb` (counts, MOS hist, direction) | M | S0 | A | Setup [15] |
| B-04 | `metrics.py` registry + smoke run of all metrics on 20 images | M | S0 | A | Impl [15] |
| **Epic 2 — Metric correctness** |||||
| B-05 | `tests/test_sanity.py`: identical pair, skimage cross-check, Y/RGB note, cross-device check, committed-scores reproduction | M | S1 | A→B | Impl [15] |
| B-06 | Monotonicity-vs-level check per distortion type | S | S1 | A→B | Impl [15] |
| **Epic 3 — Quantitative evaluation** |||||
| B-07 | `compute_scores.py` full run (resumable) → `results/scores.csv` + `run_info.json` | M | S1 | A | Quant [20] |
| B-08 | `evaluate.py`: SRCC, PLCC (raw + logistic), direction alignment | M | S1 | A | Quant [20] |
| B-09 | Overall table + per-distortion-type SRCC heatmap + scatter plots | M | S1 | A→B | Quant [20] |
| B-10 | Write the interpretation notes for the quant results (where each metric wins/loses) | S | S1 | A+B | Analysis [20] |
| **Epic 4 — Failure analysis** |||||
| B-11 | `failures.py`: rank gaps (metric↔MOS, PSNR/SSIM↔LPIPS/DISTS), diversity cap | M | S2 | A→B | Failure [20] |
| B-12 | Select ≥5 cases (target 6–8, then cut); triptych figures with scores + MOS rank | M | S2 | A→B | Failure [20] |
| B-13 | Perceptual explanation per case (structure/texture/shift/color/contrast/saliency) | L | S2 | A+B | Analysis [20] |
| **Epic 5 — Report & submission** |||||
| B-14 | Report skeleton (6 required sections), decide D-08 | S | S1 | B | Clarity [10] |
| B-15 | Draft §1 Metrics & Dataset, §2 Setup | M | S2 | A | Repro [15] |
| B-16 | Draft §3 Results, §4 Failure cases, §5 Discussion (+ "factor not captured") | L | S2 | A+B | Analysis [20] |
| B-17 | Contribution statement + GenAI disclosure (from `PROMPT_LOG.md`) | S | Rel | B | — |
| B-18 | Clean-clone reproduction test on **both** macOS and WSL; README up to date | S | Rel | A+B | Repro [15] |
| B-19 | Page check (≤4 pp), proofread, submit PDF + code | S | Rel | A+B | Clarity [10] |
| B-22 | SSIM variants: original auto-downsampling (`downsample=True`) and unclamped cs (D-20). Our SRCC 0.62 is below commonly reported values. Decide which to report | S | S1 | A→B | Impl [15] |
| **Stretch** |||||
| B-20 | Add LPIPS-vgg or a no-reference metric for contrast | M | — | — | — |
| B-21 | Repeat the key results on TID2013 subset for robustness | L | — | — | — |
