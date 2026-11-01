# Prompt Log

Log prompts whose output we **kept** (code, analysis, report text, or a decision). Skip trivial Q&A.
This is our evidence for the course GenAI disclosure.

| ID | Date | Who | Tool / Model | Item | Prompt (summary) | Output → where | Human check |
|---|---|---|---|---|---|---|---|
| P-001 | 2026-10-30 | A | Claude Code / Opus 5.5 | — | Read HW3.pdf; suggest plan & env for Mac M2 Pro; is GCloud needed? use uv | Plan, env commands → D-01..D-05 | Read spec ourselves; decisions still Proposed |
| P-002 | 2026-10-30 | A | Claude Code / Opus 5.5 | — | Set up AI-native Scrum-lite for 2 people: process, architecture, sprints in markdown | `doc/*`, `CLAUDE.md` | Team review at S0 planning |
| P-003 | 2026-10-30 | A | Claude Code / Opus 5.5 | B-01 | Support teammate on WSL; add README + agent link file; init uv & git | `README.md`, `AGENTS.md`, `CLAUDE.md`, `.git*`, `pyproject.toml`, `uv.lock`, `src/humper/device.py`, D-09..D-12 | Ran `uv run` import + PSNR identical-pair check on macOS; WSL check pending (B-01c) |
| P-004 | 2026-10-30 | A | Claude Code / Opus 5.5 | B-01b | Set GitHub id hwupu in docs; public repo OK incl. HW3.pdf; create & push repo | README, AGENTS, PROCESS, scrum docs, D-13; GitHub repo | Checked repo page after push |
| P-005 | 2026-10-30 | A | Claude Code / Opus 5.5 | B-03, B-04, B-07, B-08 | "proceed with implementations": data loader, metric registry, evaluate, scoring script, notebooks 01/02 | `src/humper/{data,metrics,evaluate}.py`, `scripts/compute_scores.py`, `notebooks/0{1,2}_*.ipynb`, `tests/test_evaluate.py`, D-14..D-17 | Data-check asserts; distortion-type montage inspected; identical-pair values; CPU vs MPS diff; pytest |
| P-006 | 2026-11-02 | A | Claude Code / Opus 5.5 | B-05 | "what backbone do you use? put ALLEN07056 on code review; continue with next ticket" | `tests/{conftest,test_sanity}.py`, `.github/CODEOWNERS`, PROCESS/ARCHITECTURE/BACKLOG/BOARD, D-20..D-22 | pytest 12 passed; SSIM deviation traced to `F.relu(cs_map)` in pyiqa source and reproduced to 1e-5 |
| P-007 | 2026-11-02 | A | Claude Code / Opus 5.5 | B-20 | "use LPIPS-VGG because this HW is about comparing formulas; update the code" | `metrics.py` (METRICS / EXTRA_METRICS), `data.py` (mos_var), `compute_scores.py` (append-only run log), tests, `scores.csv` + tables, D-23 | 17 tests pass; LPIPS-VGG identical-pair = 0; spatial-map mean equals scalar score; full 10,125-pair run on MPS |
| P-010 | 2026-11-02 | A | Claude Code / Opus 5.5 | B-26 (+B-06, B-09, B-22, B-10, B-13–B-17) | "close all the tickets; extend the draft to describe the experiment, quantitative results and interpretation; create PR; @ALLEN07056's approval = all tickets closed and verified" | `report/report_draft.md`, `evaluate.monotonicity`, `ssim_variants.py` + script, `viz.scatter_figure/type_heatmap`, `notebooks/05_quant_figures.ipynb`, `results/figures/quant/*`, tables, D-08/D-22/D-26, board | 22 tests (variants match pyiqa to 1e-5/1e-4); every number in the draft recomputed; DISTS training data checked in Ding et al. §II-D; macOS clean-clone run |
| P-009 | 2026-11-02 | A | Claude Code / Opus 5.5 | B-25 | "choose the top disagreement for A-1 SSIM vs MS-SSIM, A-2 LPIPS vs DISTS, B-1 vs SRCC, B-2 vs PLCC, C-1 SSIM vs LPIPS; heatmap for distortions × levels; draft a report with formulas, direction claim, possible factors" | `failures.logistic_residuals/grid`, `family_pairs(trad=, learned=)`, `viz` heatmaps, `notebooks/04_selected_cases.ipynb`, `results/figures/selected/*`, `results/failures/selected_report5*.csv`, `report/failure_cases_draft.md`, D-25 | 3 new unit tests; figures inspected; SSIM luma/per-scale claims recomputed independently; factors marked as draft hypotheses for B-13 |
| P-008 | 2026-11-02 | A | Claude Code / Opus 5.5 | B-11, B-12 | "how to compute strongest disagreement / top-5 for the 3 cases?" → plan; "plan is good, proceed" | `src/humper/{failures,viz}.py`, `tests/test_failures.py`, `notebooks/03_failure_cases.ipynb`, `results/failures/*`, `results/figures/failures/*`, D-24 | 5 synthetic unit tests (planted disagreements found); figures inspected; selection only, no explanations written by AI |

## Template

```
| P-xxx | YYYY-MM-DD | A/B | <tool/model> | B-xx | <1-line summary of prompt> | <file / section> | <how we verified> |
```
