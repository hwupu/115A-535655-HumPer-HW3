# Decision Log

Lightweight ADRs. Newest at the bottom. Status: `Proposed` → `Accepted` / `Rejected` / `Superseded by D-xx`.
A `Proposed` decision needs both members to agree at the next planning/review.

| ID | Date | Decision | Why | Status | Owner |
|---|---|---|---|---|---|
| D-01 | 2026-10-30 | Run locally (Mac M2 Pro + WSL2 PC), not Google Cloud | Inference only; LPIPS/DISTS are light; avoids setup and cost | Proposed | A |
| D-02 | 2026-10-30 | Use `uv` for env + lockfile, Python 3.11 | Reproducible (`uv.lock`), fast | Proposed | A |
| D-03 | 2026-10-30 | Dataset: KADID-10k (fallback: TID2013) | Public, no request form, 25 distortion types × 5 levels → rich per-type analysis | Proposed | A |
| D-04 | 2026-10-30 | Metrics: PSNR, SSIM, MS-SSIM, LPIPS-alex, DISTS (all via `pyiqa`) | Meets the "traditional + learned + extra" requirement; one consistent API | Proposed | A |
| D-05 | 2026-10-30 | PLCC after 5-param logistic fit; also report raw PLCC | Standard VQEG practice; transparent | Proposed | B |
| D-06 | 2026-10-30 | Failure cases = top rank-gap, max 1–2 per distortion type, then visual inspection | Rubric forbids cherry-picking; quantitative first | Proposed | B |
| D-07 | 2026-10-30 | Process: Scrum-lite with markdown backlog/board, 3 short sprints | 2-person team, ~2.5 weeks | Proposed | B |
| D-08 | — | Report tool (LaTeX/Overleaf vs Typst vs Markdown→PDF) | — | Open | B |
| D-09 | 2026-10-30 | Support macOS + WSL2: `get_device()` (cuda>mps>cpu), pathlib relative paths, LF via `.gitattributes`, clone inside WSL fs | Member B uses WSL; same code on both machines | Proposed | A |
| D-10 | 2026-10-30 | `AGENTS.md` is canonical agent context; `CLAUDE.md` = `@AGENTS.md` | Vendor-neutral and works on any checkout (no symlinks, which break on Windows-side clones) | Proposed | A |
| D-11 | 2026-10-30 | Default PyPI torch wheels (no custom index); one universal `uv.lock` | Linux wheel already has CUDA and falls back to CPU; simplest config. Cost: large download on WSL | Proposed | A |
| D-12 | 2026-10-30 | Final reported numbers come from one full run on one machine; other device only cross-checks | Avoid mixing MPS/CUDA/CPU numerical differences in one table | Proposed | B |
| D-13 | 2026-10-30 | Public GitHub repo `hwupu/115A-535655-HumPer-HW3`; HW3.pdf committed; dataset never committed | Owner's choice; easy sharing with B. Dataset excluded for licensing/size | Accepted | A |

## Template

```
| D-xx | YYYY-MM-DD | <what we chose> | <why; alternatives considered> | Proposed | <A/B> |
```
