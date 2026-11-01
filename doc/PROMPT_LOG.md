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

## Template

```
| P-xxx | YYYY-MM-DD | A/B | <tool/model> | B-xx | <1-line summary of prompt> | <file / section> | <how we verified> |
```
