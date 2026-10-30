# AGENTS.md — AI agent context for HW3

This is the canonical agent file, read by Codex, Cursor, Copilot, etc.
`CLAUDE.md` only contains `@AGENTS.md`, so Claude Code loads the same content. **Edit only this file.**

**Task:** NYCU 535655 HW3: Perceptual Metric Reproduction & Failure Analysis (spec: `doc/references/HW3.pdf`).
Due **2026-11-17 before class**. Repo: https://github.com/hwupu/115A-535655-HumPer-HW3 (public).
Team of 2: Member A (@hwupu, macOS, M2 Pro) and Member B (@ALLEN07056, Windows + WSL2).

## Read first
- `README.md`: setup & run commands
- `doc/PROCESS.md`: how we work (Scrum-lite, Definition of Done, AI rules)
- `doc/ARCHITECTURE.md`: layout, data flow, interfaces, sanity checks
- `doc/DECISIONS.md`: what is already decided; don't re-litigate `Accepted` items
- `doc/scrum/BOARD.md`: current sprint; `doc/scrum/BACKLOG.md`: all items

## Rules for the agent
- Work on a backlog item (`B-xx`). Reference it in commits.
- After producing kept output, append a row to `doc/PROMPT_LOG.md`.
- For a new non-trivial choice, add a `Proposed` row to `doc/DECISIONS.md`. Don't silently decide.
- Keep reusable logic in `src/humper/`. Notebooks stay thin and import from it.
- Never correlate a metric before aligning its direction (LPIPS/DISTS lower = better).
- Do not write the perceptual explanations of failure cases as final text. Draft observations only; humans own the interpretation.

## Cross-platform rules (macOS + WSL)
- Get the device via `humper.get_device()` (cuda > mps > cpu; override with `HUMPER_DEVICE`). Never hard-code `"mps"` or `"cuda"`.
- Use `pathlib.Path` relative to the repo root. No absolute paths, no `\` separators.
- Commands go through `uv run ...`. No OS-specific shell scripts unless they're POSIX `sh`.
- Results must not depend on the device: compare CPU vs MPS/CUDA on a few images when adding a metric.

## Env
- uv + Python 3.11 (`.python-version`). Dependencies are in `pyproject.toml`; `uv.lock` is committed and universal (macOS + Linux).
- `data/` is gitignored (KADID-10k).
