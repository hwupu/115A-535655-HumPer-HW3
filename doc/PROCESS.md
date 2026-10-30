# Process — AI-native Scrum-lite (2 people)

Goal: ship HW3 (report PDF + reproducible code) by **Tue 2026-11-17, before class**,
with the least process that still keeps us honest.

## 1. Team & roles

| Who | Machine | Scrum hat | Dev focus |
|---|---|---|---|
| **Member A** (@hwupu) | macOS, M2 Pro (MPS) | Product Owner — owns `BACKLOG.md` priority, accepts work | Pipeline: data loading, metric computation, correlations |
| **Member B** | Windows + WSL2 Ubuntu (CUDA or CPU) | Scrum Master — keeps `BOARD.md` current, runs ceremonies | Analysis: failure-case mining, figures, report |
| **AI agent** (Claude Code, Codex, …) | — | Not a team member — a tool | Drafts code/docs on request; a human owns every output |

Both members review each other's work. Both write the report's Discussion section.

## 2. Cadence

| Sprint | Dates | Goal |
|---|---|---|
| S0 | Fri 10-30 → Sun 11-01 | Env + data + one metric runs end-to-end |
| S1 | Mon 11-02 → Sun 11-08 | All metric scores + SRCC/PLCC (overall & per distortion) |
| S2 | Mon 11-09 → Sun 11-15 | Failure cases + full report draft |
| Release | Mon 11-16 → Tue 11-17 | Freeze, proofread, submit |

## 3. Ceremonies (keep it short)

| Ceremony | When | Length | Output |
|---|---|---|---|
| Planning | First day of sprint | 15 min | Items moved Backlog → `BOARD.md` "To Do", owners set |
| Async daily | Any time each day | 1 line each | Update item status on `BOARD.md` (or chat: *done / next / blocked*) |
| Review + Retro | Last day of sprint | 20 min | Demo notebook outputs; 1 keep / 1 change written in `BOARD.md` |

## 4. Workflow for one item

1. Pick an item from **To Do** and put your name on it → **Doing**.
2. If AI is used: work from `AGENTS.md` context, and log significant prompts in `PROMPT_LOG.md`.
3. If a non-trivial choice is made (dataset, metric config, formula, threshold) → add an entry to `DECISIONS.md`.
4. Commit with the item ID in the message, e.g. `B-07: add logistic PLCC`.
5. Move to **Review**. The other member checks it against the Definition of Done → **Done**.

## 5. Definition of Done

- [ ] Runs from a clean clone with `uv sync` + `uv run ...` (no hidden local state)
- [ ] Works on **both** macOS and WSL. The reviewer runs it on the other platform at least once per sprint.
- [ ] Outputs saved under `results/` (CSV/PNG), not only shown in a notebook cell
- [ ] AI-generated code was read and checked by a human (sanity check written down)
- [ ] `PROMPT_LOG.md` / `DECISIONS.md` updated if relevant
- [ ] Reviewed by the other member

## 6. AI-native rules

- **Context first.** `AGENTS.md` and `doc/` are the shared memory. `AGENTS.md` is canonical; `CLAUDE.md` just imports it, so both Claude Code and other agents (Codex, Cursor, Copilot) read the same rules. Update them so any session (human or AI) can resume.
- **Humans decide, AI drafts.** Every decision in `DECISIONS.md` has a human owner.
- **Verify, don't trust.** Metric code must pass the sanity checks in `ARCHITECTURE.md §5` before its numbers are used.
- **Log what matters.** Log prompts that produced code, analysis or report text that we kept. Skip trivial Q&A.
- **Policy.** Follow the course Academic Integrity & GenAI policy. Disclose AI use in the report (using `PROMPT_LOG.md` as evidence). Interpretation of failure cases must be our own reasoning.

## 7. Git (minimal)

- Repo: **public** at <https://github.com/hwupu/115A-535655-HumPer-HW3> (owner @hwupu; B is added as collaborator).
- Public, so never commit secrets/tokens or the dataset (license: research use, not redistributable). WSL member clones into `~/`, not `/mnt/c/`.
- `.gitattributes` forces LF line endings; WSL member also sets `core.autocrlf input`.
- Use a single `main` branch with small commits; prefix each message with its backlog ID.
- `data/` and large `results/` artifacts are gitignored; `uv.lock` is committed.
- Dependency changes go through `uv add` / `uv remove` only, and the commit includes `uv.lock`. The other member runs `uv sync` after pulling.
- Notebooks: clear heavy outputs before committing, to keep diffs small and avoid merge conflicts. Figures go to `results/figures/`.
