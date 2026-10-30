# Sprint Board

Status flow: **To Do → Doing → Review → Done**. Move a line between sections and put the owner's name on it.
Async daily: add one dated line under *Daily log* (`done / next / blocked`).

---

## ▶ Sprint 0 — Fri 10-30 → Sun 11-01 (current)

**Goal:** env + data ready; every metric runs on 20 images end-to-end.

### To Do
- [ ] B-01c WSL setup + `uv sync` + device check — B
- [ ] B-02 GenAI policy check; agree on decisions D-01..D-07 — B
- [ ] B-03 KADID-10k download, `data.py`, `01_data_check.ipynb` — A
- [ ] B-04 metric registry + 20-image smoke run — A

### Doing

### Review
- [ ] B-01b pushed to github.com/hwupu/115A-535655-HumPer-HW3 (public); still todo: add B as collaborator — A
- [ ] B-01 git + uv + layout + README + AGENTS.md — A (reviewer: B, check on WSL)

### Done

### Daily log
- 10-30 A: planning docs drafted (P-002). next: B-01.
- 10-30 A: B-01 done → Review (P-003). uv env verified on macOS: torch 2.15 + MPS, pyiqa 0.1.16. next: B-01b.
- 10-30 A: repo public on GitHub, HW3.pdf included (P-004, D-13). next: add B as collaborator, B-03.

### Review + Retro (fill on 11-01)
- Demo:
- Keep:
- Change:

---

## Sprint 1 — Mon 11-02 → Sun 11-08

**Goal:** all scores computed and validated; SRCC/PLCC tables + per-distortion heatmap.
Planned: B-05, B-06, B-07, B-08, B-09, B-10, B-14

## Sprint 2 — Mon 11-09 → Sun 11-15

**Goal:** ≥5 failure cases with figures and explanations; full report draft.
Planned: B-11, B-12, B-13, B-15, B-16

## Release — Mon 11-16 → Tue 11-17 (before class)

**Goal:** submitted.
Planned: B-17, B-18, B-19 — **code freeze Mon 11-16 18:00**
