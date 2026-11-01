# Sprint Board

Status flow: **To Do → Doing → Review → Done**. Move a line between sections and put the owner's name on it.
Async daily: add one dated line under *Daily log* (`done / next / blocked`).

---

## Sprint 0 — Fri 10-30 → Sun 11-01 (closed; open items carried to S1, retro still to fill)

**Goal:** env + data ready; every metric runs on 20 images end-to-end.

### To Do
- [ ] B-01c accept GitHub invite; WSL setup + `uv sync` + device check — B (@ALLEN07056)
- [ ] B-02 GenAI policy check; agree on decisions D-01..D-07 — B

### Doing

### Review
- [ ] B-03 KADID-10k (HF mirror, D-14), `data.py`, `01_data_check.ipynb`: 10,125 pairs, MOS ↓ with level — A
- [ ] B-04 `metrics.py` registry + smoke run (40 pairs, CPU vs MPS agree < 1e-5) — A
- [ ] B-07 (pulled from S1) full run: `results/scores.csv` 10,125 × 5 metrics, MPS, 40 min — A
- [ ] B-08 (pulled from S1) `evaluate.py` SRCC/KRCC/PLCC(raw+logistic5) + tests + `02_correlation.ipynb` tables — A
- [ ] B-01 git + uv + layout + README + AGENTS.md — A (reviewer: B, check on WSL)

### Done
- [x] B-01b public repo pushed; @ALLEN07056 invited with write access (accept invite on GitHub) — A

### Daily log
- 10-30 A: planning docs drafted (P-002). next: B-01.
- 10-30 A: B-01 done → Review (P-003). uv env verified on macOS: torch 2.15 + MPS, pyiqa 0.1.16. next: B-01b.
- 10-30 A: repo public on GitHub, HW3.pdf included (P-004, D-13). next: add B as collaborator, B-03.
- 10-30 A: B-03, B-04, B-08 → Review; B-07 running (P-005). Found: pyiqa SSIM needs float64, which fails on MPS, so traditional metrics now run on CPU (D-17). Pulled B-07/B-08 from S1 since S0 finished early. blocked: none.
- 10-31 A: invited @ALLEN07056 (write). Full scoring run at 25%.
- 10-31 A: B-07 done → Review. Overall SRCC: MS-SSIM .826, LPIPS .822, DISTS .814, PSNR .676, SSIM .619. Found 320 pixel-identical pairs (D-18) and 2 logistic-fit failures (D-19). Added B-22 (SSIM downsampling). next: B can start B-05/B-06/B-09 from committed scores.csv.

### Review + Retro (fill on 11-01)
- Demo:
- Keep:
- Change:

---

## ▶ Sprint 1 — Mon 11-02 → Sun 11-08 (current)

**Goal:** all scores validated; SRCC/PLCC tables + per-distortion heatmap; report skeleton.
Code: @hwupu implements, @ALLEN07056 reviews PRs (D-21).

### To Do
- [ ] B-01c (from S0) accept GitHub invite; WSL setup + `uv sync` + `uv run pytest` — @ALLEN07056
- [ ] B-02 (from S0) GenAI policy check; accept/reject Proposed decisions D-01..D-22 — @ALLEN07056 + @hwupu
- [ ] Review S0 PR-less commits (B-01, B-03, B-04, B-07, B-08 on `main`) — @ALLEN07056
- [ ] B-06 monotonicity vs level — A→B
- [ ] B-22 SSIM variants (downsample, unclamped cs) — A→B
- [ ] B-09 heatmap + scatter figures — A→B
- [ ] B-14 report skeleton, decide D-08 — @ALLEN07056
- [ ] B-10 interpretation notes — A+B

### Doing

### Review
- [ ] B-20 main LPIPS → VGG16 (D-23); overall SRCC 0.720 (Alex 0.822, now supplementary) — A→B, PR `b-20-lpips-vgg` (stacked on #1)
- [ ] B-05 sanity tests (12 tests; found pyiqa SSIM cs-clamp, D-20) — A→B, PR `b-05-sanity-tests`

### Done

### Daily log
- 11-02 A: S1 opened. Code review moves to PRs with @ALLEN07056 (D-21, CODEOWNERS). B-05 → Review (P-006). next: B-06.
- 11-02 A: B-20 → Review (P-007). LPIPS-VGG close to DISTS within types, but weaker across types (color .48, spatial .53). next: B-11/B-12 failure mining (plan approved, D-24).

## Sprint 2 — Mon 11-09 → Sun 11-15

**Goal:** ≥5 failure cases with figures and explanations; full report draft.
Planned: B-11, B-12, B-13, B-15, B-16

## Release — Mon 11-16 → Tue 11-17 (before class)

**Goal:** submitted.
Planned: B-17, B-18, B-19 — **code freeze Mon 11-16 18:00**
