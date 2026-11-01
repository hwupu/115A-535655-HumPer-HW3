# Sprint Board

Status flow: **To Do → Doing → Review → Done**. Move a line between sections and put the owner's name on it.
Async daily: add one dated line under *Daily log* (`done / next / blocked`).

---

## Sprint 0 — Fri 10-30 → Sun 11-01 (closed)

**Goal:** env + data ready; every metric runs on 20 images end-to-end. ✔

### Done (verified by @ALLEN07056's approval of the report PR, D-26)
- [x] B-01 git + uv + layout + README + AGENTS.md — A
- [x] B-01b public repo, @ALLEN07056 invited — A
- [x] B-01c WSL setup + `uv sync` + `uv run pytest` — @ALLEN07056 (PR review checklist)
- [x] B-02 GenAI policy check; Proposed decisions accepted (D-26) — @ALLEN07056 + @hwupu
- [x] B-03 KADID-10k + `data.py` + notebook 01 — A
- [x] B-04 `metrics.py` registry + smoke run — A

### Retro
- Keep: committed `scores.csv` lets B work without re-running 40 min of scoring; tests catch implementation details (cs clamp).
- Change: stacked PRs + delete-branch closed #2/#4 on merge; retarget to `main` before merging the next PR.

---

## Sprint 1 — Mon 11-02 → Sun 11-08 (closed early, 11-02)

**Goal:** all scores validated; SRCC/PLCC tables + per-distortion heatmap; report skeleton. ✔ (plus the S2 drafting work)
Code: @hwupu implements, @ALLEN07056 reviews PRs (D-21).

### Done (verified by @ALLEN07056's approval of the report PR, D-26)
- [x] Review of S0 commits that went straight to `main` — @ALLEN07056 (covered by the report PR review)
- [x] B-05 sanity tests (D-20) — PR #1
- [x] B-20 main LPIPS → VGG16 (D-23) — PR #2
- [x] B-11 + B-12 failure mining + candidate figures (D-24) — PR #3
- [x] B-25 five report cases + type × level heatmaps (D-25) — PR #4
- [x] B-06 monotonicity vs level (`evaluate.monotonicity`, notebook 05) — report PR
- [x] B-22 SSIM variants: downsampling +0.13 SRCC, clamp no effect (D-26) — report PR
- [x] B-09 scatter + SRCC-by-type heatmap (notebook 05) — report PR
- [x] B-10 interpretation notes → report §3 — report PR
- [x] B-13 failure explanations → report §4 (AI draft; team owns on approval) — report PR
- [x] B-14 report skeleton + D-08 (Markdown → PDF) — report PR
- [x] B-15 §1–§2, B-16 §3–§5 drafted — report PR
- [x] B-17 contribution statement + GenAI disclosure drafted — report PR
- [x] B-27 case-specific failure figures (@hwupu: "why show LPIPS for SSIM vs MS-SSIM?") — report PR
- [x] D-22 verified: DISTS trained on KADID-10k (footnoted)

### Daily log
- 11-02 A: S1 opened. Code review moves to PRs with @ALLEN07056 (D-21, CODEOWNERS). B-05 → Review (P-006).
- 11-02 A: B-20 → Review (P-007). LPIPS-VGG close to DISTS within types, but weaker across types (color .48, spatial .53).
- 11-02 A: B-11/B-12 → Review (P-008). SSIM fails systematically on color (color_block 47%), PSNR on brightness (brighten 25%).
- 11-02 A: B-25 → Review (P-009). Cross-type calibration is the main failure; SSIM blind to chroma.
- 11-02 A: PRs #1–#4 merged at @hwupu's request (#2/#4 had to be reopened and rebased, see retro). Report draft PR opened (P-010): B-06, B-09, B-22 done; DISTS confirmed in-domain. All tickets close when @ALLEN07056 approves (D-26).

## Sprint 2 — Mon 11-09 → Sun 11-15

**Goal:** ≥5 failure cases with figures and explanations; full report draft. ✔ Done early in S1.
Use the sprint for: team proofreading of `report/report_draft.md`, cutting to ≤ 4 pages.

## Release — Mon 11-16 → Tue 11-17 (before class)

**Goal:** submitted.
Open: B-18 (macOS ✔ 11-02; WSL in report PR review), B-19 PDF + submission — **code freeze Mon 11-16 18:00**. B-21 (stretch) dropped.
