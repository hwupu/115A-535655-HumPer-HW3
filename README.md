# HW3 — Perceptual Metric Reproduction & Failure Analysis

NYCU 535655 *Human Perceptual-Centric Vision and Multimodal Representation Learning*, Fall 2026.
Repo: <https://github.com/hwupu/115A-535655-HumPer-HW3>  
Due **Tue 2026-11-17, before class**. Spec: [`doc/references/HW3.pdf`](doc/references/HW3.pdf).

We evaluate PSNR, SSIM, MS-SSIM, LPIPS (VGG16) and DISTS (VGG16) against human MOS on KADID-10k. The two learned metrics share a backbone, so their comparison is about the formula (D-23); LPIPS-AlexNet is kept as a supplementary column. We report SRCC/PLCC overall and per distortion type, then analyse cases where metrics disagree with humans or with each other.

## Setup

Works on **macOS (Apple Silicon)** and **Windows via WSL2 (Ubuntu)**. Only inference is needed, so no cloud GPU is required.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS & WSL
```

### 2. Clone & sync

```bash
git clone https://github.com/hwupu/115A-535655-HumPer-HW3.git && cd 115A-535655-HumPer-HW3
uv sync                     # creates .venv from uv.lock (Python 3.11 auto-downloaded)
uv run python -c "import humper; print(humper.get_device())"
```

**WSL notes**
- Clone inside the Linux filesystem (e.g. `~/code/`), **not** under `/mnt/c/...`. File I/O on `/mnt/c` is very slow.
- On Linux, `uv sync` installs the CUDA build of PyTorch (a large download).
  - With an NVIDIA GPU and a current Windows driver, `get_device()` returns `cuda`. Don't install a Linux NVIDIA driver inside WSL.
  - Without a GPU it falls back to `cpu`, which is still fine for this homework.
- Set `git config --global core.autocrlf input`. `.gitattributes` already forces LF.

You don't need to recompute: `results/scores.csv` is committed (D-16).

**macOS notes**
- `get_device()` returns `mps`. Unsupported ops fall back to CPU automatically.
- Force a device with `HUMPER_DEVICE=cpu uv run ...`.

### 3. Get the data

Download KADID-10k (3.1 GB). The official server ([database.mmsp-kn.de](https://database.mmsp-kn.de/kadid-10k-database.html)) was unreachable on 2026-10-30, so we use the pyiqa mirror (D-14):

```bash
mkdir -p data/_download
curl -L -C - -o data/_download/kadid10k.tgz \
  https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main/kadid10k.tgz
tar -xzf data/_download/kadid10k.tgz -C data --exclude='.DS_Store'
uv run jupyter nbconvert --to notebook --execute --stdout notebooks/01_data_check.ipynb > /dev/null  # asserts 10,125 pairs
```

The layout should be:

```
data/kadid10k/
├── dmos.csv
└── images/   # I01.png … I81.png and I01_01_01.png …
```

`data/` is gitignored. Never commit the dataset.

## Run

```bash
uv run jupyter lab                          # macOS: opens the browser
uv run jupyter lab --no-browser             # WSL: open the printed http://localhost:8888/... URL in Windows
uv run python scripts/compute_scores.py     # full scoring → results/scores.csv (~40 min on M2 Pro; resumable)
uv run python scripts/compute_scores.py --limit 40 --out results/smoke_cpu.csv   # quick smoke test
uv run pytest                               # unit + sanity tests
```

VS Code users can open the repo with the *Remote - WSL* extension and pick the `.venv` kernel.

## Repo layout

```
src/humper/    reusable code (device, data, metrics, evaluate, failures, viz)
scripts/       long-running jobs
notebooks/     01_data_check, 02_correlation, 03_failure_cases, 04_selected_cases, 05_quant_figures
tests/         sanity checks
data/          dataset (gitignored)
results/       scores.csv, tables/, figures/
report/        report source + final PDF
doc/           process, architecture, decisions, prompt log, scrum board
```

## How we work

AI-native Scrum-lite for 2 people. See [`doc/PROCESS.md`](doc/PROCESS.md).
- Current sprint: [`doc/scrum/BOARD.md`](doc/scrum/BOARD.md)
- Backlog: [`doc/scrum/BACKLOG.md`](doc/scrum/BACKLOG.md)
- Decisions: [`doc/DECISIONS.md`](doc/DECISIONS.md)
- AI usage log: [`doc/PROMPT_LOG.md`](doc/PROMPT_LOG.md)
- Agent context: [`AGENTS.md`](AGENTS.md) (`CLAUDE.md` imports it)
