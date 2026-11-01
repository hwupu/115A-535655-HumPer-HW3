"""Compute every metric for every KADID-10k pair -> results/scores.csv (resumable).

    uv run python scripts/compute_scores.py                      # full run
    uv run python scripts/compute_scores.py --limit 20 --out results/smoke.csv
    HUMPER_DEVICE=cpu uv run python scripts/compute_scores.py ...  # force a device
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from humper import get_device
from humper.data import REPO_ROOT, load_image, load_pairs
from humper.metrics import ALL_METRICS, BY_NAME, create, device_for, score


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "scores.csv")
    p.add_argument("--metrics", nargs="+", default=[m.name for m in ALL_METRICS], choices=list(BY_NAME))
    p.add_argument("--limit", type=int, help="only the first N pairs (smoke test)")
    p.add_argument("--n-refs", type=int, help="random subset of reference images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", help="cuda | mps | cpu (default: auto)")
    p.add_argument("--save-every", type=int, default=50, help="checkpoint every N batches")
    return p.parse_args()


def save(df: pd.DataFrame, out: Path) -> None:
    metric_cols = [m.name for m in ALL_METRICS if m.name in df]
    df = df[[c for c in df if c not in metric_cols and c != "device"] + metric_cols + ["device"]]
    tmp = out.with_suffix(".tmp")
    df.to_csv(tmp, index=False, float_format="%.6g")
    tmp.replace(out)  # atomic, so an interrupted run never leaves a broken CSV


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)

    pairs = load_pairs(n_refs=args.n_refs, seed=args.seed)
    if args.limit:
        pairs = pairs.head(args.limit)

    df = pairs.copy()
    if args.out.exists():  # resume: keep already-computed scores
        prev = pd.read_csv(args.out)
        keep = [c for c in prev.columns if c in BY_NAME or c == "device"]
        df = df.merge(prev[["id", *keep]], on="id", how="left")
    for name in args.metrics:
        if name not in df:
            df[name] = float("nan")
    if "device" not in df:
        df["device"] = None

    todo = df.index[df[args.metrics].isna().any(axis=1)]
    print(f"device={device}  pairs={len(df)}  todo={len(todo)}  metrics={args.metrics}")
    if len(todo) == 0:
        return

    models = {name: create(BY_NAME[name], device) for name in args.metrics}
    ref_cache: dict[str, torch.Tensor] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    for b, idx in enumerate(tqdm(batches, unit="batch")):
        rows = df.loc[idx]
        for p in rows["ref_path"]:
            if p not in ref_cache:
                ref_cache[p] = load_image(p)
        # stay on CPU: pyiqa moves inputs to each model's device (see metrics.device_for)
        ref = torch.stack([ref_cache[p] for p in rows["ref_path"]])
        dist = torch.stack([load_image(p) for p in rows["dist_path"]])
        for name, model in models.items():
            df.loc[idx, name] = score(model, dist, ref).numpy()
        df.loc[idx, "device"] = device.type
        if (b + 1) % args.save_every == 0:
            save(df, args.out)
    save(df, args.out)

    info = {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
        "device": device.type,
        "device_per_metric": {m: device_for(BY_NAME[m], device).type for m in args.metrics},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["torch", "torchvision", "pyiqa", "numpy", "pandas", "scipy"]},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_pairs": len(df),
        "metrics": {m: {"pyiqa": BY_NAME[m].pyiqa_name, "higher_is_better": BY_NAME[m].higher_is_better}
                    for m in args.metrics},
    }
    # one entry per run: a later run that adds a metric must not erase how earlier columns were made
    info_path = args.out.with_name(args.out.stem + "_run_info.json")
    runs = json.loads(info_path.read_text())["runs"] if info_path.exists() else []
    info_path.write_text(json.dumps({"runs": [*runs, info]}, indent=2))
    print(f"wrote {args.out} in {info['seconds']} s")


if __name__ == "__main__":
    main()
