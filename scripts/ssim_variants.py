"""B-22: SSIM variants on all pairs -> results/ssim_variants.csv (see humper.ssim_variants).

    uv run python scripts/ssim_variants.py [--limit N]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from humper.data import REPO_ROOT, load_image, load_pairs
from humper.ssim_variants import VARIANTS, luma255, ssim


def one(row: tuple[str, str, str]) -> dict:
    pid, ref_path, dist_path = row
    x, y = luma255(load_image(ref_path)), luma255(load_image(dist_path))
    return {"id": pid, **{name: ssim(x, y, **kw) for name, kw in VARIANTS.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "ssim_variants.csv"))
    args = ap.parse_args()
    pairs = load_pairs().head(args.limit) if args.limit else load_pairs()
    rows = list(pairs[["id", "ref_path", "dist_path"]].itertuples(index=False, name=None))
    with ProcessPoolExecutor() as ex:
        out = pd.DataFrame(ex.map(one, rows, chunksize=64))
    out.to_csv(args.out, index=False, float_format="%.6f")
    print(f"{len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
