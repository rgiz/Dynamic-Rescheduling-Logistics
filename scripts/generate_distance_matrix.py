#!/usr/bin/env python3
from __future__ import annotations  # must follow shebang

"""Generate straight‑line distance/time matrices from *center_coordinates*.

The CSV can have either column set:
* `location_id,X,Y`  — earlier assumption
* `center_id,x,y`    — confirmed by user

Auto‑detection is case‑insensitive. Override via `--id-col`, `--x-col`, `--y-col` if needed.

Usage (repo root):
```
python3 scripts/generate_distance_matrix.py \
        --coords data/processed/center_coordinates.csv \
        --outfile data/dist_matrix.npz \
        --speed-kmph 60
```
```text
Arguments
─────────
--coords       Path to centre coordinate CSV  (required)
--outfile      Output .npz path               (default data/dist_matrix.npz)
--speed-kmph   Average speed for km → minutes (default 60)
```
"""


import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────

def _detect_cols(df: pd.DataFrame, id_col: str | None, x_col: str | None, y_col: str | None):
    cols = {c.lower(): c for c in df.columns}

    def pick(opts, override):
        if override:
            return override
        for o in opts:
            if o in cols:
                return cols[o]
        raise ValueError(f"Could not find any of {opts} in columns {list(df.columns)}")

    id_c = pick(["location_id", "center_id", "id"], id_col)
    x_c = pick(["x", "longitude"], x_col)
    y_c = pick(["y", "latitude"], y_col)
    return id_c, x_c, y_c


def build_matrices(df: pd.DataFrame, id_c: str, x_c: str, y_c: str, speed_kmph: float):
    df = df.sort_values(id_c).reset_index(drop=True)
    ids = df[id_c].to_numpy()
    x = df[x_c].to_numpy()
    y = df[y_c].to_numpy()

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_km = np.hypot(dx, dy).astype(np.float32)
    time_min = (dist_km / speed_kmph * 60.0).astype(np.float32)  # 60 km/h → 1 km/min
    return ids, dist_km, time_min

# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coords", required=True, type=Path, help="CSV with coordinates")
    p.add_argument("--outfile", default=Path("data/dist_matrix.npz"), type=Path)
    p.add_argument("--speed-kmph", default=60.0, type=float)
    p.add_argument("--id-col", default=None)
    p.add_argument("--x-col", default=None)
    p.add_argument("--y-col", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.coords)
    id_c, x_c, y_c = _detect_cols(df, args.id_col, args.x_col, args.y_col)
    ids, dist, time = build_matrices(df, id_c, x_c, y_c, args.speed_kmph)

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.outfile, ids=ids, dist=dist, time=time)
    print(f"Saved {args.outfile}  |  shape = {dist.shape}")


if __name__ == "__main__":
    main()