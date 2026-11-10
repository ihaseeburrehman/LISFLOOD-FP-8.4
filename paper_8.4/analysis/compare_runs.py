#!/usr/bin/env python3
"""Compare the depth grids of two runs of the same case (Table 4).

    compare_runs.py RUN_A RUN_B [--last K]

RUN_A, RUN_B: output folders holding <root>-NNNN.wd grids (ESRI ASCII).
Outputs 1..K present in both are compared (output 0 is skipped: Metal writes it
after the first step, CUDA before). Wet cells: deeper than 1 mm in either run.
Prints: outputs, % wet cells identical, % within 1 cm, largest difference (mm),
largest relative difference in stored volume (%).
"""
import glob
import os
import sys

import numpy as np


def grids(d):
    return {int(p[-7:-3]): p for p in glob.glob(os.path.join(d, "*-[0-9][0-9][0-9][0-9].wd"))}


def compare(a_dir, b_dir, last=None):
    a, b = grids(a_dir), grids(b_dir)
    keys = sorted(k for k in a if k in b and k >= 1 and (last is None or k <= last))
    wet = ident = close = 0
    worst = vol = 0.0
    for k in keys:
        x = np.loadtxt(a[k], skiprows=6)
        y = np.loadtxt(b[k], skiprows=6)
        m = (x > 1e-3) | (y > 1e-3)
        d = np.abs(x - y)[m]
        wet += m.sum()
        ident += (d == 0).sum()
        close += (d <= 0.01).sum()
        worst = max(worst, d.max() if d.size else 0.0)
        vy = y[y > 0].sum()
        vol = max(vol, abs(x[x > 0].sum() - vy) / vy if vy > 0 else 0.0)
    w = max(wet, 1)
    return len(keys), 100 * ident / w, 100 * close / w, 1000 * worst, 100 * vol


if __name__ == "__main__":
    last = int(sys.argv[sys.argv.index("--last") + 1]) if "--last" in sys.argv else None
    n, ident, close, worst, vol = compare(sys.argv[1], sys.argv[2], last)
    print(f"{n},{ident:.2f},{close:.2f},{worst:.0f},{vol:.4f}")
