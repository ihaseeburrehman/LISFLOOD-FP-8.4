#!/usr/bin/env python3
"""Compare final-hour Alzette depth rasters from Metal and CUDA runs.

Usage: compare_alzette.py METAL_WORK CUDA_WORK [OUTPUT_CSV]
"""
import csv
import math
from pathlib import Path
import sys


LOSSES = ("off", "uniform", "evaporation", "greenampt")
UNIFORM_SOLVERS = ("fv1", "fv2", "dg2", "acc")
METAL_ALIASES = (("hwfv1", "fv1"), ("mwdg2", "dg2"),
                 ("acc_nugrid", "acc"))


def final_grid(work, backend, solver, loss):
    grids = sorted((work / f"{backend}_{solver}_{loss}").glob("r-*.wd"))
    if not grids:
        raise FileNotFoundError(f"no depth grids for {backend}/{solver}/{loss}")
    return grids[-1]


def values(path):
    words = path.read_text().split()
    return [float(value) for value in words[12:]]


def metrics(left, right):
    a = values(left)
    b = values(right)
    if len(a) != len(b):
        raise ValueError(f"grid size differs: {left} and {right}")
    pairs = [(x, y) for x, y in zip(a, b) if x > -9000 and y > -9000]
    errors = [abs(x - y) for x, y in pairs]
    return {
        "cells": len(errors),
        "rmse_m": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "mean_abs_m": sum(errors) / len(errors),
        "max_abs_m": max(errors),
        "within_1mm_pct": 100 * sum(error <= 0.001 for error in errors) / len(errors),
        "identical_cells_pct": 100 * sum(error == 0 for error in errors) / len(errors),
    }


def main():
    metal = Path(sys.argv[1]).resolve()
    cuda = Path(sys.argv[2]).resolve()
    output = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else metal / "comparison.csv"
    rows = []
    for solver in UNIFORM_SOLVERS:
        for loss in LOSSES:
            row = {"comparison": "metal_vs_cuda", "solver": solver, "loss": loss}
            row.update(metrics(final_grid(metal, "metal", solver, loss),
                               final_grid(cuda, "cuda", solver, loss)))
            rows.append(row)
    for alias, reference in METAL_ALIASES:
        for loss in LOSSES:
            row = {"comparison": f"metal_{alias}_vs_{reference}",
                   "solver": alias, "loss": loss}
            row.update(metrics(final_grid(metal, "metal", alias, loss),
                               final_grid(metal, "metal", reference, loss)))
            rows.append(row)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['comparison']} {row['solver']} {row['loss']}: "
              f"RMSE={row['rmse_m']:.6g} m, max={row['max_abs_m']:.6g} m, "
              f"within 1 mm={row['within_1mm_pct']:.3f}%")
    print(f"comparison: {output}")


if __name__ == "__main__":
    main()
