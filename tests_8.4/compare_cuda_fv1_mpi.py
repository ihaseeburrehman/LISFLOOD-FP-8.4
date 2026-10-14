#!/usr/bin/env python3
"""Compare single-rank and multi-rank CUDA FV1 Alzette outputs."""
import math
import sys
from pathlib import Path


def read_raster(path):
    lines = path.read_text().splitlines()
    header = lines[:6]
    values = [float(value) for line in lines[6:] for value in line.split()]
    return header, values


def compare_rasters(single, multi, tolerance):
    patterns = ("r-*.wd", "r-*.Qx", "r-*.Qy", "r-*.elev", "r.max")
    worst = (0.0, None, 0.0)
    compared = 0
    for pattern in patterns:
        one = sorted(single.glob(pattern))
        many = sorted(multi.glob(pattern))
        if len(one) != len(many) or bool(one) != bool(many):
            raise RuntimeError(
                f"{pattern} file count differs: single={len(one)}, multi={len(many)}")
        if not one:
            continue
        if worst[1] is None:
            worst = (0.0, one[0].name, 0.0)
        for a, b in zip(one, many):
            ha, va = read_raster(a)
            hb, vb = read_raster(b)
            if ha != hb or len(va) != len(vb):
                raise RuntimeError(f"raster shape/header differs: {a.name} vs {b.name}")
            if not all(math.isfinite(v) for v in va + vb):
                raise RuntimeError(f"non-finite values in {a.name} or {b.name}")
            errors = [abs(x-y) for x, y in zip(va, vb)]
            maximum = max(errors, default=0.0)
            rmse = math.sqrt(sum(e*e for e in errors) / max(1, len(errors)))
            unit = "m" if a.suffix in (".wd", ".elev", ".max") else "m2/s"
            print(f"{a.name}: max_abs={maximum:.9g} {unit}, rmse={rmse:.9g} {unit}")
            if maximum > worst[0]:
                worst = (maximum, a.name, rmse)
            compared += 1
    if not compared:
        raise RuntimeError("no snapshot or maximum-field rasters found")
    print(f"worst raster: {worst[1]} max_abs={worst[0]:.9g} rmse={worst[2]:.9g}")
    if worst[0] > tolerance:
        raise RuntimeError(f"raster error {worst[0]:.9g} exceeds {tolerance:.9g}")


def read_mass(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or not fields[0][0].isdigit():
            continue
        rows.append([float(value) for value in fields])
    return rows


def compare_mass(single, multi, tolerance):
    a = read_mass(single / "r.mass")
    b = read_mass(multi / "r.mass")
    if len(a) != len(b) or not a:
        raise RuntimeError(f"mass row count differs: single={len(a)}, multi={len(b)}")
    max_relative = 0.0
    for i, (ra, rb) in enumerate(zip(a, b)):
        if len(ra) != len(rb):
            raise RuntimeError(f"mass column count differs at row {i}")
        for col, (x, y) in enumerate(zip(ra, rb)):
            # Verror is a small residual; scale it by total stored water
            # volume so harmless reduction-order rounding is assessed against
            # the conserved quantity instead of against zero.
            scale = max(1.0, abs(x), abs(y))
            if col == 10 and len(ra) > 5:
                scale = max(1.0, abs(ra[5]), abs(rb[5]))
            relative = abs(x-y) / scale
            max_relative = max(max_relative, relative)
            if relative > tolerance:
                raise RuntimeError(f"mass mismatch row {i} col {col}: {x} vs {y}")
    print(f"mass rows={len(a)}, maximum scaled difference={max_relative:.9g}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: compare_cuda_fv1_mpi.py SINGLE_DIR MULTI_DIR [raster_tol_m]")
    compare_rasters(Path(sys.argv[1]), Path(sys.argv[2]),
                    float(sys.argv[3]) if len(sys.argv) == 4 else 1e-6)
    compare_mass(Path(sys.argv[1]), Path(sys.argv[2]), 1e-6)
    print("CUDA FV1 single/multi-GPU comparison PASSED")
