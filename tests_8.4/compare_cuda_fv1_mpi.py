#!/usr/bin/env python3
"""Compare single-rank and multi-rank CUDA FV1 Alzette outputs."""
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np


def compare_rasters(single, multi, tolerance):
    # Accept the original small-test `r-*` prefix and named cases such as
    # Alzette's `1hr-*` output prefix.
    patterns = ("*.wd", "*.Qx", "*.Qy", "*.elev", "*.max")
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
            maximum = 0.0
            squared_error = 0.0
            value_count = 0
            with a.open() as fa, b.open() as fb:
                ha = [fa.readline().rstrip("\n") for _ in range(6)]
                hb = [fb.readline().rstrip("\n") for _ in range(6)]
                if ha != hb:
                    raise RuntimeError(f"raster header differs: {a.name} vs {b.name}")
                for row_a, row_b in zip_longest(fa, fb):
                    if row_a is None or row_b is None:
                        raise RuntimeError(f"raster row count differs: {a.name} vs {b.name}")
                    va = np.fromstring(row_a, sep=" ")
                    vb = np.fromstring(row_b, sep=" ")
                    if va.shape != vb.shape:
                        raise RuntimeError(f"raster row width differs: {a.name} vs {b.name}")
                    if not np.isfinite(va).all() or not np.isfinite(vb).all():
                        raise RuntimeError(f"non-finite values in {a.name} or {b.name}")
                    errors = np.abs(va - vb)
                    if errors.size:
                        maximum = max(maximum, float(errors.max()))
                        squared_error += float(np.dot(errors, errors))
                        value_count += errors.size
            rmse = float(np.sqrt(squared_error / value_count)) if value_count else 0.0
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
    def mass_file(directory):
        candidates = sorted(directory.glob("*.mass"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one .mass file in {directory}, found {len(candidates)}")
        return candidates[0]

    a = read_mass(mass_file(single))
    b = read_mass(mass_file(multi))
    if len(a) != len(b) or not a:
        raise RuntimeError(f"mass row count differs: single={len(a)}, multi={len(b)}")
    max_relative = 0.0
    max_qout_relative = 0.0
    max_residual_scaled = 0.0
    for i, (ra, rb) in enumerate(zip(a, b)):
        if len(ra) != len(rb):
            raise RuntimeError(f"mass column count differs at row {i}")
        for col, (x, y) in enumerate(zip(ra, rb)):
            scale = max(1.0, abs(x), abs(y))
            allowed = tolerance
            if col == 8:  # Qout is a floating-point reduction over the domain.
                allowed = 1e-4
                max_qout_relative = max(max_qout_relative, abs(x-y)/scale)
            if col in (9, 10) and len(ra) > 5:
                # Qerror and Verror are residual diagnostics; compare their
                # reduction noise against stored water volume, not zero.
                scale = max(1.0, abs(ra[5]), abs(rb[5]))
                max_residual_scaled = max(max_residual_scaled, abs(x-y)/scale)
            relative = abs(x-y) / scale
            max_relative = max(max_relative, relative)
            if relative > allowed:
                raise RuntimeError(f"mass mismatch row {i} col {col}: {x} vs {y}")
    print(f"mass rows={len(a)}, maximum scaled difference={max_relative:.9g}, "
          f"Qout relative difference={max_qout_relative:.9g}, "
          f"scaled residual difference={max_residual_scaled:.9g}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: compare_cuda_fv1_mpi.py SINGLE_DIR MULTI_DIR [raster_tol_m]")
    compare_rasters(Path(sys.argv[1]), Path(sys.argv[2]),
                    float(sys.argv[3]) if len(sys.argv) == 4 else 1e-6)
    compare_mass(Path(sys.argv[1]), Path(sys.argv[2]), 1e-6)
    print("CUDA FV1 single/multi-GPU comparison PASSED")
