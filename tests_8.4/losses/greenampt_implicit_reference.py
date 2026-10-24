#!/usr/bin/env python3
"""Compare a ponded CUDA Green-Ampt run with its continuous equation.

Run after a case with the parameters in docs/DEVELOPMENT_LOG.md has produced
ga.mass and ga-NNNN.wd in WORK. This is independent of LISFLOOD's stepper.
"""
import math
import re
import sys
from pathlib import Path


KS = 10.9 / 1000.0 / 3600.0  # Rawls sandy-loam value, m/s
PSI = 110.1 / 1000.0         # m
DELTA_THETA = 0.412 - 0.041
H0 = 0.09                    # initial ponded depth, m
TOL_M = 2.0e-5               # 0.02 mm


def infiltration_at(time_s):
    """Exact ponded GA solution while h=H0-F remains positive."""
    b = 1.0 - DELTA_THETA
    a = DELTA_THETA * (PSI + H0)
    c = a / b

    def time_for(f):
        return (f - c * math.log1p(f / c)) / (KS * b)

    if time_s >= time_for(H0):
        return H0
    lo, hi = 0.0, H0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if time_for(mid) < time_s:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def grid_values(path):
    lines = path.read_text().splitlines()
    return [float(value) for line in lines[6:] for value in line.split()]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: greenampt_implicit_reference.py WORK")
    work = Path(sys.argv[1]).resolve()
    mass = work / "ga.mass"
    if not mass.is_file():
        raise SystemExit(f"missing {mass}")
    log = work / "run.log"
    if not log.is_file() or "GPU enabled" not in log.read_text() or "Using CUDA device" not in log.read_text():
        raise SystemExit("run.log does not confirm CUDA GPU execution")

    rows = []
    for line in mass.read_text().splitlines():
        columns = line.split()
        if columns and re.fullmatch(r"\d+(?:\.\d+)?", columns[0]):
            rows.append(float(columns[0]))
    if len(rows) != 11:
        raise SystemExit(f"expected initial row plus ten hourly outputs, found {len(rows)} rows")

    errors = []
    hourly_errors = []
    print("time_s, mean_depth_m, exact_depth_m, error_mm")
    for index, time_s in enumerate(rows):
        path = work / f"ga-{index:04d}.wd"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        values = grid_values(path)
        if len(values) != 100:
            raise SystemExit(f"{path}: expected 100 cells, found {len(values)}")
        depth = sum(values) / len(values)
        if max(values) - min(values) > 1.0e-7:
            raise SystemExit(f"{path}: flat-basin depths are not spatially uniform")
        reference = max(0.0, H0 - infiltration_at(time_s))
        error_mm = 1000.0 * (depth - reference)
        errors.append(abs(error_mm))
        if time_s >= 3600.0:
            hourly_errors.append(abs(error_mm))
        print(f"{time_s:.3f}, {depth:.9f}, {reference:.9f}, {error_mm:.6f}")

    worst = max(errors, default=float("inf"))
    worst_hourly = max(hourly_errors, default=float("inf"))
    b = 1.0 - DELTA_THETA
    c = DELTA_THETA * (PSI + H0) / b
    dry_time_s = (H0 - c * math.log1p(H0 / c)) / (KS * b)
    print(f"max_abs_error_all_outputs_mm={worst:.6f}")
    print(f"max_abs_error_hourly_outputs_mm={worst_hourly:.6f}")
    print(f"continuous_solution_dry_time_h={dry_time_s / 3600.0:.6f}")
    if worst_hourly > 1000.0 * TOL_M:
        raise SystemExit(f"FAIL: hourly error exceeds {1000.0 * TOL_M:.3f} mm")
    print("PASS")


if __name__ == "__main__":
    main()
