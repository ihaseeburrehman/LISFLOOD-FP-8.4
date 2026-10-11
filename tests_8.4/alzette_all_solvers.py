#!/usr/bin/env python3
"""Run every GPU solver for six hours on the 100 m Alzette basin.

The test writes hourly rasters for no loss, uniform infiltration, evaporation,
and Green-Ampt.  It is intentionally sequential because all runs share one
GPU.

Usage:
  alzette_all_solvers.py BINARY CASE_DIR WORK [metal|cuda] [options]

Options: --solvers a,b,...  only these solvers;
         --epsilon E        adaptive threshold for hwfv1/mwdg2/acc_nugrid
                            (default 0: the finest mesh, as the original matrix);
         --timeout S        seconds per run (default 1800).
"""
import csv
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


SOLVERS = {
    "fv1": "fv1\n",
    "fv2": "fv2\n",
    "dg2": "dg2\n",
    "acc": "acceleration\n",
    "acc_nugrid": "acc_nugrid\nL 9\nepsilon 0\n",
    "hwfv1": "hwfv1\nmax_ref_lvl 9\nepsilon 0\nwall_height 10000\n",
    "mwdg2": "mwdg2\nmax_ref_lvl 9\nepsilon 0\nwall_height 10000\n",
}
LOSSES = {
    "off": "",
    "uniform": "infiltration 0.00000055555556\n",
    "evaporation": "evaporation evap.dat\n",
    "greenampt": (
        "infiltration greenampt\ngreenampt_ks 10\ngreenampt_psi 110\n"
        "greenampt_porosity 0.43\ngreenampt_theta_init 0.30\n"
    ),
}


def copy_inputs(case, work):
    data = work / "data"
    data.mkdir(parents=True, exist_ok=True)
    source = case / "data"
    names = {
        "dem_100m.asc": "dem.asc",
        "start_100m.wd": "start.wd",
        "manning_100m.asc": "manning.asc",
    }
    for old, new in names.items():
        dst = data / new
        if not dst.exists():
            shutil.copy2(source / old, dst)
    rain = data / "rain.nc"
    if not rain.exists():
        rain.symlink_to((source / "rain_100m.nc").resolve())
    shutil.copy2(case / "Border.bci", work / "Border.bci")
    (work / "stage.txt").write_text("0\n")
    (work / "evap.dat").write_text("Evaporation 3 mm/day\n2 days\n3 0\n3 10\n")

    # A common piecewise-constant modal projection makes CUDA and Metal DG2
    # start from exactly the same input without modifying the reference case.
    for base in (data / "dem.asc", data / "start.wd"):
        lines = base.read_text().splitlines()
        ncols = int(lines[0].split()[1]); nrows = int(lines[1].split()[1])
        zero = "\n".join(lines[:6]) + "\n" + (" ".join(["0"] * ncols) + "\n") * nrows
        for suffix in ("1x", "1y"):
            modal = Path(str(base) + suffix)
            if not modal.exists():
                modal.write_text(zero)


def raster_values(path):
    words = path.read_text().split()
    return [float(x) for x in words[12:]]


def vtk_depth_values(path):
    """Read the ACC_NUGRID cell-depth array without loading its large mesh."""
    values = []
    remaining = 0
    in_depth = False
    with path.open() as stream:
        for line in stream:
            if line.startswith("CELL_DATA "):
                remaining = int(line.split()[1])
            elif line.startswith("SCALARS h "):
                in_depth = True
            elif in_depth and line.startswith("LOOKUP_TABLE"):
                continue
            elif in_depth:
                values.extend(float(word) for word in line.split())
                if len(values) >= remaining:
                    return values[:remaining]
    return values


def output_grids(out, backend, solver):
    if backend == "cuda" and solver == "acc_nugrid":
        return sorted(out.glob("r-*.vtk"))
    return sorted(out.glob("r-*.wd"))


OPTIONS = {"solvers": None, "epsilon": None, "timeout": 1800}


def run(binary, work, backend, solver, loss):
    label = f"{backend}_{solver}_{loss}"
    out = work / label
    grids = output_grids(out, backend, solver) if out.exists() else []
    if len(grids) >= 7:
        return out, grids, 0.0, "cached"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    settings = (
        "DEMfile data/dem.asc\nstartfile data/start.wd\n"
        "manningfile data/manning.asc\ndynamicrainfile data/rain.nc\n"
        "bcifile Border.bci\nstagefile stage.txt\nresroot r\n"
        f"dirroot {label}\n{backend}\n{solver_lines(solver)}"
        "sim_time 21600\ninitial_tstep 10\nmassint 3600\nsaveint 3600\n"
        "nodata_elevation -9999\ndrain_nodata\nqoutput\noutput_precision 9\n"
        + LOSSES[loss]
    )
    if backend == "cuda" and solver in ("hwfv1", "mwdg2"):
        settings += "raster_out\n"
    par = work / f"{label}.par"
    par.write_text(settings)
    start = time.monotonic()
    with (out / "run.log").open("w") as log:
        result = subprocess.run([str(binary), str(par)], cwd=work,
                                stdout=log, stderr=subprocess.STDOUT,
                                timeout=OPTIONS["timeout"])
    elapsed = time.monotonic() - start
    grids = output_grids(out, backend, solver)
    return out, grids, elapsed, f"rc={result.returncode}"


def solver_lines(solver):
    lines = SOLVERS[solver]
    if OPTIONS["epsilon"] is not None:
        lines = lines.replace("epsilon 0\n", f"epsilon {OPTIONS['epsilon']}\n")
    return lines


def main():
    args = sys.argv[1:]
    for key in ("solvers", "epsilon", "timeout"):
        if "--" + key in args:
            i = args.index("--" + key)
            value = args[i + 1]
            OPTIONS[key] = (value.split(",") if key == "solvers" else
                            value if key == "epsilon" else int(value))
            del args[i:i + 2]
    binary, case, work = map(lambda p: Path(p).resolve(), args[0:3])
    backend = args[3] if len(args) > 3 else "metal"
    work.mkdir(parents=True, exist_ok=True)
    copy_inputs(case, work)
    rows = []
    failures = []
    for solver in OPTIONS["solvers"] or SOLVERS:
        baseline = None
        for loss in LOSSES:
            out, grids, elapsed, status = run(binary, work, backend, solver, loss)
            if grids and grids[-1].suffix == ".vtk":
                final = vtk_depth_values(grids[-1])
            else:
                final = raster_values(grids[-1]) if grids else []
            finite = bool(final) and all(math.isfinite(v) for v in final)
            physical = [v for v in final if v > -9000] if finite else []
            volume = sum(max(0.0, v) for v in physical) * 10000.0
            minimum = min(physical, default=math.nan)
            maximum = max(physical, default=math.nan)
            negative_cells = sum(v < -1e-6 for v in physical)
            if loss == "off":
                baseline = volume
            removed = baseline - volume if baseline is not None else math.nan
            loss_is_active = loss == "off" or removed > 0
            ok = (status in ("rc=0", "cached") and len(grids) == 7 and finite
                  and loss_is_active)
            print(("PASS " if ok else "FAIL ") +
                  f"{backend} {solver} {loss}: {status}, hourly={len(grids)}, "
                  f"runtime={elapsed:.2f}s, volume={volume:.6g}, "
                  f"removed={removed:.6g}, min={minimum:.6g}, "
                  f"max={maximum:.6g}, negatives={negative_cells}",
                  flush=True)
            if not ok:
                failures.append(f"{solver}/{loss}")
            rows.append({"backend": backend, "solver": solver, "loss": loss,
                         "runtime_s": f"{elapsed:.6f}", "hourly_grids": len(grids),
                         "finite": finite, "min_depth_m": f"{minimum:.9g}",
                         "max_depth_m": f"{maximum:.9g}",
                         "negative_cells": negative_cells,
                         "water_volume_m3": f"{volume:.9g}",
                         "less_than_off_m3": f"{removed:.9g}", "status": status})
    with (work / f"summary_{backend}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"{len(failures)} failures; summary: {work / ('summary_' + backend + '.csv')}")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
