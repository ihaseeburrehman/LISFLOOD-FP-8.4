#!/usr/bin/env python3
"""Horton ponded-infiltration test with outputs every 5 min (Fig. 1c, d).

    horton_series.py BUILD_DIR WORK OUT_CSV [cuda|metal]

The case of tests_8.4/losses/horton_free_test.py run for 2 h: a closed flat
box of 32 x 32 cells (10 m), 0.2 m of water at rest, no rain, Horton with
f0 = 60 mm/h, fc = 10 mm/h, k = 2 /h. The water never runs out, so the depth
infiltrated is F(t) = fc t + (f0 - fc)/k (1 - exp(-k t)). Writes the mean
depth of every saved grid for FV1, FV2, DG2 and ACC.
"""
import csv
import pathlib
import subprocess
import sys

N, DX, H0 = 32, 10.0, 0.2
F0, FC, K = 60.0, 10.0, 2.0
T_END, SAVE = 7200.0, 300.0
SOLVERS = {"fv1": "fv1\n", "fv2": "fv2\n", "dg2": "dg2\n", "acc": "acceleration\n"}


def grid(path, v):
    rows = "\n".join(" ".join("%.6f" % v for _ in range(N)) for _ in range(N))
    path.write_text(f"ncols {N}\nnrows {N}\nxllcorner 0\nyllcorner 0\ncellsize {DX:g}\nNODATA_value -9999\n{rows}\n")


def main():
    build, work, out = (pathlib.Path(a).resolve() for a in sys.argv[1:4])
    backend = sys.argv[4] if len(sys.argv) > 4 else "cuda"
    work.mkdir(parents=True, exist_ok=True)
    grid(work / "dem.asc", 10.0)
    grid(work / "start.wd", H0)
    for s in ("1x", "1y"):
        grid(work / f"start.wd{s}", 0.0)
    (work / "gen.par").write_text("DEMfile dem.asc\nnodata_elevation -9999\n")
    subprocess.run([str(build / "generateDG2DEM"), "gen.par"], cwd=work, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    rows = []
    for s, sel in SOLVERS.items():
        par = (f"DEMfile dem.asc\nresroot r\ndirroot {s}\n{backend}\n{sel}fpfric 0.03\n"
               f"massint 300\nsaveint {SAVE:g}\noutput_precision 9\nstartfile start.wd\n"
               f"sim_time {T_END + 60:g}\ninfiltration horton\nhorton_f0 {F0}\n"
               f"horton_fc {FC}\nhorton_k {K}\n")
        (work / f"{s}.par").write_text(par)
        (work / s).mkdir(exist_ok=True)
        with open(work / s / "run.log", "w") as log:
            subprocess.run([str(build / "lisflood"), f"{s}.par"], cwd=work, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=900)
        for p in sorted((work / s).glob("r-*.wd")):
            k = int(p.stem.split("-")[-1])
            h = [float(w) for w in p.read_text().split()[12:]]
            rows.append((backend, s, k * SAVE, sum(h) / len(h)))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "solver", "t_s", "mean_depth_m"])
        w.writerows(rows)
    print(f"{len(rows)} rows written to {out}")


if __name__ == "__main__":
    main()
