#!/usr/bin/env python3
"""Run one Halandri simulation and score its outlet discharge against the measurements.

    python3 run.py BUILD NAME SOLVER BACKEND [KEY=VALUE ...]

SOLVER   fv1 | acc;  BACKEND  metal | cuda
KEY=VALUE adds parameter-file lines, e.g. infiltration=horton horton_f0=60 ...
Writes runs/NAME/ and prints one line: NSE, peak ratio, peak timing, volume ratio,
infiltrated volume and wall time. Discharge is the .mass Qout (water draining into the
NoData cells around the catchment) at the 15 min stamps of the measurements, plus the
measured base flow before the storm (0.16 m3/s), which the model does not contain.
"""
import csv
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, "case")
T_END = 32400.0                    # 05:15 to 14:15
BASEFLOW = 0.16
SEL = {"fv1": "fv1", "acc": "acceleration"}


def observed():
    t, q = [], []
    for r in csv.DictReader(open(os.path.join(CASE, "obs.csv"))):
        if r["kind"] == "observed" and 0 < float(r["t_s"]) <= T_END:
            t.append(float(r["t_s"])); q.append(float(r["Q"]))
    return np.array(t), np.array(q)


def score(t_model, q_model):
    t, o = observed()
    s = np.interp(t, t_model, q_model) + BASEFLOW
    nse = 1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    return dict(nse=nse, peak_ratio=s.max() / o.max(), dt_peak=(t[s.argmax()] - t[o.argmax()]) / 60,
                vol_ratio=np.trapz(s, t) / np.trapz(o, t))


def run(build, name, solver, backend, extra):
    out = os.path.join(HERE, "runs", name)
    os.makedirs(out, exist_ok=True)
    par = os.path.join(out, "run.par")
    lines = [f"DEMfile {CASE}/dem.asc", f"manningfile {CASE}/manning.asc", f"rainfall {CASE}/rain.txt",
             f"dirroot {out}", "resroot r", backend, SEL[solver], f"sim_time {T_END:g}",
             "saveint 3600", "massint 60", "nodata_elevation -9999", "drain_nodata"]
    lines += [kv.replace("=", " ", 1) for kv in extra]
    open(par, "w").write("\n".join(lines) + "\n")
    t0 = time.time()
    with open(os.path.join(out, "run.log"), "w") as log:
        subprocess.run([os.path.join(build, "lisflood"), par], cwd=out, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    wall = time.time() - t0
    m = np.loadtxt(os.path.join(out, "r.mass"), skiprows=1)
    tm, qout, inf = m[:, 0], m[:, 8], -m[-1, -1]
    s = score(np.r_[0.0, tm], np.r_[0.0, qout])
    s.update(name=name, inf_m3=inf, wall_s=wall, stored_m3=m[-1, 5])
    with open(os.path.join(out, "score.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s)); w.writeheader(); w.writerow(s)
    print(f"{name:28s} NSE {s['nse']:6.3f}  peak {s['peak_ratio']:5.2f}  dt {s['dt_peak']:+4.0f} min  "
          f"vol {s['vol_ratio']:5.2f}  inf {inf / 1e3:7.1f} km3  wall {wall:5.1f} s", flush=True)
    return s


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
