#!/usr/bin/env python3
"""Build the LISFLOOD-FP inputs of the Halandri benchmark (Bellos and Tsakiris, 2016).

    python3 make_case.py [OUT=case]

From data/ (as distributed by V. Bellos, see data/readme.txt):
  dem.asc       10 m terrain of dtm_modified.csv; cells outside the catchment are NoData,
                so with drain_nodata the water that reaches them leaves the domain and
                the .mass Qout column is the outflow of the catchment
  manning.asc   friction zones A/B/C: n = 0.070 / 0.020 / 0.025
  rain.txt      uniform rain of Palea Penteli, 15 min depths as a stepwise rate (mm/h);
                t = 0 at 22 Feb 2013 05:15, the start of the FLOW-R2D simulation
  obs.csv       measured outlet discharge (m3/s) and the FLOW-R2D results, t in s
"""
import csv
import datetime as dt
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "case")
T0 = dt.datetime(2013, 2, 22, 5, 15)
MANNING = {1: 0.070, 2: 0.020, 3: 0.025}
os.makedirs(OUT, exist_ok=True)

a = np.loadtxt(os.path.join(DATA, "dtm_modified.csv"), delimiter=",")
x, y, z, zone = a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(int)
dx = 10.0
xs, ys = np.unique(x), np.unique(y)
nx, ny = len(xs), len(ys)
col = np.rint((x - xs[0]) / dx).astype(int)
row = np.rint((ys[-1] - y) / dx).astype(int)            # row 0 = north
dem = np.full((ny, nx), -9999.0)
n = np.full((ny, nx), -9999.0)
inside = zone > 0
dem[row[inside], col[inside]] = z[inside]
n[row[inside], col[inside]] = [MANNING[k] for k in zone[inside]]
hdr = (f"ncols {nx}\nnrows {ny}\nxllcorner {xs[0] - dx / 2:.3f}\nyllcorner {ys[0] - dx / 2:.3f}\n"
       f"cellsize {dx:g}\nNODATA_value -9999\n")
for name, g, fmt in (("dem.asc", dem, "%.3f"), ("manning.asc", n, "%.3f")):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(hdr)
        np.savetxt(f, g, fmt=fmt)


def when(s):
    for f in ("%d/%m/%Y %I:%M %p", "%d/%m/%Y %I:%M:%S %p"):
        try:
            return dt.datetime.strptime(s.strip(), f)
        except ValueError:
            pass
    raise ValueError(s)


meas = [(when(r[0]), float(r[1]), float(r[2])) for r in list(csv.reader(open(os.path.join(DATA, "measurements.csv"))))[1:]]
# a depth stamped at time T fell during the 15 min ending at T
steps = [((t - T0).total_seconds() - 900, d * 4.0) for t, d, _ in meas if (t - T0).total_seconds() > 0]
pts = []
for t0, rate in steps:
    pts += [(rate, t0), (rate, t0 + 899.0)]
pts.append((0.0, steps[-1][0] + 900.0))
with open(os.path.join(OUT, "rain.txt"), "w") as f:
    f.write(f"Palea Penteli, 22 Feb 2013, 15 min depths as mm/h, t=0 at {T0:%H:%M}\n{len(pts)} seconds\n")
    f.writelines(f"{r:.4f} {t:.0f}\n" for r, t in pts)

sim = list(csv.reader(open(os.path.join(DATA, "flowr2d.csv"))))[1:]
with open(os.path.join(OUT, "obs.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kind", "t_s", "Q"])
    for t, _, q in meas:
        w.writerow(["observed", (t - T0).total_seconds(), q])
    for r in sim:
        if all(r[1:4]):
            t = (when(r[0]) - T0).total_seconds()
            for k, i in (("flowr2d_kostiakov", 1), ("hybrid_kostiakov", 2), ("hybrid_greenampt", 3)):
                w.writerow([k, t, float(r[i])])

area = inside.sum() * dx * dx
rain = sum(d for t, d, _ in meas if (t - T0).total_seconds() > 0)
print(f"{nx} x {ny} cells of {dx:g} m, {inside.sum()} in the catchment ({area / 1e6:.2f} km2); "
      f"rain {rain:.1f} mm from {T0:%H:%M} ({rain / 1000 * area / 1e6:.3f} Mm3); written to {OUT}")
