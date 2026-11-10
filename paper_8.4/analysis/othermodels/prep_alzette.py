"""Alzette 10 m, 96 h (13 Jul 18 UTC to 17 Jul 18 UTC 2021), radar R_1km_5min: the inputs of the
LISFLOOD-FP 8.4 and HiPIMS runs converted for SERGHEI-SWE and TRITON, unchanged in value.

    prep_alzette.py OUT

DEM, Manning n and start depth are the files of the LISFLOOD-FP runs (radar_scale_2026/inputs).
The rain is the LISFLOOD-FP 10 m forcing (forcing/rain_R_1km_5min.nc: depth in mm per 5 min
interval, valid from its time stamp).
  SERGHEI: the same field as a rate in mm/h (CDF-5 NetCDF, read by PnetCDF).
  TRITON (no gridded rain): one runoff zone per distinct rain time series; cells whose 1153 values
  are identical share a zone, so every cell receives exactly its LISFLOOD-FP rain.
Outlet: the LISFLOOD-FP FREE segment (Border.bci: north edge, x 75641 to 75666 m); other edges closed.
"""
import os
import sys

import netCDF4
import numpy as np
import pandas as pd

OUT = sys.argv[1]
W = "/ec/res4/scratch/lux0804/radar_scale_2026"
os.makedirs(OUT, exist_ok=True)


def read_asc(p):
    with open(p) as f:
        hdr = [f.readline() for _ in range(6)]
    a = pd.read_csv(p, skiprows=6, sep=r"\s+", header=None, dtype=np.float64, engine="c").values
    return hdr, a


hdr, dem = read_asc(W + "/inputs/Alzette_sub_basin_10m_bridge_burn.asc")
_, man = read_asc(W + "/inputs/manning.n.ascii")
_, h0 = read_asc(W + "/inputs/13_july_18hr.wd")
H = {l.split()[0].lower(): float(l.split()[1]) for l in hdr}
NR, NC, CS = int(H["nrows"]), int(H["ncols"]), H["cellsize"]
X0, Y0, NOD = H["xllcorner"], H["yllcorner"], H["nodata_value"]
act = dem != NOD
print("grid", NR, NC, CS, X0, Y0, "active cells", int(act.sum()), flush=True)
man = np.where(act, man, 0.035)
h0 = np.where(act, np.maximum(h0, 0.0), 0.0)
print("start volume m3", round(float(h0[act].sum() * CS * CS), 1), flush=True)

# rain: one pass over the frames: the SERGHEI rate file and a 64-bit hash of each cell's series
src = netCDF4.Dataset(W + "/forcing/rain_R_1km_5min.nc")
t_h = np.asarray(src["time"][:], dtype=np.float64)
xs, ys = np.asarray(src["x"][:]), np.asarray(src["y"][:])
rd = src["rainfall_depth"]
NT = len(t_h)
step = np.diff(t_h)
assert np.allclose(step, step[0]), "uneven rain steps"
step = float(step[0])
north_first = ys[0] > ys[-1]
print("rain frames", NT, "step h", step, "x", xs[0], xs[-1], "y", ys[0], ys[-1],
      "north first" if north_first else "south first", flush=True)
assert rd.shape[1:] == (NR, NC), rd.shape
assert abs(xs[0] - (X0 + CS / 2)) < 1e-6, "rain x is not the DEM cell centres"

dst = netCDF4.Dataset(OUT + "/atmforcing.nc", "w", format="NETCDF3_64BIT_DATA")
dst.createDimension("time", None)
dst.createDimension("y", len(ys))
dst.createDimension("x", len(xs))
vt = dst.createVariable("time", "f8", ("time",))
vt.units = "hours"
vx = dst.createVariable("x", "f8", ("x",))
vx.units = "m"
vx[:] = xs
vy = dst.createVariable("y", "f8", ("y",))
vy.units = "m"
vy[:] = ys
vr = dst.createVariable("rain", "f4", ("time", "y", "x"))
vr.units = "mm/h"

act_src = act if north_first else act[::-1]
hsh = np.zeros((NR, NC), np.uint64)
K = np.uint64(1099511628211)
basin_mm = 0.0
for k in range(NT):
    d = np.asarray(rd[k], dtype=np.float32)
    vt[k] = t_h[k]
    vr[k] = d / np.float32(step)
    hsh = (hsh * K) ^ d.view(np.uint32).astype(np.uint64)
    if k < NT - 1:
        basin_mm += float(d[act_src].mean())
    if k % 200 == 0:
        print("frame", k, flush=True)
dst.close()
print("basin mean rain mm, frames 0 to NT-2:", round(basin_mm, 4), "(LISFLOOD-FP json: 82.7217)", flush=True)

hgrid = hsh if north_first else hsh[::-1]
key = np.where(act, hgrid, np.uint64(0))
u, inv = np.unique(key, return_inverse=True)
inv = inv.reshape(NR, NC)
# zone 0 = no rain: NoData cells, and any cell whose series is all zero (its hash stays 0)
zid = np.empty(len(u), np.int64)
is0 = u == 0
zid[is0] = 0
zid[~is0] = np.arange(1, (~is0).sum() + 1) if is0.any() else np.arange(len(u))
zones = zid[inv]
nz = int(zones.max()) + 1
print("runoff zones", nz, flush=True)

flat = zones.ravel()
rep = np.full(nz, -1, np.int64)
rep[flat[::-1]] = np.arange(flat.size)[::-1]          # first cell of each zone
ri, ci = np.unravel_index(rep, (NR, NC))
ri_src = ri if north_first else NR - 1 - ri
series = np.zeros((NT, nz))
for k in range(NT):
    d = np.asarray(rd[k], dtype=np.float64)
    series[k] = d[ri_src, ci] / step
series[:, 0] = 0.0
# check: every cell of a zone has the zone series (sampled frames)
for k in (300, 450, 600):
    d = np.asarray(rd[k], dtype=np.float64)
    d = d if north_first else d[::-1]
    bad = np.abs(d[act] / step - series[k][zones[act]]).max()
    print("frame", k, "largest cell-vs-zone difference mm/h", bad, flush=True)
    assert bad == 0.0

# TRITON
T = OUT + "/triton"
os.makedirs(T, exist_ok=True)
with open(T + "/alzette.dem", "w") as f:
    f.write("".join(hdr))
    np.savetxt(f, dem, fmt="%.4f")
np.savetxt(T + "/alzette.mann", man, fmt="%.4f")
np.savetxt(T + "/alzette.inith", h0, fmt="%.6f")
np.savetxt(T + "/alzette.rmap", zones, fmt="%d")
rows = np.column_stack([t_h, series])
rows = np.vstack([rows, np.concatenate([[t_h[-1] + 1.0], np.zeros(nz)])])
with open(T + "/alzette.roff", "w") as f:
    f.write("% Time(hr) Runoff(mm/hr) per zone\n")
    np.savetxt(f, rows, fmt="%.10g", delimiter=",")
ytop = Y0 + NR * CS
with open(T + "/alzette.extbc", "w") as f:
    f.write("% BC Type, X1, Y1, X2, Y2, BC\n")
    f.write(f"0,75641.0,{ytop},75666.0,{ytop},0\n")

# SERGHEI
S = OUT + "/serghei"
os.makedirs(S, exist_ok=True)
for name, a, fmt in (("dem", dem, "%.4f"), ("roughness", man, "%.4f"), ("hini", h0, "%.6f")):
    with open(f"{S}/{name}.input", "w") as f:
        f.write("".join(hdr))
        np.savetxt(f, a, fmt=fmt)
with open(S + "/polygon.input", "w") as f:
    f.write("NPOINTS 4\n")
    for x, y in ((75636.0, ytop - CS), (75671.0, ytop - CS), (75671.0, ytop + CS), (75636.0, ytop + CS)):
        f.write(f"{x} {y}\n")
np.savez_compressed(OUT + "/zones.npz", zones=zones.astype(np.int32), series=series.astype(np.float32), t_h=t_h)
print("written", OUT, flush=True)
