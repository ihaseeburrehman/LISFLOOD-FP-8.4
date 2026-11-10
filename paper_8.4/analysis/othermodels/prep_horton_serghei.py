"""SERGHEI Horton inputs from the LISFLOOD-FP 8.4 Horton maps (f0, fc in mm/h, k in 1/h, one set per cell).
SERGHEI takes Horton parameters per class (16-bit labels, at most 65535) in mm/s and 1/s, so the cell
parameter sets are rounded to SIG significant figures and each distinct rounded set becomes one class.
    prep_horton_serghei.py OUT"""
import os, sys
import numpy as np, pandas as pd
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
I = "/ec/res4/scratch/lux0804/lf84_cases/inputs_horton_v3b/"
DEM = "/ec/res4/scratch/lux0804/radar_scale_2026/inputs/Alzette_sub_basin_10m_bridge_burn.asc"
rd = lambda p: pd.read_csv(p, skiprows=6, sep=r"\s+", header=None, dtype=np.float64, engine="c").values
with open(DEM) as f: hdr = [f.readline() for _ in range(6)]
act = rd(DEM) != -9999.0
f0, fc, k = rd(I + "horton_f0_mmh.asc"), rd(I + "horton_fc_mmh.asc"), rd(I + "horton_k_per_h.asc")
act &= f0 > -9000
P = np.stack([f0[act], fc[act], k[act]], 1)
imperv = (P[:, 0] <= 0) & (P[:, 1] <= 0)
def sig(x, n):
    with np.errstate(divide="ignore"):
        e = np.where(x > 0, np.floor(np.log10(np.abs(x))), 0)
    return np.round(x / 10 ** (e - n + 1)) * 10 ** (e - n + 1)
for n in (4, 3, 2):
    Q = np.stack([sig(P[:, j], n) for j in range(3)], 1)
    u, inv = np.unique(Q[~imperv], axis=0, return_inverse=True)
    if len(u) <= 65534: break
print("significant figures", n, "classes", len(u), "cells", int(act.sum()), "impervious", int(imperv.sum()))
rel = np.abs(Q[~imperv] - P[~imperv]) / np.maximum(P[~imperv], 1e-12)
print("largest relative rounding error f0 %.2e fc %.2e k %.2e" % tuple(rel.max(0)))
lab = np.zeros(act.shape, np.int64); L = np.zeros(int(act.sum()), np.int64); L[~imperv] = inv.ravel() + 1; lab[act] = L
with open(OUT + "/infiltrationMap.input", "w") as f:
    f.write("".join(hdr)); np.savetxt(f, lab, fmt="%d")
with open(OUT + "/infiltration.input", "w") as f:
    f.write("InfModel : horton\nspatialInput : raster\ninfiltrationClasses : %d\n" % len(u))
    for a, b, c in u:   # SERGHEI units: mm/s, mm/s, 1/s (its clock t is in s)
        f.write("k : %.10g\nfc : %.10g\nf0 : %.10g\n" % (c / 3600.0, b / 3600.0, a / 3600.0))
print("written", OUT)
