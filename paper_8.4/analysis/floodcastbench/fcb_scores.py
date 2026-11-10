#!/usr/bin/env python3
"""Hourly scores of a run against the FloodCastBench reference and, if given,
against the generation scheme without flux clipping (re-run h30_<s>.npy).
Mozambique fields are reduced to the 480 m reference grid with inter()
(bilinear at np.linspace points), as the dataset was; the other events are
compared on the model grid.
    fcb_scores.py EVENT RUN_DIR [NOCLIP_DIR] OUT.csv
Columns: hour, nse, r, csi1cm, csi5cm, rmse, mean depth model and reference
(m, over the compared grid), and nse_nc, csi5cm_nc against the unclipped scheme.
"""
import glob, os, re, sys
import numpy as np
import rasterio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_exact import inter, csi

ev, run = sys.argv[1], sys.argv[2]
nc_dir = sys.argv[3] if len(sys.argv) > 4 else None
out = sys.argv[-1]
fid, res = ("High", "30m") if ev in ("UK", "Australia") else ("Low", "480m")
ref = f"/ec/res4/scratch/lux0804/lf84/floodcastbench/ref/FloodCastBench/{fid}-fidelity flood forecasting/{res}/{ev}"
files = glob.glob(os.path.join(run, 'r-*.wd'))
if files:
    items = [(int(re.findall(r'(\d+)\.wd$', p)[0]) * 3600, p) for p in files]
    load = lambda p: np.loadtxt(p, skiprows=6)
else:
    items = [(int(re.findall(r'_(\d+)\.npy$', p)[0]), p) for p in glob.glob(os.path.join(run, 'h30_*.npy'))]
    load = np.load


def reduce(a, shape):
    a = np.maximum(a.astype(np.float64), 0)
    return np.maximum(inter(a), 0) if a.shape != shape else a


def nse(m, r):
    return 1 - ((m - r) ** 2).sum() / ((r - r.mean()) ** 2).sum()


rows = []
for s, p in sorted(items):
    rp = f"{ref}/{s}.tif"
    if s == 0 or not os.path.exists(rp):
        continue
    r = np.maximum(np.nan_to_num(rasterio.open(rp).read(1).astype(np.float64)), 0)
    m = reduce(load(p), r.shape)
    row = [s / 3600, nse(m, r), np.corrcoef(m.ravel(), r.ravel())[0, 1], csi(m, r, .01), csi(m, r, .05),
           np.sqrt(((m - r) ** 2).mean()), m.mean(), r.mean(), np.nan, np.nan]
    q = f"{nc_dir}/h30_{s}.npy" if nc_dir else None
    if q and os.path.exists(q):
        c = reduce(np.load(q), r.shape)
        row[8:] = [nse(m, c), csi(m, c, .05)]
    rows.append(row)
a = np.array(rows)
np.savetxt(out, a, delimiter=',', fmt='%.6g',
           header='hour,nse,r,csi1cm,csi5cm,rmse,mean_model_m,mean_ref_m,nse_noclip,csi5cm_noclip', comments='')
mn = np.nanmean(a, axis=0)
print(f"{ev} {run}: {len(a)} h  NSE {mn[1]:.3f} r {mn[2]:.3f} CSI1 {mn[3]:.3f} CSI5 {mn[4]:.3f} RMSE {mn[5]:.3f} "
      f"vol {a[-1,6]/a[-1,7]-1:+.3%} | no-clip NSE {mn[8]:.3f} CSI5 {mn[9]:.3f}")
