#!/usr/bin/env python3
"""Score a 30 m run (LISFLOOD r-NNNN.wd hourly, or re-run h30_<s>.npy) against
the 480 m FloodCastBench reference, reduced as the generation code did:
inter(h, 16) = bilinear at np.linspace(0, n - 1, n // 16) on the cropped grid.
    compare_exact.py RUN_DIR EVENT [CSV]
"""
import glob, os, re, sys
import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator

run, ev = (sys.argv[1:3] + [None, None])[:2]
csv = sys.argv[3] if len(sys.argv) > 3 else None
fid, res = ("High", "30m") if ev in ("UK", "Australia") else ("Low", "480m")
ref = f"/ec/res4/scratch/lux0804/lf84/floodcastbench/ref/FloodCastBench/{fid}-fidelity flood forecasting/{res}/{ev}"


def inter(a, size=16):
    h, w = a.shape
    nh, nw = np.floor_divide((h, w), size)
    f = RegularGridInterpolator((np.arange(h), np.arange(w)), a, method='linear')
    R, C = np.meshgrid(np.linspace(0, h - 1, nh), np.linspace(0, w - 1, nw), indexing='ij')
    return f((R, C))


def csi(a, b, t):
    A, B = a > t, b > t
    return (A & B).sum() / max((A | B).sum(), 1)


def main():
    files = glob.glob(os.path.join(run, 'r-*.wd'))
    if files:
        items = [(int(re.findall(r'(\d+)\.wd$', p)[0]) * 3600, p) for p in files]
        load = lambda p: np.loadtxt(p, skiprows=6)
    else:
        items = [(int(re.findall(r'_(\d+)\.npy$', p)[0]), p) for p in glob.glob(os.path.join(run, 'h30_*.npy'))]
        load = np.load
    rows = []
    for s, p in sorted(items):
        rp = f"{ref}/{s}.tif"
        if not os.path.exists(rp):
            continue
        r = np.maximum(np.nan_to_num(rasterio.open(rp).read(1).astype(np.float64)), 0)
        m = np.maximum(load(p).astype(np.float64), 0)
        if m.shape != r.shape:
            m = np.maximum(inter(m), 0)
        assert m.shape == r.shape, (m.shape, r.shape)
        e = m - r
        rows.append((s / 3600, np.sqrt((e ** 2).mean()), 1 - (e ** 2).sum() / ((r - r.mean()) ** 2).sum(),
                     np.corrcoef(m.ravel(), r.ravel())[0, 1], csi(m, r, .01), csi(m, r, .05), m.sum() / r.sum()))
        print("%5.0f h  RMSE %.3f  NSE %6.3f  r %.3f  CSI1cm %.3f  CSI5cm %.3f  vol %.3f" % rows[-1], flush=True)
    a = np.array(rows)
    if len(a):
        print(f"mean over {len(a)} frames: RMSE {a[:,1].mean():.3f} NSE {a[:,2].mean():.3f} r {a[:,3].mean():.3f} "
              f"CSI1cm {a[:,4].mean():.3f} CSI5cm {a[:,5].mean():.3f}; last volume ratio {a[-1,6]:.3f}")
        if csv:
            np.savetxt(csv, a, delimiter=',', fmt='%.6g', header='hour,rmse,nse,r,csi1cm,csi5cm,vol_ratio', comments='')


if __name__ == '__main__':
    main()
