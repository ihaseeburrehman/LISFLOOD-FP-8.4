"""Where the water is stored at 96 h: SERGHEI, TRITON and LISFLOOD-FP (no infiltration), by distance to the NoData edge."""
import numpy as np, pandas as pd, netCDF4
from scipy import ndimage
W = "/ec/res4/scratch/lux0804/"
rd = lambda p: pd.read_csv(p, skiprows=6, sep=r"\s+", header=None, dtype=np.float64, engine="c").values
dem = rd(W + "radar_scale_2026/inputs/Alzette_sub_basin_10m_bridge_burn.asc"); act = dem != -9999
lf = rd(W + "radar_scale_2026/runs/L84_R_1km_5min_noinf_dp_1g/results/1hr-0096.wd")
d = netCDF4.Dataset(W + "othermodels/runs/noinf96_serghei_fi_1g/output/output.nc")
sg = np.nan_to_num(np.asarray(d["h"][-1], np.float64))
if d["y"][0] < d["y"][-1]: sg = sg[::-1]
a = np.fromfile(W + "othermodels/runs/noinf96_triton_drain_4g/output/bin/H_96_00.out", np.float64); tr = a[2:].reshape(int(a[0]), int(a[1]))
dist = ndimage.distance_transform_edt(act) * 10.0
for nm, h in (("LISFLOOD-FP", lf), ("SERGHEI", sg), ("TRITON", tr)):
    v = lambda m: h[act & m].sum() * 100 / 1e6
    print(f"{nm:12s} total {v(np.ones_like(act)):6.2f} Mm3 | within 50 m of NoData edge {v(dist <= 50):6.2f} | 50-500 m {v((dist > 50) & (dist <= 500)):6.2f} | >500 m {v(dist > 500):6.2f} | max depth {h[act].max():.1f} m")
dif = sg - lf; j = np.unravel_index(np.argmax(dif), dif.shape)
print("largest SERGHEI excess", round(float(dif[j]), 2), "m at row", j[0], "col", j[1], "dist to edge", dist[j], "m")
big = act & (dif > 0.5); lab, n = ndimage.label(big); s = ndimage.sum(dif * 100, lab, range(1, n + 1)) / 1e6
o = np.argsort(s)[::-1][:6]
for k in o:
    rr, cc = np.where(lab == k + 1); print(f"  pond {k+1}: {s[k]:.2f} Mm3 extra, {len(rr)} cells, rows {rr.min()}-{rr.max()} cols {cc.min()}-{cc.max()}, min dist to edge {dist[rr, cc].min():.0f} m")
