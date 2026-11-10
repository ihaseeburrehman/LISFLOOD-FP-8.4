"""Soil statistics of the Alzette basin (cells with data only) for the paper: share of pervious soil, mean Ks, median
f0 and kH of Horton, and the mean ponded infiltration of Green-Ampt and Horton after 6 and 24 h over the pervious cells."""
import numpy as np, pandas as pd
W = "/ec/res4/scratch/lux0804/"
rd = lambda p: pd.read_csv(p, skiprows=6, sep=r"\s+", header=None, dtype=np.float64, engine="c").values
act = rd(W + "radar_scale_2026/inputs/Alzette_sub_basin_10m_bridge_burn.asc") != -9999
G = W + "lf84_cases/inputs_v3b/"; H = W + "lf84_cases/inputs_horton_v3b/"
ks, psi, por, th = rd(G + "ga_ks_mmh.asc"), rd(G + "ga_psi_mm.asc"), rd(G + "ga_porosity.asc"), rd(G + "ga_theta_init.asc")
f0, fc, k = rd(H + "horton_f0_mmh.asc"), rd(H + "horton_fc_mmh.asc"), rd(H + "horton_k_per_h.asc")
p = act & (ks > 0)
print("cells with data", int(act.sum()), "pervious", int(p.sum()), f"share {100 * p.sum() / act.sum():.1f} %")
print(f"mean Ks {ks[p].mean():.3f} mm/h, median Ks {np.median(ks[p]):.3f}; median f0 {np.median(f0[p]):.2f} mm/h, "
      f"median fc {np.median(fc[p]):.3f}, median k {np.median(k[p]):.3f} 1/h")
K, M = ks[p], psi[p] * np.maximum(por[p] - th[p], 0.0)
def ga(t):        # ponded Green-Ampt (no ponding depth): Ks t = F - M ln(1 + F/M), Newton
    F = np.sqrt(2 * K * M * t) + K * t
    for _ in range(60):
        g = F - M * np.log1p(F / np.maximum(M, 1e-12)) - K * t
        d = 1 - M / np.maximum(M + F, 1e-12)
        F = np.maximum(F - g / np.maximum(d, 1e-9), 0)
    return F
hor = lambda t: fc[p] * t + (f0[p] - fc[p]) / np.maximum(k[p], 1e-12) * (1 - np.exp(-k[p] * t))
for t in (6, 24):
    print(f"ponded, mean over pervious cells after {t} h: Green-Ampt {ga(t).mean():.1f} mm, Horton {hor(t).mean():.1f} mm")
