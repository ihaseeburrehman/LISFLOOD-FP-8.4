"""Is the low Horton infiltration of SERGHEI-SWE explained by its time-based Horton form alone?
Point test without any flow: 200 000 pervious cells of the Alzette basin, each with its own f0, fc, k (the grids given
to both models) and its own radar rainfall (5 min), over the 96 h. Each cell infiltrates min(capacity x dt, rain x dt);
surplus rain runs off (no run-on, no ponding). Two capacities:
  time-based (SERGHEI-SWE):  f(t) = fc + (f0 - fc) exp(-k t), t since the start of the simulation
  cumulative (LISFLOOD-FP):  f = f(t_e), with t_e the equivalent time at which the ponded curve has infiltrated F
Prints the infiltrated depth of both forms, their ratio, and the same for the first 12 h and up to 24 h."""
import numpy as np, pandas as pd, netCDF4
W = "/ec/res4/scratch/lux0804/"
H = W + "lf84_cases/inputs_horton_v3b/"
rd = lambda p: pd.read_csv(p, skiprows=6, sep=r"\s+", header=None, dtype=np.float64, engine="c").values
f0, fc, k = rd(H + "horton_f0_mmh.asc"), rd(H + "horton_fc_mmh.asc"), rd(H + "horton_k_per_h.asc")
dem = rd(W + "radar_scale_2026/inputs/Alzette_sub_basin_10m_bridge_burn.asc")
perv = (dem != -9999) & (f0 > 0)
idx = np.flatnonzero(perv.ravel())
rng = np.random.default_rng(1); s = np.sort(rng.choice(idx, 200000, replace=False))
F0, FC, K = f0.ravel()[s], fc.ravel()[s], np.maximum(k.ravel()[s], 1e-9)       # mm/h, 1/h
r, c = np.unravel_index(s, dem.shape)
nc = netCDF4.Dataset(W + "radar_scale_2026/forcing/rain_R_1km_5min.nc")
ys = np.asarray(nc["y"][:]); flip = ys[0] < ys[-1]
rr = (dem.shape[0] - 1 - r) if flip else r
t_h = np.asarray(nc["time"][:]); dt = float(t_h[1] - t_h[0])                   # h
Fcum = lambda te: FC * te + (F0 - FC) / K * (1 - np.exp(-K * te))
def te_of(F):                                   # invert the ponded curve by Newton iteration
    te = F / np.maximum(F0, 1e-12)
    for _ in range(30):
        g = Fcum(te) - F; d = FC + (F0 - FC) * np.exp(-K * te)
        te = np.maximum(te - g / np.maximum(d, 1e-12), 0.0)
    return te
I_time = np.zeros(len(s)); I_cum = np.zeros(len(s)); rain = np.zeros(len(s)); marks = {}
nsteps = int(round(96 / dt))
for n in range(nsteps):
    p = np.asarray(nc["rainfall_depth"][n, :, :], dtype=np.float64)[rr, c]     # mm in this interval
    t0 = n * dt
    cap_time = FC * dt + (F0 - FC) / K * (np.exp(-K * t0) - np.exp(-K * (t0 + dt)))   # integral of f(t) over the step
    te = te_of(I_cum)
    cap_cum = Fcum(te + dt) - Fcum(te)
    I_time += np.minimum(cap_time, p); I_cum += np.minimum(cap_cum, p); rain += p
    for hh in (12, 17, 24, 28, 96):
        if abs(t0 + dt - hh) < 1e-6: marks[hh] = (I_time.mean(), I_cum.mean(), rain.mean())
print("sample: 200000 pervious cells; median f0 %.2f fc %.2f k %.3f" % (np.median(F0), np.median(FC), np.median(K)))
for hh, (a, b, rn) in sorted(marks.items()):
    print(f"to {hh:2d} h: rain {rn:6.1f} mm | time-based {a:6.2f} mm | cumulative {b:6.2f} mm | ratio {a / b:5.2f}")
