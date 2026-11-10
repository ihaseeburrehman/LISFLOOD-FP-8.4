"""Alzette July 2021, radar rainfall, 96 h: LISFLOOD-FP 8.4, HiPIMS, TRITON and SERGHEI-SWE scored at the four gauges.
    alzette_models.py OUTDIR
Discharge is the integral of q.n over the full wetted valley at every gauge (transects_fullvalley.json), as in Table 6
of the paper. Observations: obs_hourly.csv, clock "utc1" (records in UTC+1 shifted to UTC). Writes
    scores_models.csv       peak ratio, peak timing, bias, r, NSE, KGE and the mean of hours 1-12 per run and gauge
    hydrographs_models.csv  hourly discharge per run and gauge
    water_balance_models.csv rain, infiltration, outflow and storage (10^6 m3) where the model reports them"""
import csv, os, sys
import numpy as np
S = "/ec/res4/scratch/lux0804/"
O = S + "othermodels/runs/"
G = ["Livange", "Hesperange", "Pfaffenthal", "Steinsel"]
RUNS = [("LISFLOOD-FP", "none", S + "radar_scale_2026/runs/L84_R_1km_5min_noinf_dp_1g/segflux_fv.npz"),
        ("LISFLOOD-FP", "Green-Ampt", S + "radar_scale_2026/runs/L84_R_1km_5min_ga3b_dp/segflux_fv.npz"),
        ("LISFLOOD-FP", "Horton", S + "othermodels/cmp/lf_horton/segflux_fv.npz"),
        ("HiPIMS", "none", S + "hipims/runs/R_1km_5min_mg4/segflux_fv.npz"),
        ("HiPIMS", "Green-Ampt", S + "hipims/runs/R_1km_5min_inf_mg4/segflux_fv.npz"),
        ("TRITON", "none", O + "noinf96_triton_drain_4g/segflux_fv.npz"),
        ("SERGHEI-SWE", "none", O + "noinf96_serghei_crit_1g/segflux_fv.npz"),
        ("SERGHEI-SWE", "Horton", O + "horton96_serghei_crit_1g/segflux_fv.npz")]
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

obs = {}
for r in csv.DictReader(open(S + "hipims/obs_hourly.csv")):
    if r["clock"] == "utc1":
        obs.setdefault(r["gauge"], {})[int(r["h"])] = float(r["Q"])


def q_valley(z, g):
    return dict(zip(z["steps"].tolist(), (z[g + "_qn"] * z[g + "_L"]).sum(1)))


def score(qs, qo):
    h = [k for k in range(1, 97) if k in qs and not np.isnan(qo.get(k, np.nan))]
    s = np.array([qs[k] for k in h]); o = np.array([qo[k] for k in h])
    r = np.corrcoef(s, o)[0, 1]; beta = s.mean() / o.mean(); gam = (s.std() / s.mean()) / (o.std() / o.mean())
    return dict(peak=s.max(), peak_obs=o.max(), peak_ratio=s.max() / o.max(),
                t_peak=h[int(s.argmax())], dt_peak=h[int(s.argmax())] - h[int(o.argmax())],
                beta=beta, r=r, gamma=gam, nse=1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum(),
                kge=1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gam - 1) ** 2),
                q_h1_12=np.mean([qs[k] for k in range(1, 13)]), q_obs_h1_12=np.mean([qo[k] for k in range(1, 13)]))


rows, hyd = [], []
for m, inf, p in RUNS:
    z = np.load(p)
    for g in G:
        q = q_valley(z, g)
        rows.append(dict(model=m, infiltration=inf, gauge=g, **{k: round(float(v), 4) for k, v in score(q, obs[g]).items()}))
        hyd += [(m, inf, g, k, round(float(v), 3)) for k, v in sorted(q.items())]
with open(OUT + "/scores_models.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(OUT + "/hydrographs_models.csv", "w") as f:
    w = csv.writer(f); w.writerow(["model", "infiltration", "gauge", "h", "Q"]); w.writerows(hyd)
with open(OUT + "/obs_utc.csv", "w") as f:
    w = csv.writer(f); w.writerow(["gauge", "h", "Q"]); w.writerows((g, k, obs[g][k]) for g in G for k in sorted(obs[g]))


# water balance, 10^6 m3
def lf_mass(run):
    """Initial and final storage, integrated outflow and the last column (rain minus losses, cumulative) of a .mass file."""
    a = np.loadtxt(run + "/results/1hr.mass", skiprows=1)
    dt = np.diff(np.r_[0, a[:, 0]])
    return a[0, 5], a[-1, 5], (a[:, 8] * dt).sum(), a[-1, 11]


A = S + "lf84/paper84/alzette/"                # the final 8.4 build (double precision)
_, _, _, RAIN = lf_mass(A + "L84_R_1km_5min_noinf_dpn_1g")   # no losses: the last column is the rain itself
wb = []
for inf, run in (("none", "L84_R_1km_5min_noinf_dpn_1g"), ("Green-Ampt", "L84_R_1km_5min_ga3b_dpn"),
                 ("Horton", "TEST_horton_R_1km_5min_fitGA24_4g")):
    v0, v1, qo, net = lf_mass(A + run)
    i = RAIN - net
    wb.append(dict(model="LISFLOOD-FP", infiltration=inf, rain=RAIN, infiltrated=i, outflow=qo, storage0=v0, storage1=v1,
                   residual=v0 + RAIN - i - qo - v1))
for inf, run in (("none", "noinf96_serghei_crit_1g"), ("Horton", "horton96_serghei_crit_1g")):
    # the first row (t = 0) carries no boundary columns; the full rows start at the second output
    lines = [l.split() for l in open(O + run + "/output/domainTimeSeries.out").readlines()[1:]]
    s0 = float(lines[0][1])
    a = np.array([[float(x) for x in l] for l in lines if len(l) >= 10])
    i = a[-1, 11] if a.shape[1] > 11 else 0.0
    wb.append(dict(model="SERGHEI-SWE", infiltration=inf, rain=a[-1, 9], infiltrated=i, outflow=a[-1, 5],
                   storage0=s0, storage1=a[-1, 1], residual=s0 + a[-1, 9] - i - a[-1, 5] - a[-1, 1]))
# TRITON writes no balance: storage from the depth grids, outflow as the closing term
B = O + "noinf96_triton_drain_4g/output/bin/"
# TRITON writes no grid at t = 0; its start depth is the common start file (storage s0 of SERGHEI-SWE, same file)
# TRITON drains through its NoData cells, which act as sinks: storage counts the cells with data only
import pandas as pd
dem = pd.read_csv(S + "radar_scale_2026/inputs/Alzette_sub_basin_10m_bridge_burn.asc", skiprows=6, sep=r"\s+",
                  header=None, dtype=np.float64, engine="c").values
h96 = np.fromfile(B + "H_96_00.out", np.float64)[2:].reshape(dem.shape)
st = [s0, h96[dem != -9999].sum() * 100.0]
wb.append(dict(model="TRITON", infiltration="none", rain=RAIN, infiltrated=0.0, outflow=st[0] + RAIN - st[1],
               storage0=st[0], storage1=st[1], residual=np.nan))
with open(OUT + "/water_balance_models.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(wb[0])); w.writeheader()
    for r in wb:
        w.writerow({k: (round(v / 1e6, 3) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})

# summary on stdout
print(f"{'model':12s} {'inf':10s} " + " ".join(f"{g[:5]:>12s}" for g in G) + "   mean PR  NSE   KGE")
for m, inf, _ in RUNS:
    rr = [r for r in rows if r["model"] == m and r["infiltration"] == inf]
    cells = " ".join(f"{r['peak_ratio']:5.2f}/{r['nse']:5.2f}" for r in rr)
    print(f"{m:12s} {inf:10s} {cells}   {np.mean([r['peak_ratio'] for r in rr]):5.2f} {np.mean([r['nse'] for r in rr]):5.2f} "
          f"{np.mean([r['kge'] for r in rr]):5.2f}")
for r in wb:
    print(r["model"], r["infiltration"], {k: round(v / 1e6, 2) for k, v in r.items() if isinstance(v, (float, np.floating))})
