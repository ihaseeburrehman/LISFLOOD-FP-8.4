"""LISFLOOD-FP hourly 1hr-*.Qx/.Qy/.wd -> segflux npz, for a run folder anywhere (extract_case.py takes radar_scale_2026 cases only).
    lf_extract.py RUN_DIR   env TRANSECTS (default transects400.json), OUTNPZ (default /segflux.npz)"""
import sys, os, glob, re, numpy as np
sys.path.insert(0, "/ec/res4/scratch/lux0804/radar_scale_2026/code")
import seg_extract as se
RUN = sys.argv[1]
T = se.load("/ec/res4/scratch/lux0804/radar_scale_2026/code/" + os.environ.get("TRANSECTS", "transects400.json")); need = se.rows_needed(T)
files = sorted(glob.glob(RUN + "/results/1hr-*.Qx"), key=lambda p: int(re.search(r"-(\d+)\.Qx", p).group(1)))
out = {n: dict(qn=[], wd=[]) for n in T}; steps = []
for p in files:
    res = se.seg_flux(T, se.read_rows(p, need), se.read_rows(p[:-3] + ".Qy", need), se.read_rows(p[:-3] + ".wd", need))
    for n in T: out[n]["qn"].append(res[n]["qn"]); out[n]["wd"].append(res[n]["wd"])
    steps.append(int(re.search(r"-(\d+)\.Qx", p).group(1)))
save = {"steps": np.array(steps)}
for n in T:
    save[f"{n}_t"] = res[n]["t"]; save[f"{n}_L"] = res[n]["L"]
    save[f"{n}_qn"] = np.array(out[n]["qn"]); save[f"{n}_wd"] = np.array(out[n]["wd"])
np.savez_compressed(RUN + os.environ.get("OUTNPZ", "/segflux.npz"), **save)
print(RUN, len(steps), "steps extracted")
