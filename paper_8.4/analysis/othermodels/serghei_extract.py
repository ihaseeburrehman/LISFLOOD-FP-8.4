"""SERGHEI-SWE hourly output.nc (h, u, v) -> segflux.npz in the format of the LISFLOOD-FP and HiPIMS extractions.
    serghei_extract.py RUN_DIR SIGN_Y [HOURS]   (HOURS: comma list; default all; with HOURS nothing is saved)
Unit discharges are h*u and h*v; NoData cells (NaN) count as dry. Rows are turned to north first."""
import sys, os, numpy as np, netCDF4
sys.path.insert(0, "/ec/res4/scratch/lux0804/radar_scale_2026/code")
import seg_extract as se
RUN, SIGN_Y = sys.argv[1], float(sys.argv[2])
T = se.load("/ec/res4/scratch/lux0804/radar_scale_2026/code/" + os.environ.get("TRANSECTS", "transects400.json")); need = sorted(se.rows_needed(T))
nc = netCDF4.Dataset(RUN + "/output/output.nc")
t = np.asarray(nc["t"][:]); NY = len(nc["y"]); south_first = nc["y"][0] < nc["y"][-1]
assert nc["h"].dimensions[1:] == ("y", "x"), nc["h"].dimensions
src = [NY - 1 - r if south_first else r for r in need]
def one(i):
    g = {v: np.nan_to_num(np.asarray(nc[v][i, src, :], dtype=np.float64)) for v in ("h", "u", "v")}
    h, qx, qy = g["h"], g["h"] * g["u"], SIGN_Y * g["h"] * g["v"]
    return se.seg_flux(T, {r: qx[j] for j, r in enumerate(need)}, {r: qy[j] for j, r in enumerate(need)},
                       {r: h[j] for j, r in enumerate(need)})
hours = [int(round(x / 3600)) for x in t]
if len(sys.argv) > 3:
    for k in map(int, sys.argv[3].split(",")):
        r = one(hours.index(k)); print(k, {n: round(float((r[n]["qn"] * r[n]["L"]).sum()), 2) for n in T})
    sys.exit()
RES = [one(i) for i in range(len(hours))]
save = {"steps": np.array(hours)}
for n in T:
    save[f"{n}_t"] = RES[-1][n]["t"]; save[f"{n}_L"] = RES[-1][n]["L"]
    save[f"{n}_qn"] = np.array([r[n]["qn"] for r in RES]); save[f"{n}_wd"] = np.array([r[n]["wd"] for r in RES])
np.savez_compressed(RUN + os.environ.get("OUTNPZ", "/segflux.npz"), **save)
print(RUN, len(hours), "hours extracted")
