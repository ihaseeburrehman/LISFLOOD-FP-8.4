"""TRITON hourly H/QX/QY binary rasters -> segflux.npz in the format of the LISFLOOD-FP and HiPIMS extractions.
    triton_extract.py RUN_DIR SIGN_Y [HOURS]   (HOURS: comma list; default all; with HOURS nothing is saved)
TRITON files hold rows, cols then the grid in row order from the top (north) row; QX, QY are unit discharges."""
import sys, glob, re, os, numpy as np
sys.path.insert(0, "/ec/res4/scratch/lux0804/radar_scale_2026/code")
import seg_extract as se
RUN, SIGN_Y = sys.argv[1], float(sys.argv[2])
B = RUN + "/output/bin/"
T = se.load("/ec/res4/scratch/lux0804/radar_scale_2026/code/" + os.environ.get("TRANSECTS", "transects400.json")); need = se.rows_needed(T)
def grid(name, k):
    a = np.fromfile(f"{B}{name}_{k:02d}_00.out", np.float64)
    return a[2:].reshape(int(a[0]), int(a[1]))
def one(k):
    h, qx, qy = grid("H", k), grid("QX", k), SIGN_Y * grid("QY", k)
    return se.seg_flux(T, {r: qx[r] for r in need}, {r: qy[r] for r in need}, {r: h[r] for r in need})
hours = sorted(int(re.search(r"H_(\d+)_00", p).group(1)) for p in glob.glob(B + "H_*_00.out"))
if len(sys.argv) > 3:
    for k in map(int, sys.argv[3].split(",")):
        r = one(k); print(k, {n: round(float((r[n]["qn"] * r[n]["L"]).sum()), 2) for n in T})
    sys.exit()
import multiprocessing as mp
with mp.Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as pool: RES = pool.map(one, hours)
save = {"steps": np.array(hours)}
for n in T:
    save[f"{n}_t"] = RES[-1][n]["t"]; save[f"{n}_L"] = RES[-1][n]["L"]
    save[f"{n}_qn"] = np.array([r[n]["qn"] for r in RES]); save[f"{n}_wd"] = np.array([r[n]["wd"] for r in RES])
np.savez_compressed(RUN + os.environ.get("OUTNPZ", "/segflux.npz"), **save)
print(RUN, len(hours), "hours extracted")
