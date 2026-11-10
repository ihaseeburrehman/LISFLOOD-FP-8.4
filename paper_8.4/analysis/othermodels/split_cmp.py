"""SERGHEI split runs against the 1-GPU run: same grid, same inputs, only parNx x parNy differ.
    split_cmp.py RUN_1G RUN_SPLIT [RUN_SPLIT ...]
A correct domain decomposition gives the same fields (to round-off) and therefore the same time steps."""
import sys, numpy as np, netCDF4
def load(r):
    d = netCDF4.Dataset(r + "/output/output.nc")
    return d, np.asarray(d["t"][:])
d0, t0 = load(sys.argv[1])
NY, NX = d0["h"].shape[1:]
for r in sys.argv[2:]:
    d, t = load(r)
    print(f"== {r}  (grid {NY} x {NX})")
    for name in ("z",):
        a, b = np.asarray(d0[name][:], np.float64), np.asarray(d[name][:], np.float64)
        a, b = a.reshape(-1, NY, NX)[0], b.reshape(-1, NY, NX)[0]
        m = np.isfinite(a) & np.isfinite(b)
        print(f"  z: max |diff| {np.abs(a - b)[m].max():.3e}, NaN pattern equal {np.array_equal(np.isnan(a), np.isnan(b))}")
    for k in range(len(t)):
        i0 = int(np.argmin(np.abs(t0 - t[k])))
        out = []
        for v in ("h", "u", "v"):
            a = np.nan_to_num(np.asarray(d0[v][i0], np.float64)); b = np.nan_to_num(np.asarray(d[v][k], np.float64))
            e = np.abs(a - b); j = np.unravel_index(np.argmax(e), e.shape)
            out.append(f"{v} max|d| {e.max():.3e} at row {j[0]} col {j[1]}")
        b = np.nan_to_num(np.asarray(d["h"][k], np.float64)); u = np.nan_to_num(np.asarray(d["u"][k], np.float64)); vv = np.nan_to_num(np.asarray(d["v"][k], np.float64))
        sp = np.hypot(u, vv) + np.sqrt(9.81 * b); j = np.unravel_index(np.argmax(np.where(b > 1e-3, sp, 0)), sp.shape)
        print(f"  t={t[k]:6.0f}: " + "; ".join(out) + f"; fastest wave {sp[j]:.2f} m/s at row {j[0]} col {j[1]}")
