"""SERGHEI rain forcing as in prep_alzette.py, coordinates shifted by (-5, +5) m so samples fall at cell centres."""
import netCDF4, numpy as np
W = "/ec/res4/scratch/lux0804/radar_scale_2026"
OUT = "/ec/res4/scratch/lux0804/othermodels/cases/alzette10m/atmforcing_c.nc"
src = netCDF4.Dataset(W + "/forcing/rain_R_1km_5min.nc")
t = np.asarray(src["time"][:], np.float64); xs = np.asarray(src["x"][:]); ys = np.asarray(src["y"][:])
rd = src["rainfall_depth"]; step = float(t[1] - t[0])
dst = netCDF4.Dataset(OUT + ".tmp", "w", format="NETCDF3_64BIT_DATA")
dst.createDimension("time", None); dst.createDimension("y", len(ys)); dst.createDimension("x", len(xs))
vt = dst.createVariable("time", "f8", ("time",)); vt.units = "hours"
vx = dst.createVariable("x", "f8", ("x",)); vx.units = "m"; vx[:] = xs - 5.0
vy = dst.createVariable("y", "f8", ("y",)); vy.units = "m"; vy[:] = ys + 5.0
vr = dst.createVariable("rain", "f4", ("time", "y", "x")); vr.units = "mm/h"
for k in range(len(t)):
    vt[k] = t[k]; vr[k] = np.asarray(rd[k], np.float32) / np.float32(step)
    if k % 200 == 0: print("frame", k, flush=True)
dst.close()
import os; os.replace(OUT + ".tmp", OUT); print("written", OUT, xs[0] - 5.0, ys[0] + 5.0, flush=True)
