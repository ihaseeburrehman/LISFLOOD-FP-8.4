#!/usr/bin/env python3
"""A FloodCastBench event as a LISFLOOD-FP case, set up as the dataset's
generation code (HydroPML/FloodCastBench, Data_Generation_Code, c27d108):

- domain: dem[top:-bottom, left:-right] of the 30 m DEM (CROP below)
- missing DEM cells raised to max + 30 m (utils.fix_missing_values)
- Manning's n from land cover (Table 3 of Xu et al. 2025)
- rain: the 30 min frames as given, frame int(t / 1800) held for 30 min
- Pakistan runs at 480 m on inter(., 16) of DEM, Manning and rain (bilinear at
  np.linspace points); the others at 30 m, Mozambique reduced to 480 m by
  inter() for comparison (compare_exact.py)
- start: dry for Mozambique (as main.py), reference frame 0 otherwise
- closed edges (zero-padded fluxes)
    make_case_exact.py EVENT OUT
"""
import glob, os, sys
import numpy as np
import rasterio
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

EV, OUT = sys.argv[1], sys.argv[2]
CROP = {'UK': (5, 5, 5, 5), 'Australia': (0, 0, 0, 0), 'Mozambique': (150, 50, 50, 50), 'Pakistan': (37, 32, 170, 58)}[EV]
F = 16 if EV == 'Pakistan' else 1
RB = {'UK': 1, 'Australia': 8, 'Mozambique': 8, 'Pakistan': 1}[EV]      # model cells per rain cell
SRC = 'raw/FloodCastBench/Relevant data/'
fid, res = ('High', '30m') if EV in ('UK', 'Australia') else ('Low', '480m')
REF = f"ref/FloodCastBench/{fid}-fidelity flood forecasting/{res}/{EV}"
os.makedirs(OUT, exist_ok=True)
T, B, L, R = CROP
with rasterio.open(f"{SRC}DEM/{EV}_DEM.tif") as s:
    dem = s.read(1).astype(np.float64); tr = s.transform
win = (slice(T, dem.shape[0] - B), slice(L, dem.shape[1] - R))


def inter(a, size=16):
    h, w = a.shape
    nh, nw = np.floor_divide((h, w), size)
    f = RegularGridInterpolator((np.arange(h), np.arange(w)), a, method='linear')
    rr, cc = np.meshgrid(np.linspace(0, h - 1, nh), np.linspace(0, w - 1, nw), indexing='ij')
    return f((rr, cc))


dem = dem[win]
m = dem < -2000
dem[m] = dem[~m].max() + 30
H30, W30 = dem.shape
if F > 1:
    dem = inter(dem)
h, w = dem.shape
dx = 30.0 * F                                          # as the generation code (inter() points are ~0.2 % wider apart)
x0, ytop = tr.c + L * 30.0, tr.f - T * 30.0


def write_asc(name, a, fmt):
    with open(os.path.join(OUT, name), 'w') as f:
        f.write(f"ncols {w}\nnrows {h}\nxllcorner {x0:.3f}\nyllcorner {ytop - h * dx:.3f}\n"
                f"cellsize {dx:.6f}\nNODATA_value -9999\n")
        np.savetxt(f, a, fmt=fmt)


write_asc('dem.asc', dem, '%.3f')
lulc = rasterio.open(f"{SRC}Land use and land cover/{EV}.tif").read(1)[win]
n = np.full(lulc.shape, 0.035)
for c, v in {1: 0.035, 2: 0.12, 4: 0.08, 5: 0.035, 7: 0.375, 8: 0.0265, 11: 0.0375}.items():
    n[lulc == c] = v
write_asc('manning.asc', inter(n) if F > 1 else n, '%.4f')
if EV == 'Mozambique':
    start = np.zeros((h, w))
else:
    start = np.maximum(np.nan_to_num(rasterio.open(f"{REF}/0.tif").read(1).astype(np.float64)), 0)
    assert start.shape == (h, w), (start.shape, (h, w))
write_asc('start.wd', start, '%.5f')
open(os.path.join(OUT, 'free.bci'), 'w').close()

frames = sorted(glob.glob(f"{SRC}Rainfall/{EV} flood/*.tif"))
ny, nx = -(-h // RB), -(-w // RB)


def frame(p):
    a = rasterio.open(p).read(1).astype(np.float64)[win]
    a[(a < 0) | ~np.isfinite(a)] = 0
    if F > 1:
        a = np.maximum(inter(a), 0)
    pad = np.pad(a, ((0, ny * RB - h), (0, nx * RB - w)), mode='edge')
    return pad.reshape(ny, RB, nx, RB).mean(axis=(1, 3))


sim = max(float(os.path.basename(p)[:-4]) for p in glob.glob(f"{REF}/*.tif"))   # frames named by seconds
nf = int(round(sim / 1800))
assert nf <= len(frames), (nf, len(frames))
nc = Dataset(os.path.join(OUT, 'rain.nc'), 'w')
nc.createDimension('time', nf + 1); nc.createDimension('x', nx); nc.createDimension('y', ny)
t = nc.createVariable('time', 'f4', ('time',)); t.units = 'hour'; t.axis = 'T'
xs = nc.createVariable('x', 'f8', ('x',)); xs.units = 'm'; xs.axis = 'X'
ys = nc.createVariable('y', 'f8', ('y',)); ys.units = 'm'; ys.axis = 'Y'
rd = nc.createVariable('rainfall_depth', 'f4', ('time', 'y', 'x'), zlib=True, complevel=1); rd.units = 'mm'
t[:] = 0.5 * np.arange(nf + 1)
xs[:] = x0 + RB * dx * (np.arange(nx) + 0.5)
ys[:] = ytop - RB * dx * (np.arange(ny) + 0.5)
tot = 0.0
for k in range(nf):
    d = 0.5 * frame(frames[k])                            # mm/h held for 0.5 h
    rd[k] = d.astype(np.float32); tot += d.mean()
rd[nf] = np.zeros((ny, nx), np.float32)
nc.close()
open(os.path.join(OUT, 'case.txt'), 'w').write(
    f"{EV} as the FloodCastBench generation code: crop [{T}:-{B}, {L}:-{R}] of the 30 m DEM, "
    f"{w} x {h} cells of {dx:.3f} m, {m.sum()} missing DEM cells raised to max + 30 m, "
    f"{nf} rain frames of 30 min (domain mean {tot:.1f} mm), sim_time {sim:g} s, "
    f"start {'dry' if EV == 'Mozambique' else 'reference frame 0'}, closed edges\n")
print(open(os.path.join(OUT, 'case.txt')).read())
