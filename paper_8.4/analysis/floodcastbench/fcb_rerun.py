#!/usr/bin/env python3
"""Re-run the FloodCastBench data-generation solver (HydroPML/FloodCastBench,
Data_Generation_Code, commit c27d108) to find out how the reference was made.

    fcb_rerun.py EVENT TOP BOTTOM LEFT RIGHT HOURS CLIP OUT [F]

Follows main.py: dem[TOP:-BOTTOM, LEFT:-RIGHT], missing DEM -> max + 30,
Manning from land cover (Table 3), rain frame int(t / 1800) in mm/h, dry
start, closed edges (zero-padded fluxes), dt = 0.7 dx / (g + max h) cut at
30 s marks. CLIP 1 keeps main.py's F.threshold on the fluxes (only positive
fluxes survive), CLIP 0 drops it. F = 16 runs on inter()-resampled inputs
(Pakistan); F = 1 runs at 30 m. Saves depth every hour, at the model grid and
resampled to 480 m with inter() (bilinear at linspace points).
"""
import os, sys, glob, time
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, '/ec/res4/scratch/lux0804/lf84/floodcastbench/fcbcode/Data_Generation_Code')
from hydraulics import saint_venant

ev = sys.argv[1]
top, bot, left, right, hours, clip = map(int, sys.argv[2:8])
out = sys.argv[8]
FAC = int(sys.argv[9]) if len(sys.argv) > 9 else 1
os.makedirs(out, exist_ok=True)
SRC = '/ec/res4/scratch/lux0804/lf84/floodcastbench/raw/FloodCastBench/Relevant data/'
dev = torch.device('cuda')
g = torch.tensor(9.80616, dtype=torch.float64)


def crop(a):
    return a[top:a.shape[0] - bot, left:a.shape[1] - right]


def inter(a, size):
    h, w = a.shape
    nh, nw = np.floor_divide((h, w), size)
    f = RegularGridInterpolator((np.arange(h), np.arange(w)), a, method='linear')
    R, C = np.meshgrid(np.linspace(0, h - 1, nh), np.linspace(0, w - 1, nw), indexing='ij')
    return f((R, C))


dem = crop(rasterio.open(f"{SRC}DEM/{ev}_DEM.tif").read(1).astype(np.float64))
m = dem < -2000
dem[m] = dem[~m].max() + 30
lulc = crop(rasterio.open(f"{SRC}Land use and land cover/{ev}.tif").read(1))
MANNING = {1: 0.035, 2: 0.12, 4: 0.08, 5: 0.035, 7: 0.375, 8: 0.0265, 11: 0.0375}
man = np.full(lulc.shape, 0.035)
for c, v in MANNING.items():
    man[lulc == c] = v
dx = 30.0
if FAC > 1:
    dem, man, dx = inter(dem, FAC), inter(man, FAC), 30.0 * FAC
print(f"{ev}: grid {dem.shape} at {dx:g} m, crop [{top}:-{bot}, {left}:-{right}], clip {clip}", flush=True)
z = torch.from_numpy(dem).float().to(dev)[None, None]
n = torch.from_numpy(man).float().to(dev)[None, None]
frames = sorted(os.listdir(f"{SRC}Rainfall/{ev} flood"))


def rain(k):
    a = crop(rasterio.open(f"{SRC}Rainfall/{ev} flood/{frames[k]}").read(1).astype(np.float64))
    if FAC > 1:
        a = inter(a, FAC)
    a[a < 0] = 0
    return torch.from_numpy(a).float().to(dev)[None, None]


solver = saint_venant.SaintVenantFlux(dx, 0.7).cuda()
h = torch.zeros_like(z)
if os.environ.get('START') == 'ref':                # start from reference frame 0 (UK, Australia: end of the pre-run)
    res = '30m' if ev in ('UK', 'Australia') else '480m'
    fid = 'High' if res == '30m' else 'Low'
    a0 = np.maximum(np.nan_to_num(rasterio.open(f"/ec/res4/scratch/lux0804/lf84/floodcastbench/ref/FloodCastBench/{fid}-fidelity flood forecasting/{res}/{ev}/0.tif").read(1).astype(np.float64)), 0)
    assert a0.shape == dem.shape, (a0.shape, dem.shape)
    h = torch.from_numpy(a0).float().to(dev)[None, None]
h0 = h.clone()
qx = torch.zeros(1, 1, z.shape[2], z.shape[3] - 1, device=dev)
qy = torch.zeros(1, 1, z.shape[2] - 1, z.shape[3], device=dev)
t, nn_, kr, R = 0.0, 1, -1, None
t_end, t0 = hours * 3600.0, time.time()
nxt = 3600.0
while t < t_end - 0.5:
    k = int(t / 1800)
    if k != kr:
        R, kr = rain(k), k
    mh = max(float(h.max()), 0.01)
    dt = min(abs(30.0 * nn_ - t), 0.7 * dx / (9.80616 + mh))
    h, qx, qy = solver(z, h, qx, qy, dt, R, n, h0)
    h = F.threshold(h, 0, 0)
    if clip:
        qx = F.threshold(qx, 0, 0); qy = F.threshold(qy, 0, 0)
    t += dt
    if abs(t - 30.0 * nn_) < 1e-6:
        nn_ += 1
    if t >= nxt - 1e-6:
        a = h[0, 0].double().cpu().numpy()
        np.save(f"{out}/h30_{int(round(nxt))}.npy", a.astype(np.float32))
        b = a if FAC > 1 else inter(a, 16)
        np.save(f"{out}/h480_{int(round(nxt))}.npy", b.astype(np.float32))
        print(f"t {nxt/3600:6.1f} h  vol {a.sum()*dx*dx:.4g} m3  max {a.max():.2f} m  wall {time.time()-t0:.0f} s", flush=True)
        nxt += 3600.0
