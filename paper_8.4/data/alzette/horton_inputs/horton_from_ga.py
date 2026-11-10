#!/usr/bin/env python3
"""Horton grids equivalent to the Green-Ampt soil of the Alzette case.

    horton_from_ga.py GA_DIR OUT_DIR [T_HOURS=24]

Same soil, same cells: fc = Ks (the long-time rate of both models), and f0, k
are fitted per cell so that Horton's ponded cumulative infiltration
    F_H(t) = fc t + (f0 - fc)/k (1 - exp(-k t))
matches Green-Ampt's,  Ks t = F - psi dtheta ln(1 + F/(psi dtheta)),
in least squares over 0..T. Both models in 8.4 follow cumulative F (not wall
time), so matching the ponded curves makes them the same soil in two forms.

In time tau = psi dtheta / Ks and depth psi dtheta the fit depends only on
X = T / tau, so it is done once on a table of X and interpolated per cell.
Ks = 0 cells get f0 = fc = 0; cells with dtheta <= 0 (saturated) f0 = fc = Ks.
Writes horton_f0_mmh.asc, horton_fc_mmh.asc (mm/h), horton_k_per_h.asc (1/h).
"""
import os, sys
import numpy as np
from scipy.optimize import least_squares

GA, OUT = sys.argv[1], sys.argv[2]
T = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
os.makedirs(OUT, exist_ok=True)


def read(name):
    with open(os.path.join(GA, name)) as f:
        hdr = [f.readline() for _ in range(6)]
    return hdr, np.loadtxt(os.path.join(GA, name), skiprows=6)


def ga_F(ts):
    """Dimensionless Green-Ampt ponded F*(t*): t* = F* - ln(1 + F*) (Newton)."""
    F = np.sqrt(2 * ts) + ts                       # good start for small and large t*
    for _ in range(60):
        g = F - np.log1p(F) - ts
        F = np.maximum(F - g / (F / (1 + F)), 1e-12)
    return F


def fit(X):
    ts = np.linspace(0, X, 400)[1:]
    Fg = ga_F(ts)
    def res(p):
        a, b = np.exp(p)                            # a = f0/Ks - 1 > 0, b = k tau > 0
        return (ts + a / b * (1 - np.exp(-b * ts)) - Fg) / Fg.max()
    p = least_squares(res, [np.log(1 / np.sqrt(X)), np.log(3 / X)]).x
    return np.exp(p)


Xs = np.logspace(-4, 4, 161)
tab = np.array([fit(X) for X in Xs])               # columns a, b
hdr, ks = read('ga_ks_mmh.asc')
_, psi = read('ga_psi_mm.asc')
_, por = read('ga_porosity.asc')
_, th0 = read('ga_theta_init.asc')
nod = (ks < -9000) | (psi < -9000) | (por < -9000) | (th0 < -9000)
dth = np.where(nod, 0.0, por - th0)
pervious = ~nod & (ks > 0) & (dth > 0) & (psi > 0)
fc = np.where(nod, -9999.0, np.maximum(ks, 0.0))
f0 = fc.copy()
k = np.where(nod, -9999.0, 1.0)
S = psi * dth                                      # mm
tau = np.where(pervious, S / np.where(ks > 0, ks, 1.0), 1.0)   # h
X = np.clip(T / tau, Xs[0], Xs[-1])
a = np.exp(np.interp(np.log(X), np.log(Xs), np.log(tab[:, 0])))
b = np.exp(np.interp(np.log(X), np.log(Xs), np.log(tab[:, 1])))
f0 = np.where(pervious, ks * (1 + a), f0)
k = np.where(pervious, b / tau, k)


def write(name, arr, fmt):
    with open(os.path.join(OUT, name), 'w') as f:
        f.writelines(hdr)
        np.savetxt(f, arr, fmt=fmt)


write('horton_fc_mmh.asc', fc, '%.6g')
write('horton_f0_mmh.asc', f0, '%.6g')
write('horton_k_per_h.asc', k, '%.6g')

# check: cumulative infiltration after 1, 6, 24 h of ponding, basin mean over pervious cells
p = pervious
for t in (1, 6, 24, 96):
    Fg = S[p] * ga_F(t / tau[p])
    Fh = ks[p] * t + (f0[p] - ks[p]) / k[p] * (1 - np.exp(-k[p] * t))
    print(f"ponded {t:3d} h: Green-Ampt F {Fg.mean():7.2f} mm, Horton F {Fh.mean():7.2f} mm")
with open(os.path.join(OUT, 'README.txt'), 'w') as f:
    f.write(__doc__ + f"\nGA source: {GA}\nT = {T:g} h\n"
            f"cells: {(~nod).sum()} valid, {p.sum()} pervious\n"
            f"pervious means: Ks {ks[p].mean():.3f} mm/h, psi*dtheta {S[p].mean():.1f} mm, "
            f"tau {np.median(tau[p]):.1f} h (median)\n"
            f"Horton pervious: f0 mean {f0[p].mean():.2f} mm/h (median {np.median(f0[p]):.2f}), "
            f"fc mean {fc[p].mean():.3f} mm/h, k median {np.median(k[p]):.3f} /h\n")
print(open(os.path.join(OUT, 'README.txt')).read().split('GA source')[1])
