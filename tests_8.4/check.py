#!/usr/bin/env python3
"""Check the 8.4 tests run by run_tests.sh (Python 3, no packages).

    check.py WORK

Prints one line per check, PASS or FAIL, and ends with ALL PASSED or the
number of failures.
"""
import bisect
import filecmp
import glob
import os
import sys

WORK = os.path.abspath(sys.argv[1])
OUT = os.path.join(WORK, "out")
failures = 0


def report(ok, text):
    global failures
    failures += 0 if ok else 1
    print(("PASS  " if ok else "FAIL  ") + text)


def grid(path):
    with open(path) as f:
        for _ in range(6):
            f.readline()
        return [[float(v) for v in row.split()] for row in f if row.strip()]


def wd(run, k):
    p = glob.glob(os.path.join(OUT, run, f"*-{k:04d}.wd"))
    return grid(p[0]) if p else None


def mass(run):
    """rows of the .mass file as lists of floats"""
    p = glob.glob(os.path.join(OUT, run, "*.mass"))
    if not p:
        return []
    return [[float(v) for v in l.split()] for l in open(p[0]).readlines()[1:] if l.strip()]


TOL = 2e-6       # m: two units of the last written digit (output_precision 6)
THRESH = 0.001   # m: DepthThresh; shallower cells lose nothing (CPU rule)


def tol_for(run, h0, expected):
    """FV1 and ACC remove exactly rate x dt; FV2 and DG2 remove it too, but
    their closed walls let about 0.1 % of it back in while the level falls
    (README_8.4.md), so they are checked to 0.2 % of the expected loss"""
    if "fv2" in run or "dg2" in run:
        return max(TOL, 2e-3 * (h0 - expected))
    return TOL


def steps(run):
    """(t, dt) of every time step, from the run.log of a run with -v"""
    out = []
    for l in open(os.path.join(OUT, run, "run.log")):
        if l.startswith("t="):
            a, b = l.split()
            out.append((float(a[2:]), float(b[3:])))
    return out


def exact_depth(run, k, h0, rate, save_interval, fv1):
    """depth after the step at which output k is written, from the steps the
    model took: sum of rate x dt. (The model's clock is single precision and
    falls behind the sum of the steps on long runs, so the analytic
    h0 - rate x t at the clock time is not the right reference.) The CUDA
    FV1 solver evaluates the rate at the end of a step, ACC at its start."""
    st = steps(run)
    ends = [t + dt for t, dt in st]
    i = min(bisect.bisect_left(ends, k * save_interval - 1e-3), len(st) - 1)
    lost = sum(rate(t + dt if fv1 else t) * dt for t, dt in st[:i + 1])
    return h0 - lost


def check_depth(run, k, expected, tol, cols=None, label=""):
    g = wd(run, k)
    if g is None:
        report(False, f"{run}: output {k} missing")
        return
    vals = [row[i] for row in g for i in (cols or range(len(row)))]
    err = max(abs(v - expected) for v in vals)
    report(err <= tol, f"{run}: depth at output {k}{label} = {expected:.4f} m "
                       f"(largest error {1000 * err:.4f} mm)")


def check_dry(run, k, cols=None, label=""):
    g = wd(run, k)
    if g is None:
        report(False, f"{run}: output {k} missing")
        return
    vals = [row[i] for row in g for i in (cols or range(len(row)))]
    report(all(0.0 <= v <= THRESH for v in vals),
           f"{run}: depth at output {k}{label} between 0 and the {1000 * THRESH:g} mm "
           f"threshold (range {1000 * min(vals):.4f} to {1000 * max(vals):.4f} mm)")


def grid_volume(run, k, area=270.0 * 270.0):
    g = wd(run, k)
    return None if g is None else sum(map(sum, g)) * area


def start_volume(start_file, area=270.0 * 270.0):
    return sum(map(sum, grid(os.path.join(WORK, "t012", start_file)))) * area


def same_files(a, b, pattern="*"):
    fa = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, a, pattern))
                if not p.endswith("run.log"))
    fb = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, b, pattern))
                if not p.endswith("run.log"))
    if not fa or fa != fb:
        return False, "different file lists"
    diff = [f for f in fa if not filecmp.cmp(os.path.join(OUT, a, f), os.path.join(OUT, b, f), shallow=False)]
    return not diff, (f"{len(fa)} files identical" if not diff else "differ: " + ", ".join(diff[:5]))


def check_mass(run, label, start_file, verror=True):
    """removed volume (last .mass column) against the depth grids; Verror
    after the first row (the GPU's first row always shows the whole volume,
    as in 8.3; FV1's .mass volume includes the ghost cells, so its Verror is
    only reported)"""
    m = mass(run)
    v0, v2 = start_volume(start_file), grid_volume(run, 2)
    if not m or v0 is None or v2 is None:
        report(False, f"{run}: .mass or outputs missing")
        return
    lost, gone = -m[-1][11], v0 - v2
    verr = max(abs(r[10]) for r in m[1:])
    ok = abs(lost - gone) <= 1e-3 * gone and (not verror or verr <= 1e-3 * gone)
    report(ok, f"{run}: {label}volume removed {lost:.1f} m3, depth grids lost {gone:.1f} m3, "
               f"largest |Verror| after the first row {verr:.3g} m3"
               + ("" if verror else " (not checked: FV1 .mass volume includes ghost cells)"))


# ---------------------------------------------------------------- T012
H0 = 0.4464
T50, T100 = 4320000.0, 8640000.0
RATE = 3.6 / 1000 / 86400 + 1e-8          # evaporation + infiltration, m/s
START, WALL = "ETInfTest_start.ascii", "wall_start.ascii"
if glob.glob(os.path.join(OUT, "t012_*")):
    print("== T012: flat basin, 0.4464 m of water, 100 days")
    print("   (expected depth: h0 minus the sum of rate x dt over the steps the model took)")
    r = "t012_base_cpu_acc_84"   # CPU reference: its own clock, 1e-4 m tolerance
    check_depth(r, 1, H0 - RATE * T50, 1e-4, label=" (50 days)")
    check_dry(r, 2, label=" (100 days)")
    check_mass(r, "", START)
    GPU = [s for s in ("cuda_acc", "cuda_fv1", "cuda_fv2", "cuda_dg2")
           if os.path.isdir(os.path.join(OUT, f"t012_base_{s}_84"))]
    for s in GPU:
        r = f"t012_base_{s}_84"
        # FV1, FV2, DG2 apply the loss at the end of a step, ACC at its start;
        # their .mass volume includes the ghost cells (Verror not checked)
        end = (s != "cuda_acc")
        e = exact_depth(r, 1, H0, lambda t: RATE, T50, end)
        check_depth(r, 1, e, tol_for(r, H0, e), label=" (50 days)")
        check_dry(r, 2, label=" (100 days)")
        check_mass(r, "", START, verror=not end)
    for s in ["cpu_acc"] + GPU:
        if os.path.isdir(os.path.join(OUT, f"t012_off_{s}_84")):
            check_depth(f"t012_off_{s}_84", 2, H0, TOL, label=" (no losses, 100 days)")
    for s in GPU:
        if os.path.isdir(os.path.join(OUT, f"t012_off_{s}_83")):
            ok, msg = same_files(f"t012_off_{s}_84", f"t012_off_{s}_83")
            report(ok, f"t012 off, {s}: 8.4 vs 8.3: {msg}")
        if not os.path.isdir(os.path.join(OUT, f"t012_ramp_{s}_84")):
            continue
        fv1 = (s != "cuda_acc")
        r = f"t012_ramp_{s}_84"   # evaporation 0 -> 7.2 mm/day (about 90 and 360 mm lost)
        ramp = lambda t: 7.2 / 1000 / 86400 * t / T100
        for k, label in ((1, " (ramp, 50 days)"), (2, " (ramp, 100 days)")):
            e = exact_depth(r, k, H0, ramp, T50, fv1)
            check_depth(r, k, e, tol_for(r, H0, e), label=label)
        check_mass(r, "ramp: ", START, verror=not fv1)
        r = f"t012_wall_{s}_84"   # 0.3 mm/h on the left half only
        left = 0.3 / 1000 / 3600
        e = exact_depth(r, 1, H0, lambda t: left, T50, fv1)
        check_depth(r, 1, e, tol_for(r, H0, e), cols=range(4), label=" left half (50 days)")
        check_dry(r, 2, cols=range(4), label=" left half (100 days)")
        check_depth(r, 2, H0, TOL, cols=range(5, 10), label=" right half (100 days)")
        check_mass(r, "wall: ", WALL, verror=not fv1)

# ---------------------------------------------------------------- Alzette
if glob.glob(os.path.join(OUT, "alz_*")):
    print("\n== Alzette 10 m, 6 h")
    for s in ("fv1", "acceleration"):
        # output grids; the .mass flux columns vary in their last digit
        # between any two GPU runs (8.3 alike), so the log is not compared
        ok, msg = same_files(f"alz_off_{s}_84", f"alz_off_{s}_83", "*-*")
        report(ok, f"alzette {s}, no losses: 8.4 vs 8.3, output grids: {msg}")
    ok, msg = same_files("alz_uniform_fv1_84", "alz_grid_fv1_84", "*-*")
    report(ok, f"alzette fv1: uniform 2 mm/h vs infilfile 2 mm/h: {msg}")
    for s in ("fv1", "acceleration"):
        off, inf = mass(f"alz_off_{s}_84"), mass(f"alz_uniform_{s}_84")
        if off and inf:
            lost = -inf[-1][11]
            report(inf[-1][5] < off[-1][5] and lost > 0,
                   f"alzette {s}: volume after 6 h {off[-1][5]:.4g} m3 without, "
                   f"{inf[-1][5]:.4g} m3 with 2 mm/h; removed {lost:.4g} m3; "
                   f"largest |Verror| {max(abs(r[10]) for r in off):.3g} without, "
                   f"{max(abs(r[10]) for r in inf):.3g} with")
    ev, un = mass("alz_evap_fv1_84"), mass("alz_uniform_fv1_84")
    if ev and un:
        report(-ev[-1][11] > -un[-1][11],
               f"alzette fv1: with evaporation removed {-ev[-1][11]:.4g} m3 "
               f"(infiltration only {-un[-1][11]:.4g} m3)")

print("\nALL PASSED" if failures == 0 else f"\n{failures} FAILED")
sys.exit(1 if failures else 0)
