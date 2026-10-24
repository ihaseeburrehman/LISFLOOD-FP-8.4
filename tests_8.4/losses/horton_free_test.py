#!/usr/bin/env python3
"""Quick checks of Horton infiltration and the outflow-only FREE boundary.

    horton_free_test.py BUILD_DIR WORK [SOLVER ...]

BUILD_DIR holds lisflood and generateDG2DEM (the .par files say `cuda`, which
runs the Metal solvers on a Mac). Each run takes seconds. Two cases, 32 x 32
cells of 10 m, on every GPU solver (default: all seven):

horton  closed flat box, 0.2 m of water at rest, no rain, 1 h, Horton with
        f0 = 60 mm/h, fc = 10 mm/h, k = 2 /h. The water never runs out, so the
        depth infiltrated is exact: F(t) = fc t + (f0 - fc)/k (1 - exp(-k t)).
        Pass: the stored volume matches within 0.2 % of F.

The volume is summed from the last depth output (.wd, or .vtk for CUDA
ACC_NUGRID), interior cells only: the CUDA FV1 .mass file also counts its
ghost cells, which hold water next to a wet closed edge. Where a solver saved
no depth after the start (Metal FV2/DG2 can skip it), the .mass row is used.
free    plane falling 2 % from a FREE west edge to a closed east edge, 5 cm of
        water at the start, 30 min. All flow runs away from the FREE edge, so
        none may enter: the stored volume must equal the start volume. Pass:
        within 0.01 % (before the fix, FV1 and FV2 let water in here). DG2, ACC, ACC_NUGRID, HWFV1 and MWDG2 have FREE
        rules of their own (DG2 empties FREE-edge cells, ACC uses a slope
        outflow, HWFV1/MWDG2 a critical-depth ghost) that the fix does not
        change; their result is printed for information only.
"""
import math
import pathlib
import subprocess
import sys

N, DX = 32, 10.0
AREA = (N * DX) ** 2
F0, FC, K = 60.0, 10.0, 2.0          # mm/h, mm/h, 1/h
H0, T_H = 0.2, 3600.0                 # m, s
T_F, H_F = 1800.0, 0.05               # s, m
FREE_FIXED = ("fv1", "fv2")
SOLVERS = {
    "fv1": "fv1\n", "fv2": "fv2\n", "dg2": "dg2\n", "acc": "acceleration\n",
    "acc_nugrid": "acc_nugrid\nL 5\nepsilon 0\n",
    "hwfv1": "hwfv1\nmax_ref_lvl 5\nepsilon 0\nwall_height 10000\n",
    "mwdg2": "mwdg2\nmax_ref_lvl 5\nepsilon 0\nwall_height 10000\n",
}


def grid(path, value):
    rows = "\n".join(" ".join("%.6f" % value(i, j) for i in range(N)) for j in range(N))
    path.write_text(f"ncols {N}\nnrows {N}\nxllcorner 0\nyllcorner 0\n"
                    f"cellsize {DX:g}\nNODATA_value -9999\n{rows}\n")


def inputs(work, build, case):
    d = work / case
    d.mkdir(parents=True, exist_ok=True)
    (d / "stage.txt").write_text("0\n")
    if case == "horton":
        grid(d / "dem.asc", lambda i, j: 10.0)
        grid(d / "start.wd", lambda i, j: H0)
        for s in ("1x", "1y"):
            grid(d / f"start.wd{s}", lambda i, j: 0.0)
    else:
        grid(d / "dem.asc", lambda i, j: 0.02 * DX * (N - i))
        grid(d / "start.wd", lambda i, j: H_F)
        for s in ("1x", "1y"):
            grid(d / f"start.wd{s}", lambda i, j: 0.0)
        (d / "free.bci").write_text(f"W 0 {N * DX:g} FREE\n")
    # DG2 slope files; generateDG2DEM rewrites the DEM it is given (a copy here)
    (d / "gen.par").write_text("DEMfile dem.asc\nnodata_elevation -9999\n")
    subprocess.run([str(build / "generateDG2DEM"), "gen.par"], cwd=d,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
    return d


def depth_values(path):
    if path.suffix == ".vtk":       # CUDA ACC_NUGRID: cell data "h"
        values, n, on = [], 0, False
        for line in path.read_text().splitlines():
            if line.startswith("CELL_DATA "):
                n = int(line.split()[1])
            elif line.startswith("SCALARS h "):
                on = True
            elif on and not line.startswith("LOOKUP_TABLE"):
                values += [float(w) for w in line.split()]
                if len(values) >= n:
                    break
        return values[:n]
    return [float(w) for w in path.read_text().split()[12:]]


def run(build, d, case, solver):
    out = d / solver
    t_end = T_H if case == "horton" else T_F
    common = (f"DEMfile dem.asc\nresroot r\ndirroot {solver}\ncuda\n{SOLVERS[solver]}"
              # the save falls half a second before t_end, never on a step time
              f"fpfric 0.03\nmassint 300\nsaveint {t_end - 0.5:g}\ninitial_tstep 1\n"
              "output_precision 9\n"
              # end a minute past the save, so every solver writes it
              f"stagefile stage.txt\nstartfile start.wd\nsim_time {t_end + 60:g}\n")
    if case == "horton":
        par = common + f"infiltration horton\nhorton_f0 {F0}\nhorton_fc {FC}\nhorton_k {K}\n"
    else:
        par = common + "bcifile free.bci\n"
    if solver in ("hwfv1", "mwdg2"):
        par += "raster_out\n"
    (d / f"{solver}.par").write_text(par)
    out.mkdir(exist_ok=True)
    with open(out / "run.log", "w") as log:
        rc = subprocess.run([str(build / "lisflood"), f"{solver}.par"], cwd=d,
                            stdout=log, stderr=subprocess.STDOUT, timeout=600).returncode
    def index(p):
        digits = "".join(c for c in p.stem.split("-")[-1] if c.isdigit())
        return int(digits or -1)
    grids = sorted(list(out.glob("r-*.wd")) + list(out.glob("r-*.vtk")), key=index)
    if rc != 0:
        return None, rc, (out / "run.log").read_text()[-300:]
    if len(grids) < 2 or index(grids[-1]) < 1:
        # no saved depth after the start (Metal FV2/DG2 can skip it): use the
        # .mass row at t_end
        mass = out / "r.mass"
        if not mass.exists():
            return None, rc, "no depth output after the start and no .mass"
        rows = [r.split() for r in mass.read_text().splitlines()[1:] if r.strip()]
        row = min(rows, key=lambda r: abs(float(r[0]) - t_end))
        return (float(row[0]), float(row[5])), rc, ""
    h = depth_values(grids[-1])
    if len(h) != N * N:
        return None, rc, f"{grids[-1].name}: {len(h)} cells, expected {N * N}"
    return (t_end, sum(max(x, 0.0) for x in h) * DX * DX), rc, ""


def main():
    build = pathlib.Path(sys.argv[1]).resolve()
    work = pathlib.Path(sys.argv[2]).resolve()
    solvers = sys.argv[3:] or list(SOLVERS)
    f_exact = (FC * T_H / 3600 + (F0 - FC) / K * (1 - math.exp(-K * T_H / 3600))) / 1000
    v_horton = (H0 - f_exact) * AREA
    v_free = H_F * AREA
    print(f"horton: exact F = {f_exact*1000:.4f} mm, volume {v_horton:.3f} m3")
    print(f"free:   start volume {v_free:.3f} m3")
    ok = True
    for case in ("horton", "free"):
        d = inputs(work, build, case)
        for s in solvers:
            res, rc, tail = run(build, d, case, s)
            if res is None:
                print(f"{case:6s} {s:10s} FAIL rc={rc}  {tail.strip().splitlines()[-1] if tail.strip() else ''}")
                ok = False
                continue
            t, vol = res
            if case == "horton":
                f_model = H0 - vol / AREA
                err = (f_model - f_exact) / f_exact
                good = abs(err) < 2e-3 and t >= T_H - 1
                print(f"horton {s:10s} {'PASS' if good else 'FAIL'}  t={t:.0f} s  "
                      f"F={f_model*1000:.4f} mm  error {err*100:+.3f} %")
            else:
                err = (vol - v_free) / v_free
                good = abs(err) < 1e-4 and t >= T_F - 1
                label = "PASS" if good else "FAIL"
                if s not in FREE_FIXED:
                    label, good = "info", True
                print(f"free   {s:10s} {label}  t={t:.0f} s  "
                      f"volume={vol:.3f} m3  vs start {err*100:+.4f} %")
            ok &= good
    print("ALL PASSED" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
