#!/usr/bin/env python3
"""Metal against CUDA for the adaptive GPU solvers (MWDG2, HWFV1, ACC_NUGRID).

    adaptive_parity.py make WORK              write every case and run_cuda.sh
    adaptive_parity.py metal WORK BIN [CASE]  run the Metal binary on the cases
    adaptive_parity.py compare WORK [CASE]    compare Metal with the CUDA outputs
    adaptive_parity.py check WORK BIN         Metal against the saved CUDA outputs

Each case folder holds the inputs, `case.par` (run in the folder by both back
ends) and, after the runs, CUDA's `res/<case>-<k>.*` and Metal's
`metal-<kkkk>.*` rasters. run_cuda.sh runs a CUDA build on every case (copy
WORK to the HPC, run there, copy the `res` folders back).

Compared at every saved output k >= 1 present on both sides (CUDA saves at the
first step at or after k*saveint, Metal exactly at it; cases use steps that
divide saveint). Printed per output: depth and flow-vector normalized L2
difference (%), largest depth difference (m), and the water volume of each.
"""
import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DAM = os.path.join(HERE, "fixtures", "mwdg2_cuda_z1_l5_e001")


def grid(path, values, cellsize=1.0, nodata=-9999):
    rows, cols = values.shape
    with open(path, "w") as f:
        f.write(f"ncols {cols}\nnrows {rows}\nxllcorner 0\nyllcorner 0\n"
                f"cellsize {cellsize:g}\nNODATA_value {nodata}\n")
        for row in values:
            f.write(" ".join(f"{v:.6g}" for v in row) + "\n")


def read(path):
    return np.loadtxt(path, skiprows=6)


def dam_inputs(folder, bed=None):
    dem = read(os.path.join(DAM, "dem.asc")) if bed is None else bed
    grid(os.path.join(folder, "dem.asc"), dem)
    grid(os.path.join(folder, "start.asc"), read(os.path.join(DAM, "start.asc")))


def hill_inputs(folder):
    # sloping bed with a bump and a pond: wet/dry fronts and bed slopes
    y, x = np.mgrid[0:32, 0:32]
    dem = 0.05 * x + 2.0 * np.exp(-((x - 20) ** 2 + (y - 12) ** 2) / 20.0)
    start = np.where((x < 10) & (y > 8) & (y < 24), 3.0 - dem, 0.0).clip(min=0)
    grid(os.path.join(folder, "dem.asc"), dem)
    grid(os.path.join(folder, "start.asc"), start)


def pond_inputs(folder):
    # a bowl with an off-centre pond; the water never reaches the domain edge
    y, x = np.mgrid[0:32, 0:32]
    dem = 0.01 * ((x - 16.0) ** 2 + (y - 16.0) ** 2)
    start = np.where((x > 8) & (x < 15) & (y > 12) & (y < 20), 1.2 - dem, 0.0).clip(min=0)
    grid(os.path.join(folder, "dem.asc"), dem)
    grid(os.path.join(folder, "start.asc"), start)


def lake_hill_inputs(folder):
    # tests_8.4/fixtures/mwdg2_cuda_lake32_l5: a hill in a lake at rest
    fix = os.path.join(HERE, "fixtures", "mwdg2_cuda_lake32_l5")
    grid(os.path.join(folder, "dem.asc"), read(os.path.join(fix, "dem.asc")))
    grid(os.path.join(folder, "start.asc"), read(os.path.join(fix, "start.asc")))


COMMON = ["DEMfile dem.asc", "startfile start.asc", "max_ref_lvl 5",
          "cuda", "qoutput", "raster_out", "output_precision 12"]

# name: (inputs, solver lines, sim_time, saveint, initial_tstep, extra lines)
CASES = {
    "dam_fric_mwdg2": (dam_inputs, ["mwdg2", "epsilon 0.001", "limitslopes", "tol_Krivo 1"],
                       0.1, 0.01, 0.001, ["fpfric 0.03"]),
    "dam_fric_mwdg2_long": (dam_inputs, ["mwdg2", "epsilon 0.001", "limitslopes", "tol_Krivo 1"],
                            20, 2, 0.03, ["fpfric 0.03"]),
    "dam_fric_mwdg2_diag": (dam_inputs, ["mwdg2", "epsilon 0.001", "limitslopes", "tol_Krivo 1"],
                            0.5, 0.001, 0.03, ["fpfric 0.03"]),
    "dam_diag_nolim": (dam_inputs, ["mwdg2", "epsilon 0.001"], 0.5, 0.001, 0.03, ["fpfric 0.03"]),
    "dam_diag_nofric": (dam_inputs, ["mwdg2", "epsilon 0.001", "limitslopes", "tol_Krivo 1"],
                        0.5, 0.001, 0.03, ["fpfric 0"]),
    "dam_diag_eps0": (dam_inputs, ["mwdg2", "epsilon 0", "limitslopes", "tol_Krivo 1"],
                      0.5, 0.001, 0.03, ["fpfric 0.03"]),
    "lake_hill_mwdg2": (lake_hill_inputs, ["mwdg2", "epsilon 0.001", "limitslopes", "tol_Krivo 1"],
                        0.2, 0.1, 0.05, ["fpfric 0.03"]),
    "dam_hwfv1": (dam_inputs, ["hwfv1", "epsilon 0.001"], 0.1, 0.01, 0.001, ["fpfric 0.03"]),
    "dam_hwfv1_long": (dam_inputs, ["hwfv1", "epsilon 0.001"], 20, 2, 0.03, ["fpfric 0.03"]),
    "hill_hwfv1": (hill_inputs, ["hwfv1", "epsilon 0.01"], 30, 3, 0.05, ["fpfric 0.03"]),
    "hill_mwdg2": (hill_inputs, ["mwdg2", "epsilon 0.01"], 30, 3, 0.05, ["fpfric 0.03"]),
    "pond_mwdg2": (pond_inputs, ["mwdg2", "epsilon 0.01"], 30, 3, 0.05, ["fpfric 0.03"]),
    "pond_mwdg2_nofric": (pond_inputs, ["mwdg2", "epsilon 0.01"], 30, 3, 0.05, ["fpfric 0"]),
    "pond_hwfv1": (pond_inputs, ["hwfv1", "epsilon 0.01"], 30, 3, 0.05, ["fpfric 0.03"]),
    "pond_hwfv1_short": (pond_inputs, ["hwfv1", "epsilon 0.01"], 0.1, 0.005, 0.001, ["fpfric 0.03"]),
    "pond_mwdg2_short": (pond_inputs, ["mwdg2", "epsilon 0.01"], 0.1, 0.005, 0.001, ["fpfric 0"]),
}


BDY = "q1\n3 seconds\n0 0\n0.5 10\n0.2 40\nh1\n2 seconds\n1.0 0\n2.5 30\n"
FILES = {  # extra files per case group (boundary conditions, point sources)
    "bcfree": {"bc.bci": "W 0 32 HFIX 3.0\nE 0 32 FREE\n"},
    "bcqvar": {"bc.bci": "S 8 24 QVAR q1\nN 0 32 FREE\n", "bc.bdy": BDY},
    "ps": {"bc.bci": "P 10.5 16.5 QFIX 0.5\nP 20.5 20.5 HVAR h1\n", "bc.bdy": BDY},
    "bcinf": {"bc.bci": "W 4 28 QFIX 0.3\nE 0 32 FREE\n"},
}
BC_LINES = {"bcfree": ["bcifile bc.bci"], "bcqvar": ["bcifile bc.bci", "bdyfile bc.bdy"],
            "ps": ["bcifile bc.bci", "bdyfile bc.bdy"],
            "bcinf": ["bcifile bc.bci", "infiltration 0.00001"]}
for solver in ("mwdg2", "hwfv1"):
    for group in FILES:
        CASES[f"{group}_{solver}"] = (hill_inputs, [solver, "epsilon 0.01"], 30, 3, 0.05,
                                      ["fpfric 0.03"] + BC_LINES[group])


def rect_lake_inputs(folder):
    # 20 x 12 raster (padded to 32 x 32): hill in a lake at rest
    y, x = np.mgrid[0:12, 0:20]
    dem = 0.1 * x + 1.5 * np.exp(-((x - 8) ** 2 + (y - 5) ** 2) / 6.0)
    grid(os.path.join(folder, "dem.asc"), dem)
    grid(os.path.join(folder, "start.asc"), np.clip(3.0 - dem, 0, None))


def rect_dam_inputs(folder):
    # 20 x 12 raster: a dam break that reaches the raster's north and east edges
    y, x = np.mgrid[0:12, 0:20]
    dem = 0.02 * x
    grid(os.path.join(folder, "dem.asc"), dem)
    grid(os.path.join(folder, "start.asc"), np.where(x < 8, 2.0, 0.2))


def with_slopes(inputs):
    # acc_nugrid also reads <DEM>1x and <DEM>1y (DG2 bed slopes): zero here
    def make_inputs(folder):
        inputs(folder)
        flat = np.zeros_like(read(os.path.join(folder, "dem.asc")))
        grid(os.path.join(folder, "dem.asc1x"), flat)
        grid(os.path.join(folder, "dem.asc1y"), flat)
    return make_inputs


RAIN = "rain\n3 seconds\n0 0\n60 15\n0 30\n"
NUG = ["acc_nugrid", "L 5"]
CASES.update({
    "nug_hill": (with_slopes(hill_inputs), NUG + ["epsilon 0.01"], 30, 3, 1.0, ["fpfric 0.03"]),
    "nug_dam": (with_slopes(dam_inputs), NUG + ["epsilon 0.01"], 20, 2, 1.0, ["fpfric 0.03"]),
    "nug_bcfree": (with_slopes(hill_inputs), NUG + ["epsilon 0.01"], 30, 3, 1.0,
                   ["fpfric 0.03", "bcifile bc.bci"]),
    "nug_ps": (with_slopes(hill_inputs), NUG + ["epsilon 0.01"], 30, 3, 1.0,
               ["fpfric 0.03", "bcifile bc.bci", "bdyfile bc.bdy"]),
    "nug_raininf": (with_slopes(hill_inputs), NUG + ["epsilon 0.01"], 30, 3, 1.0,
                    ["fpfric 0.03", "rainfall rain.txt", "infiltration 0.00001"]),
})
STAGE = "2\n8.5 16.5\n24.5 8.5\n"   # cuda acc_nugrid needs a stage file
FILES.update({"nug_hill": {"stage.txt": STAGE}, "nug_dam": {"stage.txt": STAGE},
              "nug_bcfree": {"bc.bci": "W 0 32 HFIX 3.0\nE 0 32 FREE 0.01\n", "stage.txt": STAGE},
              "nug_ps": {"bc.bci": "P 10.5 16.5 QFIX 0.5\nP 20.5 20.5 HVAR h1\n", "bc.bdy": BDY,
                         "stage.txt": STAGE},
              "nug_raininf": {"rain.txt": RAIN, "stage.txt": STAGE}})
for solver in ("mwdg2", "hwfv1"):
    CASES[f"rectlake_{solver}"] = (rect_lake_inputs, [solver, "epsilon 0.01"], 1, 0.1, 0.05, ["fpfric 0.03"])
    CASES[f"rectdam_{solver}"] = (rect_dam_inputs, [solver, "epsilon 0.01"], 10, 1, 0.05, ["fpfric 0.03"])
for name in ("nug_hill", "nug_dam", "nug_bcfree", "nug_ps", "nug_raininf"):
    CASES[name][5].append("stagefile stage.txt")


def read_vtk(path):
    """leaf count and the z0 and h cell values of a write_soln_vtk file"""
    lines = open(path).read().split("\n")
    def block(name):
        i = lines.index(f"SCALARS {name} float 1") + 2
        n = int(next(l for l in lines if l.startswith("CELL_DATA")).split()[1])
        return np.array([float(v) for v in lines[i:i + n]])
    return block("z0"), block("h")


def make(work):
    os.makedirs(work, exist_ok=True)
    for name, (inputs, solver, sim, save, dt0, extra) in CASES.items():
        folder = os.path.join(work, name)
        os.makedirs(folder, exist_ok=True)
        inputs(folder)
        for fname, text in FILES.get(name if name in FILES else name.rsplit("_", 1)[0], {}).items():
            with open(os.path.join(folder, fname), "w") as f:
                f.write(text)
        lines = COMMON + solver + extra + [
            f"sim_time {sim:g}", f"saveint {save:g}", f"massint {save:g}",
            f"initial_tstep {dt0:g}"]
        with open(os.path.join(folder, "case.par"), "w") as f:
            f.write("\n".join(lines + [f"resroot {name}", ""]))
        with open(os.path.join(folder, "metal.par"), "w") as f:
            f.write("\n".join(lines + ["resroot metal", ""]))
    with open(os.path.join(work, "run_cuda.sh"), "w") as f:
        f.write("#!/bin/sh\n# run_cuda.sh LISFLOOD_BINARY: CUDA run of every case\n"
                "BIN=$1\ncd \"$(dirname \"$0\")\"\n"
                "for c in */; do c=${c%/}; [ -f \"$c/case.par\" ] || continue\n"
                "  (cd \"$c\" && mkdir -p res && \"$BIN\" case.par > cuda.log 2>&1; "
                "echo \"$c rc=$?\")\ndone\n")
    print("cases written to", work)


def metal(work, binary, only=None):
    for name in CASES:
        if only and name != only:
            continue
        folder = os.path.join(work, name)
        with open(os.path.join(folder, "metal.log"), "w") as log:
            rc = subprocess.call([binary, "-v", "metal.par"], cwd=folder,
                                 stdout=log, stderr=subprocess.STDOUT)
        print(f"{name} metal rc={rc}")


def compare(work, only=None):
    worst = {}
    for name in CASES:
        if only and name != only:
            continue
        folder = os.path.join(work, name)
        if name.startswith("nug_"):
            print(f"== {name} (VTK leaves at each save: count, depth L2 %, max |dh| m, volume ratio)")
            for k in range(1, 100):
                c, m = os.path.join(folder, f"{name}-{k:04d}.vtk"), os.path.join(folder, f"metal-{k:04d}.vtk")
                if not (os.path.exists(c) and os.path.exists(m)):
                    break
                zc, hc = read_vtk(c)
                zm, hm = read_vtk(m)
                if len(hc) != len(hm):
                    print(f"{k:4d} leaf counts differ: CUDA {len(hc)} Metal {len(hm)}")
                    worst[name] = 100.0
                    continue
                dl2 = 100 * np.linalg.norm(hm - hc) / max(np.linalg.norm(hc), 1e-30)
                print(f"{k:4d} leaves {len(hc)} bed equal {np.array_equal(zc, zm)} depth L2 {dl2:.4f}"
                      f" max|dh| {abs(hm - hc).max():.3g} sum {hc.sum():.4f} / {hm.sum():.4f}")
                worst[name] = max(worst.get(name, 0), dl2)
            continue
        print(f"== {name}")
        print("   k   depth L2%   flow L2%   max|dh| m   volume CUDA / Metal")
        cuda = {int(p.rsplit("-", 1)[1][:-3]): p[:-3]
                for p in glob.glob(os.path.join(folder, "res", f"{name}-*.wd"))}
        for k in sorted(cuda):
            m = os.path.join(folder, f"metal-{k:04d}")
            if k < 1 or not os.path.exists(m + ".wd"):
                continue
            hc, hm = read(cuda[k] + ".wd"), read(m + ".wd")
            qc = np.stack([read(cuda[k] + ".qx"), read(cuda[k] + ".qy")])
            qm = np.stack([read(m + ".Qx"), read(m + ".Qy")])
            dl2 = 100 * np.linalg.norm(hm - hc) / max(np.linalg.norm(hc), 1e-30)
            ql2 = 100 * np.linalg.norm(qm - qc) / max(np.linalg.norm(qc), 1e-30)
            print(f"{k:4d} {dl2:10.4f} {ql2:10.4f} {abs(hm - hc).max():11.3g}"
                  f"   {hc.sum():.4f} / {hm.sum():.4f}")
            worst[name] = max(worst.get(name, 0), dl2)
    return worst


# check: saved single-precision CUDA outputs (tests_8.4/fixtures/adaptive_parity,
# the last output of each case, from the A100 build of 8.4) against Metal.
# Limits: normalized L2 depth difference (%) at that output.
CHECKS = {
    "dam_fric_mwdg2_diag": ("dam_fric_mwdg2_diag-28", 0.01),  # MWDG2, friction, limiter, 29 steps
    "dam_hwfv1_long": ("dam_hwfv1_long-10", 0.1),            # HWFV1, 20 s
    "hill_hwfv1": ("hill_hwfv1-10", 0.2),                    # HWFV1, wet/dry hill, 30 s
    "bcfree_hwfv1": ("bcfree_hwfv1-10", 0.5),                # HFIX inflow, FREE outflow
    "bcqvar_hwfv1": ("bcqvar_hwfv1-10", 0.5),                # QVAR inflow, FREE outflow
    "ps_hwfv1": ("ps_hwfv1-10", 0.5),                        # QFIX and HVAR point sources
    "bcinf_hwfv1": ("bcinf_hwfv1-10", 0.5),                  # QFIX inflow with infiltration
    "nug_dam": ("nug_dam-0010", 0.5),                        # ACC_NUGRID, compressed mesh
}


def check(work, binary):
    fix = os.path.join(HERE, "fixtures", "adaptive_parity")
    make(work)
    failures = 0
    for name, (ref, limit) in CHECKS.items():
        folder = os.path.join(work, name)
        metal(work, binary, name)
        if name.startswith("nug_"):
            k = int(ref.rsplit("-", 1)[1])
            zc, hc = read_vtk(os.path.join(fix, name, ref + ".vtk"))
            zm, hm = read_vtk(os.path.join(folder, f"metal-{k:04d}.vtk"))
            ok = len(hc) == len(hm)
            dl2 = 100 * np.linalg.norm(hm - hc) / np.linalg.norm(hc) if ok else float("inf")
            text = f"{len(hm)} leaves (CUDA {len(hc)}), depth L2 {dl2:.4f}%"
        else:
            k = int(ref.rsplit("-", 1)[1])
            hc = read(os.path.join(fix, name, ref + ".wd"))
            hm = read(os.path.join(folder, f"metal-{k:04d}.wd"))
            dl2 = 100 * np.linalg.norm(hm - hc) / np.linalg.norm(hc)
            ok = True
            text = f"depth L2 {dl2:.4f}%, max |dh| {abs(hm - hc).max():.3g} m"
        ok = ok and dl2 <= limit
        failures += not ok
        print(("PASS " if ok else "FAIL ") + f"{name}: output {k} against CUDA: {text} (limit {limit}%)")
    print("ALL PASSED" if not failures else f"{failures} FAILED")
    return failures


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "make":
        make(sys.argv[2])
    elif cmd == "metal":
        metal(sys.argv[2], os.path.abspath(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == "check":
        sys.exit(1 if check(sys.argv[2], os.path.abspath(sys.argv[3])) else 0)
    elif cmd == "compare":
        compare(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        print(__doc__)
