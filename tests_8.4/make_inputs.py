#!/usr/bin/env python3
"""Write the inputs and .par files of the 8.4 tests (Python 3, no packages).

    make_inputs.py WORK [ALZETTE_PAR]

WORK/t012/      T012 variants (see README_8.4.md)
WORK/alzette/   Alzette 6 h variants, made from ALZETTE_PAR (the .par of the
                paper's 24 h run, with absolute input paths)
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(sys.argv[1])
T012 = os.path.join(HERE, "data", "T012")


def f32(x):
    """x rounded to single precision, written so that it reads back exactly"""
    return "%.9g" % struct.unpack("f", struct.pack("f", x))[0]


def grid(path, rows, header):
    with open(path, "w") as f:
        f.write(header)
        for r in rows:
            f.write(" ".join(r) + "\n")


# ---------------------------------------------------------------- T012
t = os.path.join(WORK, "t012")
os.makedirs(t, exist_ok=True)
header = ("ncols         10\nnrows         5\nxllcorner     0\nyllcorner     0\n"
          "cellsize      270\nNODATA_value  -9999\n")
for name in ("ETInfTest_dtm.ascii", "ETInfTest_start.ascii", "ETInfTest_evap.dat"):
    with open(os.path.join(T012, name)) as src, open(os.path.join(t, name), "w") as dst:
        dst.write(src.read())

# evaporation rising linearly from 0 to 7.2 mm/day over 100 days
with open(os.path.join(t, "evap_ramp.dat"), "w") as f:
    f.write("Evaporation rising linearly from 0 to 7.2 mm/day\n2 days\n0 0\n7.2 100\n")

# a wall (column 5, 25 m) between two flat halves; infiltration 0.3 mm/h on the left
grid(os.path.join(t, "wall_dtm.ascii"),
     [["25" if i == 4 else "20" for i in range(10)] for _ in range(5)], header)
grid(os.path.join(t, "wall_start.ascii"),
     [["0" if i == 4 else "0.4464" for i in range(10)] for _ in range(5)], header)
grid(os.path.join(t, "wall_infil.ascii"),
     [["0.3" if i < 4 else "0" for i in range(10)] for _ in range(5)], header)

# DG2 also reads the slopes of the DEM and of the start depth (<file>1x, 1y):
# zero on these piecewise flat grids
for name in ("ETInfTest_dtm.ascii", "ETInfTest_start.ascii", "wall_dtm.ascii", "wall_start.ascii"):
    for suffix in ("1x", "1y"):
        grid(os.path.join(t, name + suffix), [["0"] * 10 for _ in range(5)], header)

common = ("sim_time 8640000.0\nmassint 86400.0\nsaveint 4320000.0\n"
          "fpfric 0.06\nelevoff\noutput_precision 6\n")
fixed = "initial_tstep 86400.0\nadaptoff\n"   # as the original T012 (CPU ACC)
adaptive = "initial_tstep 100.0\n"           # FV1: time step from the Courant number (~65 s)
solvers = {
    "cpu_acc": "acceleration\n" + fixed,
    "cuda_acc": "acceleration\ncuda\n" + fixed,
    "cuda_fv1": "fv1\ncuda\n" + adaptive,
    "cuda_fv2": "fv2\ncuda\n" + adaptive,
    "cuda_dg2": "dg2\ncuda\n" + adaptive,
}
cases = {
    "base": ("ETInfTest_dtm.ascii", "ETInfTest_start.ascii",
             "evaporation ETInfTest_evap.dat\ninfiltration 0.00000001\n"),
    "off": ("ETInfTest_dtm.ascii", "ETInfTest_start.ascii", ""),
    "ramp": ("ETInfTest_dtm.ascii", "ETInfTest_start.ascii", "evaporation evap_ramp.dat\n"),
    "wall": ("wall_dtm.ascii", "wall_start.ascii", "infilfile wall_infil.ascii\n"),
}
for case, (dem, start, losses) in cases.items():
    for solver, settings in solvers.items():
        if case != "base" and solver == "cpu_acc" and case != "off":
            continue  # the CPU reference is needed for the base case (and off)
        with open(os.path.join(t, f"{case}_{solver}.par"), "w") as f:
            f.write(f"DEMfile {dem}\nstartfile {start}\nresroot r\n"
                    + common + settings + losses)

# ---------------------------------------------------------------- Green-Ampt
# flat closed basin (T012 grid), 0.1 m of water, loam-like soil, 1 day
for name, rows in (("ga_start.ascii", [["0.1"] * 10 for _ in range(5)]),
                   ("ga_wall_start.ascii", [["0" if i == 4 else "0.1" for i in range(10)] for _ in range(5)])):
    grid(os.path.join(t, name), rows, header)
    for suffix in ("1x", "1y"):
        grid(os.path.join(t, name + suffix), [["0"] * 10 for _ in range(5)], header)
for name, value in (("ga_ks.ascii", "10"), ("ga_psi.ascii", "110"),
                    ("ga_por.ascii", "0.43"), ("ga_theta.ascii", "0.2")):
    grid(os.path.join(t, name), [[value] * 10 for _ in range(5)], header)
grid(os.path.join(t, "ga_ks_left.ascii"), [["10" if i < 4 else "0" for i in range(10)] for _ in range(5)], header)

ga_common = ("sim_time 86400.0\nmassint 3600.0\nsaveint 3600.0\nfpfric 0.06\n"
             "elevoff\noutput_precision 6\ninfiltration greenampt\n")
ga_uniform = ("greenampt_ks 10\ngreenampt_psi 110\ngreenampt_porosity 0.43\n"
              "greenampt_theta_init 0.2\n")
ga_solvers = {
    "cuda_fv1": "fv1\ncuda\ninitial_tstep 60.0\n",
    "cuda_fv2": "fv2\ncuda\ninitial_tstep 60.0\n",
    "cuda_dg2": "dg2\ncuda\ninitial_tstep 60.0\n",
    "cuda_acc": "acceleration\ncuda\ninitial_tstep 60.0\nadaptoff\n",
}
ga_cases = {
    "ga_pond": ("ETInfTest_dtm.ascii", "ga_start.ascii", ga_uniform),
    "ga_grid": ("ETInfTest_dtm.ascii", "ga_start.ascii",
                "greenampt_ks ga_ks.ascii\ngreenampt_psi ga_psi.ascii\n"
                "greenampt_porosity ga_por.ascii\ngreenampt_theta_init ga_theta.ascii\n"),
    "ga_cap": ("ETInfTest_dtm.ascii", "ga_start.ascii", ga_uniform + "greenampt_soil_depth 0.2\n"),
    "ga_wall": ("wall_dtm.ascii", "ga_wall_start.ascii",
                ga_uniform.replace("greenampt_ks 10", "greenampt_ks ga_ks_left.ascii")),
}
for case, (dem, start, extra) in ga_cases.items():
    for solver, settings in ga_solvers.items():
        with open(os.path.join(t, f"{case}_{solver}.par"), "w") as f:
            f.write(f"DEMfile {dem}\nstartfile {start}\nresroot r\n" + ga_common + settings + extra)
# both infiltration models in one file: must stop with an error
with open(os.path.join(t, "ga_conflict_cuda_fv1.par"), "w") as f:
    f.write("DEMfile ETInfTest_dtm.ascii\nstartfile ga_start.ascii\nresroot r\n" + ga_common
            + ga_solvers["cuda_fv1"] + ga_uniform + "infiltration 0.000001\n")

# ---------------------------------------------------------------- Alzette, 6 h
if len(sys.argv) > 2:
    a = os.path.join(WORK, "alzette")
    os.makedirs(a, exist_ok=True)
    lines = [l for l in open(sys.argv[2]) if l.strip() and not l.startswith("#")]
    keep = [l for l in lines if l.split()[0] not in ("sim_time", "fv1", "acceleration", "cuda")]
    base = "".join(keep) + "sim_time 21600\ncuda\n"
    dem = [l.split()[1] for l in lines if l.split()[0] == "DEMfile"][0]

    # infiltration 2 mm/h in every basin cell, NoData outside
    with open(dem) as src, open(os.path.join(a, "infil_2mmh.ascii"), "w") as dst:
        head = [src.readline() for _ in range(6)]
        dst.write("".join(head))
        nodata = head[5].split()[1]
        for row in src:
            dst.write(" ".join(nodata if float(v) == float(nodata) else "2"
                               for v in row.split()) + "\n")
    with open(os.path.join(a, "evap_3mmday.dat"), "w") as f:
        f.write("Evaporation 3 mm/day\n2 days\n3 0\n3 10\n")

    uniform = f32(2.0 / 3600000.0)   # the same single-precision rate as 2 mm/h
    variants = {
        "off": "",
        "uniform": f"infiltration {uniform}\n",
        "grid": "infilfile infil_2mmh.ascii\n",
        "evap": f"infiltration {uniform}\nevaporation evap_3mmday.dat\n",
        # loam-like soil, already wet at the start (theta_init 0.30 of 0.43)
        "greenampt": "infiltration greenampt\ngreenampt_ks 10\ngreenampt_psi 110\n"
                     "greenampt_porosity 0.43\ngreenampt_theta_init 0.30\n",
    }
    for solver in ("fv1", "acceleration"):
        for name, losses in variants.items():
            if solver == "acceleration" and name in ("grid", "evap", "greenampt"):
                continue
            with open(os.path.join(a, f"{name}_{solver}.par"), "w") as f:
                f.write(base + solver + "\n" + losses)
print("inputs written to", WORK)
