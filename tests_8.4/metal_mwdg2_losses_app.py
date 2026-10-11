#!/usr/bin/env python3
"""Real LISFLOOD app regressions for adaptive MWDG2 raster losses."""
from pathlib import Path
import math
import re
import subprocess
import sys

binary, work = Path(sys.argv[1]), Path(sys.argv[2])
root = work / "losses"
root.mkdir(parents=True, exist_ok=True)

def write_asc(path, rows, cellsize=10):
    h, w = len(rows), len(rows[0])
    path.write_text(
        f"ncols {w}\nnrows {h}\nxllcorner 0\nyllcorner 0\n"
        f"cellsize {cellsize}\nNODATA_value -9999\n"
        + "\n".join(" ".join(map(str, row)) for row in rows) + "\n"
    )

write_asc(root / "dem.asc", [[100.0] * 5 for _ in range(3)])
write_asc(root / "start.asc", [[0.1] * 5 for _ in range(3)])
write_asc(root / "rates.asc", [[360000 if x < 2 else 0 for x in range(5)] for _ in range(3)])
(root / "evap.txt").write_text(
    "synthetic evaporation\n2 seconds\n8640000 0\n8640000 1\n"
)
(root / "rain.txt").write_text(
    "synthetic interpolated rainfall\n2 seconds\n3600 0\n7200 0.02\n"
)
base = [
    f"DEMfile {root}/dem.asc", f"startfile {root}/start.asc",
    "sim_time 0.02", "saveint 0.01", "massint 0.01", "initial_tstep 0.01",
    "fpfric 0", "max_ref_lvl 3", "epsilon 0.1", "mwdg2", "cuda",
]
variants = {
    "uniform": ["infiltration 0.1"],
    "grid": [f"infilfile {root}/rates.asc"],
    "greenampt": ["infiltration greenampt", "greenampt_ks 10",
                  "greenampt_psi 100", "greenampt_porosity 0.45",
                  "greenampt_theta_init 0.2"],
    "greenampt_cap": ["infiltration greenampt", "greenampt_ks 10",
                      "greenampt_psi 100", "greenampt_porosity 0.45",
                      "greenampt_theta_init 0.2", "greenampt_soil_depth 0.0000001"],
    "evap": [f"evaporation {root}/evap.txt"],
    "rain": [f"rainfall {root}/rain.txt"],
    "combined": ["infiltration 0.1", f"rainfall {root}/rain.txt"],
}
logs = {}
try:
    for name, settings in variants.items():
        case = root / f"{name}.par"
        case.write_text("\n".join(base[:2] + [f"resroot {root}/{name}"] +
                                  base[2:-2] + settings + base[-2:] + [""]))
        result = subprocess.run([str(binary), "-v", str(case)], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        logs[name] = result.stdout
        (root / f"{name}.log").write_text(result.stdout)
        if result.returncode:
            raise AssertionError(f"{name}: app exit {result.returncode}")
        if "Metal device:" not in result.stdout or "enabled" not in result.stdout:
            raise AssertionError(f"{name}: native adaptive Metal path was not reported")

    def last_mass(name):
        rows = (root / f"{name}.mass").read_text().strip().splitlines()[1:]
        # CUDA adaptive MWDG2 starts with a 0.001 s step: one record after it,
        # then one per massint.
        if len(rows) != 3:
            raise AssertionError(f"{name}: expected three mass records, got {len(rows)}")
        values = rows[-1].split()
        return float(values[5]), float(values[11])

    uniform, grid = last_mass("uniform"), last_mass("grid")
    green, evap = last_mass("greenampt"), last_mass("evap")
    cap, combined = last_mass("greenampt_cap"), last_mass("combined")
    rain = last_mass("rain")
    if not math.isclose(uniform[0], 147.0, abs_tol=0.05) or not math.isclose(uniform[1], -3.0, abs_tol=0.01):
        raise AssertionError(f"uniform infiltration/mass accounting failed: {uniform}")
    if not math.isclose(grid[0], 148.8, abs_tol=0.05) or not math.isclose(grid[1], -1.2, abs_tol=0.01):
        raise AssertionError(f"distributed infiltration/mass accounting failed: {grid}")
    if not (149.5 < green[0] < 150.0 and green[1] < 0):
        raise AssertionError(f"Green-Ampt was not applied: {green}")
    if not math.isclose(cap[1], -3.75e-5, abs_tol=1e-5):
        raise AssertionError(f"Green-Ampt soil-depth cap failed: {cap}")
    if not math.isclose(evap[0], 147.0, abs_tol=0.05) or not math.isclose(evap[1], -3.0, abs_tol=0.01):
        raise AssertionError(f"evaporation/mass accounting failed: {evap}")
    # LoadRain converts mm/hour to m/s. As in the fixed-grid Metal DG2 path,
    # each step adds rain over the previous step actually taken: steps are
    # 0.001 s (CUDA's first step), 0.01 s and 0.009 s (no step is shortened
    # for outputs), so rain covers the first two at the rates of t = 0.001 and
    # 0.011 s: (0.00105*0.001 + 0.00155*0.01) m/s*s over 1500 m2 = 0.024825 m3.
    if not math.isclose(rain[0], 150.0248, abs_tol=0.01) or not math.isclose(rain[1], 0.024825, abs_tol=0.0005):
        raise AssertionError(f"uniform rainfall/interpolation/mass accounting failed: {rain}")
    if not math.isclose(combined[1], 0.024825 - 3.0, abs_tol=0.01):
        raise AssertionError(f"combined rainfall/loss mass accounting failed: {combined}")
    loss_rows = (root / "combined.loss").read_text().strip().splitlines()
    if loss_rows[0] != "time,integrated_time,loss_volume,water_volume" or len(loss_rows) != 4:
        raise AssertionError(f"CUDA-compatible .loss diagnostic schema/records failed: {loss_rows}")
    if not math.isclose(float(loss_rows[-1].split(",")[2]), 3.0, abs_tol=0.01):
        raise AssertionError(f"combined run .loss total is incorrect: {loss_rows[-1]}")
    lines = (root / "grid-0002.wd").read_text().splitlines()[6:]
    depth = [[float(x) for x in row.split()] for row in lines]
    if any(abs(depth[y][x] - 0.098) > 1e-5 for y in range(3) for x in range(2)):
        raise AssertionError("per-cell infiltration grid rates were not applied to wet cells")
    if any(abs(depth[y][x] - 0.1) > 1e-5 for y in range(3) for x in range(2, 5)):
        raise AssertionError("zero-rate cells in infiltration grid lost water")
except Exception:
    for name, log in logs.items():
        print(f"\n--- {name} app log ---\n{log}", file=sys.stderr)
    raise

print("PASS: adaptive Metal uniform/grid/Green-Ampt infiltration, evaporation, interpolated uniform rain, and mass accounting")

# Build a small three-slice NetCDF forcing file. Two intensity changes occur
# between 0.1 s outputs. A NoData fringe overlaps one third of the forcing
# tile, verifying the rainfall kernel skips NoData cells in its mass total.
dynamic = root / "dynamic"
dynamic.mkdir(exist_ok=True)
write_asc(dynamic / "dem.asc", [[100.0] * 9 for _ in range(9)], cellsize=1)
dem_rows = (dynamic / "dem.asc").read_text().splitlines()[6:]
dem = [[float(x) for x in row.split()] for row in dem_rows]
# The eastern fringe is NoData, leaving two columns of the 3x3 rain tile wet.
for row in dem:
    for col in range(5, 9):
        row[col] = -9999.0
write_asc(dynamic / "dem.asc", dem, cellsize=1)
start = [[0.1 if value != -9999 else -9999 for value in row] for row in dem]
write_asc(dynamic / "start.asc", start, cellsize=1)
cdl = dynamic / "rain.cdl"
cdl.write_text('''netcdf mwdg2_rain {
dimensions: time = 3 ; x = 2 ; y = 4 ;
variables:
 float time(time) ; time:units = "hours" ; time:axis = "T" ;
 double x(x) ; x:units = "m" ; x:axis = "X" ;
 double y(y) ; y:units = "m" ; y:axis = "Y" ;
 float rainfall_depth(time, y, x) ; rainfall_depth:units = "mm" ;
data:
 time = 0, 0.000025, 0.00005 ;
 x = 1.5, 4.5 ;
 y = 10.5, 7.5, 4.5, 1.5 ;
 rainfall_depth =
  0,0, 0,0, 0,0, 0,0,
  0,0, 0,0, 0.01,0.01, 0,0,
  0,0, 0,0, 0.02,0.02, 0,0 ;
}''')
ncfile = dynamic / "rain.nc"
subprocess.run(["ncgen", "-k", "netCDF-4", "-o", str(ncfile), str(cdl)], check=True)
case = dynamic / "case.par"
case.write_text("\n".join([
    f"DEMfile {dynamic}/dem.asc", f"startfile {dynamic}/start.asc",
    f"dynamicrainfile {ncfile}", f"resroot {dynamic}/out",
    "sim_time 0.2", "saveint 0.1", "massint 0.1", "initial_tstep 0.01",
    "fpfric 0", "max_ref_lvl 4", "epsilon 0.1", "mwdg2", "cuda",
    "drain_nodata", "",
]))
result = subprocess.run([str(binary), "-v", str(case)], text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(dynamic / "run.log").write_text(result.stdout)
if result.returncode or "Metal device:" not in result.stdout:
    raise AssertionError(f"dynamic rain app failed:\n{result.stdout}")
mass_rows = (dynamic / "out.mass").read_text().strip().splitlines()[1:]
# Reproduce the CUDA DynamicRain update order: at each logged step start, use
# the current time slice and the previous step's dt. The test raster maps the
# 3 m forcing grid to 3x3 blocks; the NoData eastern fringe excludes one of
# the two wet source columns. This independently checks interpolation timing,
# timestep carry-over, and NoData masking against the reported mass total.
depth_mm = [0.0, 0.01, 0.02]
steps = [(float(t), float(dt)) for t, dt in
         re.findall(r"t=([0-9.]+)\s+dt=([0-9.]+)", result.stdout)]
expected_rain = 0.0
previous_dt = 0.01
for time, current_dt in steps:
    slice_index = 0 if time < 0.09 else 1 if time < 0.18 else 2
    rate = depth_mm[slice_index] / 1000 / 0.09
    # Slice rates occupy source y-index 2 and both x cells; raster cells
    # with x=5 are NoData, leaving 5 x 3 = 15 receiving cells.
    if slice_index:
        expected_rain += 15 * rate * previous_dt
    previous_dt = current_dt
if not mass_rows or not math.isclose(float(mass_rows[-1].split()[11]), expected_rain,
                                      rel_tol=0, abs_tol=1e-6):
    raise AssertionError(f"dynamic rainfall pulse/NoData mass accounting failed: {mass_rows[-1:]}")
# As CUDA, the first mass record follows the first step ending at or after
# 0.1 s; it holds the rain of every step that started before that time.
first_rows = [row for row in mass_rows if float(row.split()[0]) >= 0.1 - 1e-6]
# the .mass time has three decimals: take the record's end time from the log
ends = [t + dt for t, dt in steps if t + dt >= 0.1 - 1e-6]
first_time = ends[0] if first_rows and ends else None
first_expected = 0.0
previous_dt = 0.01
for time, current_dt in steps:
    if first_time is None or time >= first_time - 1e-7:
        break
    slice_index = 0 if time < 0.09 else 1 if time < 0.18 else 2
    if slice_index:
        first_expected += 15 * (depth_mm[slice_index] / 1000 / 0.09) * previous_dt
    previous_dt = current_dt
at_first_output = first_rows[:1]
if not at_first_output or not math.isclose(float(at_first_output[0].split()[11]),
                                           first_expected, rel_tol=0, abs_tol=1e-6):
    raise AssertionError(f"dynamic rainfall first output-time interpolation failed: {at_first_output}")
final_depth = [[float(value) for value in line.split()]
               for line in (dynamic / "out-0002.wd").read_text().splitlines()[6:]]
if any(final_depth[row][col] != 0.0 for row in range(9) for col in range(5, 9)):
    raise AssertionError("dynamic rainfall added water to NoData cells")
print("PASS: adaptive Metal dynamic NetCDF rain slice update, NoData skip, and mass accounting")

# The same spatial pulse with Green-Ampt and a very shallow soil cap exercises
# persistent raster soil state while the adaptive leaves are reselected. The
# total is capped at soil_depth * dtheta per valid raster cell.
regrid_case = dynamic / "regrid.par"
# a small dam break (0.12 m on the two western columns, 0.09 m elsewhere) and
# a fine threshold make the leaves change (without the rain, whose NoData-edge
# preflag would keep this small raster at the finest level). Depths stay below
# 0.125 m: a float32 step is then 7.5e-9 m, below the 2.5e-8 m soil cap per
# cell this check measures (at 0.3 m it would be 3e-8 m).
regrid_start = [[(0.12 if col < 2 else 0.09) if value != -9999 else -9999
                 for col, value in enumerate(row)] for row in start]
write_asc(dynamic / "regrid_start.asc", regrid_start, cellsize=1)
regrid_case.write_text(case.read_text().replace(f"resroot {dynamic}/out",
                                                  f"resroot {dynamic}/regrid")
                       .replace(f"startfile {dynamic}/start.asc",
                                f"startfile {dynamic}/regrid_start.asc")
                       .replace("sim_time 0.2", "sim_time 2")
                       .replace(f"dynamicrainfile {ncfile}\n", "")
                       .replace("epsilon 0.1", "epsilon 0.001") +
    "infiltration greenampt\ngreenampt_ks 10\ngreenampt_psi 100\n"
    "greenampt_porosity 0.45\ngreenampt_theta_init 0.2\n"
    "greenampt_soil_depth 0.0000001\n")
result = subprocess.run([str(binary), "-v", str(regrid_case)], text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(dynamic / "regrid.log").write_text(result.stdout)
if result.returncode:
    raise AssertionError(f"Green-Ampt adaptive regrid app failed:\n{result.stdout}")
initial_leaves = re.search(r"native adaptive leaves: (\d+)", result.stdout)
final_leaves = re.search(r"Finished native adaptive MWDG2: leaves=(\d+)", result.stdout)
if not initial_leaves or not final_leaves or initial_leaves.group(1) == final_leaves.group(1):
    raise AssertionError("Green-Ampt run did not exercise an adaptive topology change")
loss_csv = (dynamic / "regrid.loss").read_text().strip().splitlines()
cap_volume = 45 * 0.0000001 * (0.45 - 0.2)
observed_cap_loss = float(loss_csv[-1].split(",")[2])
if not 0.9 * cap_volume <= observed_cap_loss <= cap_volume + 1e-8:
    raise AssertionError(f"Green-Ampt F/soil-cap state did not persist through regrid: {loss_csv[-1]}")
print(f"PASS: Green-Ampt soil-cap state through adaptive regrid ({initial_leaves.group(1)} -> {final_leaves.group(1)} leaves)")
