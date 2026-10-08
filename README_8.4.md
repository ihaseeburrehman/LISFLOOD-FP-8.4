# LISFLOOD-FP 8.4 (in development): infiltration and evaporation on the GPU

This folder is a working copy of LISFLOOD-FP 8.3 (`README.md` describes 8.3).
**8.4 adds one thing:** floodplain infiltration and evaporation for the CUDA
FV1, FV2, DG2 and ACC solvers. Nothing else changes. It is developed in parallel
with the 8.3 paper and will be released only if every test below passes.

## Why

In 8.2 and 8.3 only the classic CPU solvers (and the sub-grid solver) remove
water by infiltration and evaporation (`infevap.cpp`). All GPU solvers, and the
CPU FV1/FV2/DG2 solvers, read the keywords and silently ignore them. For a
flood such as the July 2021 Alzette event, water that soaks into the ground
cannot be represented on the GPU at all.

## What 8.4 does

The CUDA FV1, FV2, DG2 and ACC solvers apply the CPU rules at every time step: every wet
cell (deeper than `DepthThresh`, not NoData) loses
`(infiltration rate + evaporation rate) x dt` of depth, never below zero.

| `.par` keyword | meaning | as in the CPU solvers |
|---|---|---|
| `infiltration <rate>` | uniform infiltration rate, m/s | yes |
| `infilfile <grid>` | infiltration rate per cell, ESRI ASCII grid in mm/h, same size as the DEM (like `manningfile`); NoData and negative values count as 0; used instead of `infiltration` | same file format (the CPU uses it only in the sub-grid solver) |
| `evaporation <file>` | evaporation time series in mm/day, uniform in space | yes, same file and loader (`LoadEvap`) |

**Green-Ampt infiltration**, a separate choice (`infiltration greenampt`
instead of a rate; using both stops with an error). Each parameter is a number
for the whole domain or a grid file like the Manning file:

| `.par` keyword | meaning |
|---|---|
| `infiltration greenampt` | use Green-Ampt |
| `greenampt_ks` | saturated hydraulic conductivity, mm/h |
| `greenampt_psi` | wetting-front suction head, mm |
| `greenampt_porosity` | saturated moisture content, m3/m3 |
| `greenampt_theta_init` | initial moisture content, m3/m3 (e.g. from soil-moisture data) |
| `greenampt_soil_depth` | optional: soil depth, m; infiltration stops when it is full |

Every cell with water infiltrates at `f = Ks (1 + (psi + h) dtheta / F)`
(`dtheta` = porosity - initial moisture, `h` the ponded depth, `F` the depth
infiltrated so far, kept per cell in double precision), stepped with implicit
Euler, which has a closed form per step and starts from `F = 0`; never more
than the water present, and not beyond `soil depth x dtheta`. Evaporation
works with either model. The soil does not recover between rain events.
**Cells under water at the start need care:** rivers and lakes should get
`Ks = 0`, and soils already flooded at the start should get an initial moisture
close to saturation (`theta_init` = `porosity`); sealed surfaces `Ks = 0`.
Otherwise they infiltrate as if dry soil (see the Alzette test below).

- **FV1** stores discharge, so when a cell loses depth its discharges are
  scaled with it (velocity kept), and set to zero when the cell becomes dry.
- **ACC** stores discharges on the cell faces; only the depth changes.
- **FV2 and DG2** store slopes besides the cell averages. The loss lowers the
  average depth and keeps the depth slopes (a flat water surface over sloping
  ground stays flat); discharge average and slopes are scaled with the depth.
  If a face would become dry (a cell on a wet/dry edge), the cell is made first
  order (all slopes zero): a face with discharge but no depth stopped DG2's
  time step in the first version.
- **Mass balance:** the volume removed is summed on the GPU. In the `.mass`
  file, `Verror` no longer counts it as an error, and the last column
  (`Rain-(Inf+Evap)`, always 0 on the GPU before) shows minus the volume
  removed since the start (rain is not counted in this column).
- **Without these keywords** 8.4 computes exactly what 8.3 computes; outputs,
  `.mass` file included, are identical byte for byte.
- Solvers that do not apply them (CUDA ACC-nugrid and adaptive; Metal FV1; CPU
  FV1/FV2/DG2) now print a warning instead of ignoring them silently.
- Start-up line: `LISFLOOD-FP 8.4`.

Code: `source/cuda/cuda_losses.cu` (new), calls in
`source/cuda/fv1/cuda_fv1_simulate.cu` and `source/cuda/acc/cuda_acc_simulate.cu`,
mass balance in `source/cuda/cuda_stats.cu` and `source/cuda/stats.cpp`,
warning in `source/lisflood.cpp`. `git diff 387f829` shows every change.

## How it is tested (`tests_8.4/`)

| test | what must hold |
|---|---|
| T012 (LISFLOOD-FP's own evaporation and infiltration test: flat basin, 0.4464 m of water, 3.6 mm/day evaporation + 1e-8 m/s infiltration, 100 days) | depth falls linearly and reaches 0 at 100 days, on CUDA FV1, CUDA ACC and the CPU ACC solver |
| T012 without the keywords | depth stays 0.4464 m |
| T012 with evaporation rising linearly from 0 to 7.2 mm/day | depth after 50 and 100 days as the integral of the rate (90 and 360 mm lost) |
| T012 with a wall in the middle and `infilfile` giving 0.3 mm/h on the left half only | left half falls at 0.3 mm/h, right half unchanged |
| mass balance | volume lost (last `.mass` column) = depth lost x area; `Verror` about 0 |
| Alzette 10 m, 6 h, FV1 and ACC, without the keywords | identical to 8.3, byte for byte |
| Alzette 10 m, 6 h, uniform `infiltration` vs `infilfile` with the same rate in every basin cell | identical results (checks the grid loader on a real grid) |
| Alzette 10 m, 6 h, with infiltration and evaporation | runs, less water than without, mass balance closes |

## Results (8 October 2026, NVIDIA A100, CUDA 12.6)

| test | result |
|---|---|
| T012, CUDA FV1 and ACC: depth against the sum of rate x dt over the steps taken | within 0.0004 mm (the last written digit); all checks pass |
| T012 with the wall: left half drains, right half unchanged | yes, to 0.0000 mm on the right |
| T012 and Alzette (6 h, FV1 and ACC) without the keywords: 8.4 vs 8.3 | output grids identical byte for byte |
| Alzette FV1, uniform 2 mm/h vs `infilfile` 2 mm/h | identical (grid loader checked on 10.6 M cells) |
| Alzette FV1 with infiltration (and evaporation) | 3.25 (3.44) million m3 removed in 6 h; no cell deeper than without |
| CUDA ACC, Carlisle (6 h) and EA test 5 (6 h), 5 mm/h, against the CPU ACC solver | water removed by infiltration agrees within 1.9 % and 0.7 %; time step and maximum depth unchanged |
| T012, CUDA FV2 and DG2 (base, ramp, wall, half rate) | within 0.07-0.21 % of the loss, either sign (0.15-0.76 mm of 223-360 mm); without the keywords identical to 8.3 |
| CUDA FV2 and DG2, EA test 5 (6 h), 5 mm/h | stored water falls by the volume removed within 0.1 % (FV2) and 0.5 % (DG2); without the keywords identical to 8.3 |
| Green-Ampt, flat basin with 0.1 m of water (Ks 10 mm/h, psi 110 mm, dtheta 0.23), 1 day, FV1, FV2, DG2, ACC | depth matches Green-Ampt as stepped within 0.0009 mm and the continuous Green-Ampt solution within 0.47 mm (the implicit-Euler error of the 60 s test steps; 0.01 mm at 1 s); volume removed = depth lost (364 500 m3) |
| Green-Ampt parameters as grid files vs numbers | identical output |
| Green-Ampt with soil depth 0.2 m | infiltration stops at 46 mm (0.054 m of water left) |
| Green-Ampt with Ks on the left half only (wall in between) | right half unchanged |
| `infiltration greenampt` and a rate in one file | stops with an error |
| Alzette FV1, 6 h: none / constant 2 mm/h / Green-Ampt (Ks 10 mm/h, theta_init 0.30 everywhere) | stored water after 6 h 27.4 / 24.4 / 12.6 million m3; removed 0 / 3.2 / 16.0 million m3; time step, maximum depth unchanged, no cell deeper. Of the 14.8 million m3 less water than without infiltration, 12.5 million are in cells already under water at the start (5.7 in rivers and the lake deeper than 0.5 m, 6.8 on floodplain flooded 1 cm to 0.5 m by the previous day), which were given the same dry initial moisture as everywhere else; only 2.3 million in cells dry at the start |
| cost | about 11-20 % more run time (Green-Ampt: none measurable on Alzette) |

FV1 and ACC remove exactly rate x dt. FV2 and DG2 remove it too, but their
results differ from the exact answer by about +-0.1-0.2 % of the loss (in
either direction, not proportional to the rate): their closed walls exchange a
little water while the water level changes. This is accepted and documented.

Found on the way (fixed in 8.4): depth is single precision, so small losses
per step were rounded the same way every step (3.7 mm off after 50 days in
T012; in a 5 m deep cell the loss per step was 1.8 times too large); 8.4 keeps
the rounding remainder per cell (compensated subtraction).

Found on the way, already in 8.2/8.3 and not changed here:

- The GPU clock (`solver.t`) is single precision and falls behind the sum of
  the time steps on long runs (22 min after 50 days in T012).
- CUDA ACC is unstable on the Alzette 10 m set-up even without losses (depths
  of 270 m); possibly the land-use Manning values are too low for ACC there.
- The CPU solvers' infiltration and evaporation totals in `.mass` stop growing
  (single-precision accumulator: stuck at 2048 and 512 m3 in the tests above),
  although the water is removed correctly.
- The CUDA `.mass` file: FV1 volume includes the ghost cells; the first row
  always shows the whole volume as `Verror`.
- CUDA DG2 writes negative depths on EA test 5 even without losses (down to
  -0.86 m in 27 000 cells after 6 h; total volume still conserved).
- CUDA ACC on Alzette: stable with a uniform Manning's n of 0.05, unstable with
  the land-use map (446 000 cells at n = 0.016); ACC is known to become
  unstable at low friction on fine grids.

## Plan

1. Constant-rate infiltration and evaporation in CUDA FV1, FV2, DG2 and ACC (this
   version), tested as above.
2. Done: Green-Ampt infiltration (`infiltration greenampt`), with
   soil-property and initial-moisture grids, tested as above.
   Next for real runs: Ks = 0 on river, lake and sealed cells; parameter
   grids from the soil map and soil moisture (Geoportal or WRF `SMOIS`).
   Possible later: recovery between rain events (SWMM-style, drying rate
   from evapotranspiration).
3. Possibly later: the same in Metal FV1; gridded evaporation.

Known limits, as in the CPU solvers: water shallower than `DepthThresh` does
not drain; infiltration is a constant rate (step 2 addresses this).
