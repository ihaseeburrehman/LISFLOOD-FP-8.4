# LISFLOOD-FP 8.4 (in development): infiltration and evaporation on the GPU

This folder is a working copy of LISFLOOD-FP 8.3 (`README.md` describes 8.3).
**8.4 adds one thing:** floodplain infiltration and evaporation for the CUDA
FV1 and CUDA ACC solvers. Nothing else changes. It is developed in parallel
with the 8.3 paper and will be released only if every test below passes.

## Why

In 8.2 and 8.3 only the classic CPU solvers (and the sub-grid solver) remove
water by infiltration and evaporation (`infevap.cpp`). All GPU solvers, and the
CPU FV1/FV2/DG2 solvers, read the keywords and silently ignore them. For a
flood such as the July 2021 Alzette event, water that soaks into the ground
cannot be represented on the GPU at all.

## What 8.4 does

The CUDA FV1 and ACC solvers apply the CPU rules at every time step: every wet
cell (deeper than `DepthThresh`, not NoData) loses
`(infiltration rate + evaporation rate) x dt` of depth, never below zero.

| `.par` keyword | meaning | as in the CPU solvers |
|---|---|---|
| `infiltration <rate>` | uniform infiltration rate, m/s | yes |
| `infilfile <grid>` | infiltration rate per cell, ESRI ASCII grid in mm/h, same size as the DEM (like `manningfile`); NoData and negative values count as 0; used instead of `infiltration` | same file format (the CPU uses it only in the sub-grid solver) |
| `evaporation <file>` | evaporation time series in mm/day, uniform in space | yes, same file and loader (`LoadEvap`) |

- **FV1** stores discharge, so when a cell loses depth its discharges are
  scaled with it (velocity kept), and set to zero when the cell becomes dry.
- **ACC** stores discharges on the cell faces; only the depth changes.
- **Mass balance:** the volume removed is summed on the GPU. In the `.mass`
  file, `Verror` no longer counts it as an error, and the last column
  (`Rain-(Inf+Evap)`, always 0 on the GPU before) shows minus the volume
  removed since the start (rain is not counted in this column).
- **Without these keywords** 8.4 computes exactly what 8.3 computes; outputs,
  `.mass` file included, are identical byte for byte.
- Solvers that do not apply them (CUDA FV2, DG2, ACC-nugrid, adaptive; Metal
  FV1; CPU FV1/FV2/DG2) now print a warning instead of ignoring them silently.
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
| cost | about 11-20 % more run time |

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

## Plan

1. Constant-rate infiltration and evaporation in CUDA FV1 and ACC (this
   version), tested as above.
2. Then, if 1 works: Green-Ampt infiltration, started from a soil-moisture
   grid (e.g. WRF `SMOIS` at the start of the run) and soil-property grids
   (saturated conductivity, suction, porosity), with the infiltrated depth
   tracked per cell; tested against the exact Green-Ampt solution.
3. Possibly later: the same in Metal FV1; gridded evaporation.

Known limits, as in the CPU solvers: water shallower than `DepthThresh` does
not drain; infiltration is a constant rate (step 2 addresses this).
