# LISFLOOD-FP 8.4: how it was built and tested

## Current handover: native adaptive Metal MWDG2, parity incomplete

The MWDG2 keyword now enters an Objective-C++ native adaptive Metal driver. On
Apple Silicon, mesh selection/compaction, neighbor construction, CFL, DG2
residuals, SSP-RK2 stages, positivity, friction, rainfall updates, loss kernels
and adaptive re-encoding run on the Apple GPU. The application does not route
this path through dense DG2 or CPU. Host code handles raster preparation and
output, orchestration, and scalar CFL/event scheduling.

Validated application features include uniform and per-cell infiltration,
Green-Ampt with soil-depth cap, evaporation, interpolated uniform rainfall,
time-varying NetCDF rainfall, NoData skipping, standard rasters, mass records,
and CUDA-schema `.loss` CSV output. `sh tests_8.4/run_metal_mwdg2_app.sh BIN`
exercises these alongside rectangular and mixed-mesh lake-at-rest cases,
including a full-precision still-water check. Component suites cover MRA,
selection, scan, mesh grading/neighbors, and an adaptive RK2 case.

Native Metal accepts `limitslopes` and `tol_Krivo`. Focused CUDA comparisons
of a level-5 still-water hill and a moving-water dam (to 1 s, and depth to
2,000 s against single-precision CUDA) agree within 3 %; see below. They do
not establish basin-scale parity.
Alzette with the limiter
previously developed non-finite stage-2 momentum and a collapsed timestep
(observed at 36.125425 s); the corrected code still needs the complete Alzette
rerun. Native Metal currently
rejects user boundary files and point sources (including the Alzette `FREE`
boundary), rainfall masks, channels/SGC/structures/routing, porosity, and
lat/long metrics. CUDA's adaptive solver has no rainfall input, so Metal rain
is an extension, checked against the fixed-grid Metal DG2 rain semantics. CUDA retains double
precision Green-Ampt state; Metal uses float hi/lo state, and coefficient,
regridding and long-run trajectory parity are not established. The six-hour
Alzette Metal run and like-for-like CUDA comparison are not complete. Do not
describe MWDG2 or the 8.4 basin matrix as fully validated until those gaps are
resolved and the six-hour outputs are checked.

#### Controlled CUDA/Metal MWDG2 comparison

The `tests_8.4/fixtures/mwdg2_cuda_lake32_l5` case is a 32x32 level-5 hill
with a still-water free surface, mixed adaptive mesh, slope limiting, and
matched 0.05 s steps to 0.2 s. Against the saved CUDA output, Metal has exact
depth agreement and maximum Qx/Qy differences of 1.32e-7 and 1.85e-7 m2/s.
Reproduce with `sh tests_8.4/run_metal_mwdg2_cuda_hill.sh BIN`. This test
exposed and fixed a Metal limiter state-representation error: CUDA stores the
limited depth slopes in its temporary flow state and restores eta=h+z when it
re-inserts the adaptive state; Metal must retain the limited eta coefficients
in its persistent tree. The Metal source also removed hydrostatic pressure
corrections that are absent from CUDA's DG2 operator, and corrected raster
projection and topography-only significance propagation.

**Moving-water dam (11 October 2026).** The 32x32 L5 dam (`epsilon 0.001`,
`limitslopes`, `tol_Krivo 1`, no friction, no boundary file) was compared with
CUDA after three corrections to the Metal step, each read from
`cuda_adaptive_simulate.cu`:

1. CUDA reads a leaf's own coefficients from its assembled solution (after
   friction and slope limiting), but its neighbours' from the scale-coefficient
   tree loaded before them (`load_soln_and_nghbr_coeffs`; `limit_slopes` writes
   only `d_assem_sol`). Metal limited in place, so neighbours saw limited
   slopes. `mwdg2_rhs` and `mwdg2_limit_slopes` now take the tree as a separate
   buffer. With the limiter on, the two sides of a face then use different
   traces, so CUDA, and now Metal, do not conserve mass exactly (upstream
   behaviour; 0.46 m3 of 3,440 m3 in `run_metal_mwdg2_limiter_app.sh`).
2. The SSP-RK2 average uses the limited stage-1 state: CUDA's
   `copy_to_buf_assem_sol` copy is overwritten by the stage-1 update, so it has
   no effect. The bed coefficients are taken from the tree in `mwdg2_rk`.
3. Rain (a Metal extension: CUDA adaptive MWDG2 has no rainfall) is integrated
   over the previous step actually taken, not over the next CFL proposal; the
   latter over-counted rain when a step was shortened to hit an output time.

Two facts about CUDA that matter for comparisons: it always starts with a
0.001 s step (`initial_tstep` is not read), and with no boundary file its edge
is not a wall: the ghost copies the depth and zeroes the normal discharge, so
water leaves wherever the edge cell has outward flow (the dam box loses about
60 % of its water in the first 100 s, on both back ends). CUDA zeroes the
boundary leaf's own slopes only in its built-in test cases, not in LISFLOOD
runs (`test_case == 0`).

| comparison (`run_metal_mwdg2_cuda_*.sh`) | before | after |
|---|---|---|
| 3 fixed 1 ms steps, step-1 depth max error | 5.0e-6 m | 1.8e-7 m |
| 100 fixed steps, flow normalized L2 error | 0.191 % | 0.003 % |
| 100 fixed steps, volume difference | 0.143 m3 | 0.0006 m3 |
| 1 s (CFL steps), depth / flow normalized L2 | 0.038 % / 0.138 % | 0.041 % / 0.146 % |

**2,000 s.** The earlier CUDA reference (`/tmp/mwdg2_cuda_2000_checkpoints`,
21 % depth error) came from a double-precision CUDA build
(`config.default.cmake`). The same case was rerun with the single-precision
release build (HPC job 37253843, `lf84/mwdg2_float2000`):

| t (s) | CUDA single vs CUDA double: depth / flow | Metal vs CUDA single: depth / flow | volume (m3): CUDA double / CUDA single / Metal |
|---|---|---|---|
| 100 | 0.26 % / 29 % | 3.24 % / 104 % | 1870.4 / 1870.0 / 1896.7 |
| 500 | 0.55 % / 104 % | 1.70 % / 177 % | 1859.8 / 1864.8 / 1891.4 |
| 1000 | 2.83 % / 142 % | 1.62 % / 145 % | 1837.4 / 1864.3 / 1890.8 |
| 2000 | 3.61 % / 114 % | 1.37 % / 182 % | 1801.5 / 1864.1 / 1889.6 |

The frictionless box sloshes at 0.1-0.4 m2/s for the whole run; after a few
hundred seconds the flow field's phase is set by rounding, and CUDA in single
precision differs from CUDA in double precision as much as Metal does. A 3 %
flow criterion at 2,000 s cannot be met by CUDA against itself and is not
used. Metal depth agrees with single-precision CUDA within 1.4-3.2 %
(previous Metal: 17-21 % against the double reference). Steps: Metal 29,638,
CUDA 29,073 (double) / 29,508 (single). The remaining 1.4 % volume difference
arises in the first 100 s, while most water leaves through the edges; it is
not yet explained. Outputs: `build/mwdg2_2000_v3`,
`build/mwdg2_tmp_evidence/` (both ignored by git).

The CUDA Alzette job 36660527 stalled in its MWDG2 run at 11.372408 seconds
with `dt=6.282032e-8` seconds. Separate diagnostic jobs also showed timestep
collapse. Diagnose that CUDA stability/time-stagnation problem before
resubmitting the six-hour CUDA matrix; verify every final timestamp and hourly
output. The completed dense Metal Alzette matrix is not a substitute for this
adaptive solver comparison.

`README.md` describes 8.4 and how to build and use it; this file records how
8.4 was developed and tested, and the results.
**8.4 adds two things to 8.3:** floodplain infiltration, evaporation and
Green-Ampt for GPU solvers; and Metal implementations for every GPU solver
selection. Metal MWDG2 now has an adaptive driver, while ACC_NUGRID and HWFV1
remain dense compatibility paths (see "Metal back end" below).

## Why

In 8.2 and 8.3 only the classic CPU solvers (and the sub-grid solver) remove
water by infiltration and evaporation (`infevap.cpp`). All GPU solvers, and the
CPU FV1/FV2/DG2 solvers, read the keywords and silently ignore them. For a
flood such as the July 2021 Alzette event, water that soaks into the ground
cannot be represented on the GPU at all.

## What 8.4 does

The CUDA and Metal GPU paths apply the loss rules at every time step: every
wet cell (deeper than `DepthThresh`, not NoData) loses
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
infiltrated so far; CUDA stores F in double precision, while Metal uses a
two-float hi/lo representation), stepped with implicit
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
- CPU FV1/FV2/DG2 still do not apply them and print a warning instead of
  ignoring them silently.
- `lisflood -version` and normal startup print only `LISFLOOD-FP 8.4` for the
  version banner.

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

## Results (8-9 October 2026, NVIDIA A100, CUDA 12.6)

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

FV1 and ACC remove exactly rate x dt. CUDA FV2 and DG2 remove it too, but
their results differ from the exact answer by about +-0.1-0.2 % of the loss (in
either direction, not proportional to the rate). The cause is the precision of
the bed elevation, not the losses: CUDA FV2 and DG2 keep it in single precision
as it is (20 m in T012, where one unit of the last digit is 1.9 um, about one
step's loss), so the per-cell rounding of the loss turns into small false
water-surface slopes and a little water moves. With the same DEM lowered to
0 m (9 October), CUDA FV2 and DG2 are exact (0.0002 mm, volume removed = depth
lost); the Metal solvers, which shift the DEM to its lowest cell, are exact on
the original test. Changing CUDA FV2 and DG2 would change their results
without the keywords, so 8.4 leaves them as in 8.3 and the tests allow 0.3 %
for them.

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

### Final checks of the release (9 October 2026)

| check | result |
|---|---|
| `tests_8.4`, CUDA (A100): T012 constant rate and Green-Ampt, FV1, FV2, DG2, ACC and the CPU ACC solver | all 78 checks pass (CUDA FV2 and DG2 to 0.3 % of the loss, see above) |
| `tests_8.4`, Metal (M1 Max): the same tests on the Metal FV1, FV2, DG2 and ACC solvers | all 77 checks pass; depths within 0.0004 mm of the expected values, volume removed = depth lost |
| 8.4 without the keywords against 8.3, CUDA, cases fv1, T01, T02, T03, T030, T031, T05, every solver, two runs of 8.4 | 27 runs each: identical to 8.3 (14 of the 54 comparisons differ only in the last digit of the `.mass` log, as between any two runs) |

Found and fixed in these checks: CUDA ACC wrote its last snapshot in the
background and did not wait for it before freeing the arrays it reads
(`snapshot.wait()` was commented out in 8.2), so occasionally (once in these tests, T03)
the last output file was cut short and the run ended with a segmentation
fault. It now waits; results are unchanged.

## Metal back end: all seven GPU solver selections (Apple Silicon)

8.3 ran only FV1 on Metal. 8.4 adds FV2 and ACC plus a native adaptive MWDG2
path and dense finest-level paths for ACC_NUGRID and HWFV1. A `.par` file with `cuda` (or `metal`)
and any of the seven GPU solver keywords runs on the Mac's GPU when
LISFLOOD-FP is built on macOS; the
build compiles one metallib per uniform operator
(`source/metal/{fv1,fv2,dg2,acc}`).

- **FV2** (`source/metal/fv2`) follows `cuda/fv2`: the same reconstruction,
  limiter, bed source and two Runge-Kutta stages, and CUDA's boundary
  treatment (ghost cells hold the outside state of each boundary face) and
  step order (rain and point sources with the previous dt, then ghost cells,
  then the time step over every cell including the ghost cells).
- **DG2** (`source/metal/dg2`) is a line-by-line port of `cuda/dg2` (nine
  coefficients per cell, friction at the Gauss points, Krivodonova limiter,
  DEM and start-file slopes from `generateDG2DEM`/`generateDG2start`). One
  deliberate difference: CUDA limits the slopes in place, so a cell may see a
  neighbour's slope before or after it is limited; Metal limits into a
  separate buffer, which makes the result independent of thread timing.
- **ACC** (`source/metal/acc`) follows the CPU ACC solver, with a conservative
  dry-cell limiter that keeps each face flux shared by its two cells (CUDA ACC
  does not).
- As for FV1, each face flux is computed once and shared by the two cells
  (overlapping threadgroup tiles), and the kernels are compiled with
  `-fno-fast-math`, so every run repeats bit for bit.
- Infiltration, evaporation and Green-Ampt: the same rules as CUDA, in
  `source/metal/losses` (shared by all seven Metal solver paths). Metal GPUs have no
  double precision, which CUDA uses for the Green-Ampt depth F; Metal keeps F
  as a pair of floats (exact two-sum) and computes each step's increment in a
  form without cancellation. ACC applies the losses where the CPU ACC solver
  does (after the dry check, before the depth update); FV1, FV2 and DG2 after
  the step, as CUDA.
- With `-v` the Metal solvers print one `t= dt=` line per time step, as the
  CUDA solvers do (the 8.4 tests rebuild the expected depths from it).

### ACC_NUGRID and HWFV1 dense compatibility; MWDG2 native adaptive path

The `acc_nugrid`, `hwfv1` and `mwdg2` keywords dispatch to Metal on macOS.
ACC_NUGRID uses the Metal ACC operator and HWFV1 uses the Metal hydrostatic
FV1/HLL operator; both remain dense finest-level compatibility paths without
adaptive compression. MWDG2 uses the native adaptive Metal tree and RKDG2
driver described in the current handover above. Its raster losses and uniform
or NetCDF dynamic rainfall are applied on the GPU, and loss-enabled runs emit
the CUDA-style `.loss` diagnostic. User-provided boundary files and point
sources, `limitslopes`, and several routing/channel features are explicitly
rejected. See the handover for tested scope and remaining CUDA parity gaps.

CUDA MWDG2 constructs modal coefficients directly from the input rasters.
Ordinary Metal DG2 instead expects `generateDG2DEM` and `generateDG2start`
outputs. For MWDG2 only, missing `1x`/`1y` files now select a conservative
piecewise-constant projection: the mean comes from the raster and unresolved
slopes start at zero. Supplied modal files are still used. This removed the
preprocessing failure without changing the existing `dg2` path.

Validation on the Apple M1 Max used
`python3 tests_8.4/metal_adaptive.py build/macos/lisflood`. All three paths built,
ran a 16 x 16 moving-water case with infiltration for 120 s, wrote all three
expected depth rasters, and kept every depth finite and nonnegative (HWFV1
range 0.100--0.133 m; MWDG2 0.099--0.144 m; ACC_NUGRID 0.100--0.133 m).
This older smoke run used dense compatibility drivers and is not evidence for
the current native MWDG2 tree or equality with compressed CUDA output.

The full 100 m Alzette six-hour matrix, including separate uniform infiltration,
evaporation and Green-Ampt runs for all seven keywords, is recorded in
`tests_8.4/ALZETTE_METAL_RESULTS.md`.

Lessons from the port (they apply to any new back end):
- Copying a host from another solver copies its boundary rules too. Metal
  FV2 first kept FV1's boundary handling and lost water at a fixed-discharge
  inflow (T031): its first steps ignored the inflow's speed and were 30 times
  too long. Taking the ghost-cell states and the time-step order from CUDA
  fixed this and made five of six small cases bit-identical.
- Compare with the reference on the same input files. `generateDG2DEM` and
  `generateDG2start` built on macOS and on Linux write slightly different
  DEMs and start files; the first T05 comparisons differed for that reason
  alone (and for every solver, FV1 included).

### Metal results (9 October 2026, Apple M1 Max, against CUDA 8.3 on an A100)

Inputs as used on the HPC. "Identical" counts wet cells with the same depth to
the output precision (1 mm).

| case | FV2 | DG2 |
|---|---|---|
| T01, T031 | identical, same number of steps | identical, same number of steps |
| T030 (dynamic rain) | 87.5 % identical, max 1 mm | identical |
| T02 | identical | 99.99 % identical, max 1 mm |
| T03 | identical (max 1 mm) | identical for 4 outputs, then 54 % (CPU DG2 vs CUDA: 45 %) |
| T05 (dam break, 0.32 M cells) | 99.99 %, max 4 mm, same steps (4733) | 86 % (CPU DG2 vs CUDA: 85 %), volume within 0.002 % |
| T07s (Carlisle, 6 h) | all outputs within 1 mm, volume within 0.0003 % | 99.6-99.9 % within 1 cm up to 4 h; then CUDA and Metal both blow up (below) |
| T06 (EA test 5, 1.7 M cells, 30 h) | 99.95 % within 1 cm, volume within 0.016 %, steps 230 266 vs 230 264 | 99.85 % within 1 cm, volume within 0.003 %, steps 262 559 vs 262 584 |

ACC (against the CPU ACC solver, as Metal ACC follows the CPU): T01, T02,
T030 and T031 bit-identical; T07s 99.5 % identical, max 1 mm; T05 71 % and T06
75 % (CUDA ACC against the CPU on the same cases: 13 %, unstable, and 42 %).

Every Metal run repeats bit for bit (FV2 and DG2 checked on T02, T03, T031 and
on the 2853 x 1833 grid T07x3).

Where Metal and CUDA DG2 differ (T03, T05), the difference starts at rounding
level (the first steps agree to 4-5 digits) and grows in these sensitive
cases, as it does between the CPU and CUDA DG2 solvers.

Found in the original solvers while porting (to report, not changed in 8.4):
- CUDA MWDG2 on the Alzette 100 m basin (DEM padded to 512 x 512) collapses
  to time steps of 1e-6 to 1e-8 s within the first 5-14 s of simulated time,
  with and without the loss keywords (loss-free run: 6.3e-8 s at 11.4 s), so
  the cause is in the adaptive solver, not in the losses; a smaller initial
  time step does not avoid it. CUDA HWFV1 on the same basin writes small
  negative depths (down to -0.007 m) in about 104 000 cells, also without
  losses, and its adaptive path does not read `dynamicrainfile`.
- CUDA adaptive MWDG2 (`cuda/adaptive`), found while matching Metal to it:
  with no boundary file the domain edge is not a wall (ghost depth copied,
  normal discharge set to 0, so outward flow leaves the domain); with
  `limitslopes`, a leaf uses its own limited slopes but its neighbours'
  unlimited ones, so the face fluxes of the two sides differ and mass is not
  conserved exactly; `initial_tstep` is not read (first step always 0.001 s);
  there is no rainfall input; with `epsilon 0` the written depth raster is the
  domain mean (seen on two fixtures). Metal follows the first three to match
  CUDA.
- CUDA DG2 on Carlisle (T07s) blows up after about 4 h (volume from 1.4 M m3
  to 1e10-1e11 m3), preceded by negative depths down to -0.15 m; Metal DG2
  does the same.
- CUDA DG2 and Metal DG2 both write depths down to -0.86 m on EA test 5 (T06).
- CPU DG2 differs strongly from CUDA DG2 on T031 (fixed-discharge inflow with
  NoData cells): 40 % identical, volume 29 % apart, 1283 against 219 steps.
- CUDA FV2 slope limiter: r = (local - forward) / (local - backward), which
  gives non-zero slopes only at local extrema; ported unchanged.
- CUDA applies dynamic rain and point sources with the previous step's dt
  (zero before the first step); ported unchanged.

## Problems met while building 8.4, and how each was solved

Kept here so that whoever extends 8.4 does not meet them again.

| # | problem | how it showed | solution |
|---|---|---|---|
| 1 | Most source files have CRLF line endings; editing them with tools that write LF silently converts whole files | diffs of thousands of lines | read and write with the line endings preserved; check with `file` before committing |
| 2 | The loss per step is often close to the resolution of a single-precision depth (a few um on metres of water), so it is rounded away | depths fell too slowly; small rates did nothing | compensated subtraction: each cell keeps the part of its loss that rounding dropped (`residual`) and adds it to its next step |
| 3 | The GPU model clock (single precision) falls behind the sum of the time steps on long runs | expected depth `h0 - rate t` wrong by more than the tolerance | tests compute the expected depth from the steps the model logged (`-v` prints one `t= dt=` line per step; the Metal solvers now do too) |
| 4 | DG2: a cell on a wet/dry edge whose loss makes a face dry, with discharge but no depth there | the time step collapsed to zero | if a face would become dry, the cell is made first order (all slopes zero) |
| 5 | Apple GPUs have no double precision, which CUDA uses for the Green-Ampt depth F; `F_new - F` in float cancels | Green-Ampt wrong by percent in float | F kept as a pair of floats (exact two-sum), and the increment written without cancellation (`metal_losses_kernels.metal`, `greenampt_step`) |
| 6 | Metal FV2 started from the FV1 host, whose boundary handling differs from CUDA FV2 | water lost at a fixed-discharge inflow (T031): the first steps ignored the inflow speed and were about 30 times too long | port CUDA's ghost-cell states and step order exactly (ghost cells, then the time step over every cell including them). Lesson: copying a host copies its boundary rules |
| 7 | Metal allows 31 buffer arguments per kernel; a DG2 stage reads three sets of 9 coefficients and writes one | does not fit | one buffer of 9 planes per coefficient set |
| 8 | CUDA DG2 limits slopes in place: a cell may read a neighbour's slope before or after it is limited | results could depend on thread timing | Metal limits into a separate buffer and copies back |
| 9 | `generateDG2DEM`/`generateDG2start` built on macOS and Linux write slightly different files | every solver, FV1 included, differed from CUDA on T05 | compare back ends only on the same input files (copy the HPC's `work/data`) |
| 10 | Metal writes output 0 after the first step, CUDA before it | output 0 always differs | comparisons skip output 0 |
| 11 | CUDA FV2 and DG2 keep the bed elevation in single precision as given (20 m in T012) | losses 0.1-0.2 % off, water created or destroyed | proven with the DEM lowered to 0 m (exact); left as in 8.3 (changing it would change results without the keywords); tests allow 0.3 % for them; Metal shifts the DEM to its lowest cell and is exact |
| 12 | CUDA ACC wrote its last snapshot in the background and freed the arrays without waiting (`snapshot.wait()` commented out in 8.2) | once: segmentation fault, last output cut short | `snapshot.wait()` restored |
| 13 | Test-checker mistakes: a pattern for T012 also matched the Green-Ampt runs; FV1's `.mass` volume includes the ghost cells; the GPU's first `.mass` row holds the whole volume; ACC writes output 0 after its first step | false failures | fixed in `tests_8.4/check.py` (Verror not checked where the volume includes ghost cells) |
| 14 | Long runs on a laptop stop when it sleeps | runs stalled overnight | run with `caffeinate -i` |
| 15 | On the ECMWF HPC, `sbatch` drops empty arguments, and scratch is purged when unused | jobs got wrong arguments; data at risk | pass `all`, never `""`; archive results to `$PERM` |

## Handover: what remains

CUDA and Metal solver selections support infiltration, evaporation and
Green-Ampt, with the adaptive-Metal MWDG2 limits listed above. Metal MWDG2
also accepts uniform and dynamic NetCDF rainfall. Metal ACC_NUGRID and HWFV1
remain dense compatibility paths. The sub-grid channel model remains CPU-only.

### Completed: losses in CUDA MWDG2 and HWFV1 (9 October 2026)

`Simulation::run` now receives the shared `Fnames`, `States`, `Pars` and
`Solver`, so loss keywords use the same parsing as the uniform-grid solvers.
`Losses` retains parameters, compensation residuals and double-precision
infiltrated depth F in the original finest raster layout. Morton coordinates
are converted to raster coordinates (including the reversed row order), and
padding and DEM NoData are excluded.

After the full FV1 or two-stage DG2 update, each leaf evaluates the shared
loss equations for its finest children at the leaf's mean water depth. Their
depth changes are averaged into the leaf; compensation includes the rounding
of that average. F remains separate for every child through merges and splits.
This differs from the original handover's proposal to average soil parameters:
Green-Ampt is nonlinear, and averaging parameters would mix sealed soil and
soil with conductivity or finite storage. A one-time refinement flag preserves
all spatial soil-parameter and validity-mask changes, so a flat surface cannot
hide a nonuniform soil field by merging the entire basin before losses begin.
HWFV1 scales discharge; MWDG2 applies the existing DG2 slope and dry-face rules.
For loss-enabled runs, a cache indexed by tree node preserves the depth of
leaves that survive adaptation unchanged, including their neighbour copies.
It includes point-source increments inserted before encoding. This avoids
repeated h -> eta -> h rounding near high NoData/padding walls; newly merged
or split leaves still use the multiresolution reconstruction. The cache costs
one depth and one integer generation counter per tree node.

The adaptive `.cumu` file contains timings, not a mass balance. Loss-enabled
runs therefore write a separate `.loss` CSV at `massint`: simulation time,
double-precision integrated time, cumulative volume removed and current water
volume. Storage plus losses should be constant in a closed basin, within the
underlying flow solver's conservation error. This file does not correct for
boundary flows or rainfall, and no existing output format changes when losses
are disabled.

Use `max_ref_lvl >= 5`, even for smaller rectangular DEMs. The legacy tree
traversal uses full CUDA blocks and corrupts memory for smaller padded meshes;
loss-enabled runs reject them. Tests use short relative filenames because the
legacy adaptive keyword readers copy unrelated string values into small
buffers while scanning a parameter file. These legacy routines are unchanged.

Run the standalone GPU checks (new WORK directory required):
```
python3 tests_8.4/adaptive.py BIN84 BIN83 WORK
```
They cover constant and grid rates, evaporation, drying, Green-Ampt numeric
versus grid parameters, a finite soil-storage cap, north/south orientation,
rectangular padding and NoData (HWFV1), changing meshes, repeatability and
loss-disabled equality with CUDA 8.3. Moving-case volume balance is compared
with the loss-free solver's drift; the uniform closed basins use a 0.03 %
relative volume tolerance. The full results, including the remaining
adaptive-flow conservation error, are in `tests_8.4/ADAPTIVE_RESULTS.md`.

### CUDA ACC on a non-uniform grid (`acc_nugrid`)

Constant-rate infiltration, evaporation and Green-Ampt are also applied by
CUDA `acc_nugrid`. Its static mesh has cells at several sizes. Soil state and
rounding compensation remain on the finest input raster; each active leaf
stores the average loss of its valid finest children. Soil-parameter changes,
NoData boundaries and rectangular padding are preflagged before the mesh is
built. ACC changes depth only.

The solver writes the same `.loss` CSV as MWDG2/HWFV1 and puts cumulative
infiltration plus evaporation in the final `.mass` column. Use `L >= 5` for
this legacy solver (`max_ref_lvl` is the MWDG2/HWFV1 keyword). Its existing
input path also requires a `stagefile`. A pre-existing double close of the
stage file was fixed; valid loss-disabled runs retain their 8.3 numerical
outputs.

Run the standalone checks on a GPU node:
```
python3 tests_8.4/acc_nugrid.py BIN84 BIN83 WORK
```
They cover constant and raster rates, evaporation, Green-Ampt numeric and
grid parameters, a finite soil-storage cap, raster orientation, compressed
mixed-size cells, rectangular padding, NoData, drying, repeatability, the
small-tree guard and loss-disabled equality with CUDA 8.3. The final clean
A100 run passed all 106 checks; details and remaining limits are in
`tests_8.4/ACC_NUGRID_RESULTS.md`.

### Native adaptive MWDG2 on Metal: current status

The native driver is wired into LISFLOOD and has app-level tests for rectangular
and mixed-mesh lake-at-rest cases, CFL-driven steps, uniform/grid infiltration,
Green-Ampt and its cap across a changing tree, evaporation, uniform and dynamic
rain, NoData handling, combined mass accounting, standard output and `.loss`
CSV. The GPU runs mesh selection, compaction, neighbor construction, flow/RK,
friction, losses and rainfall; host code handles input/output, orchestration,
and scalar schedule/CFL work.

`limitslopes`/`tol_Krivo` are implemented and accepted, with focused Metal
app coverage. A controlled 32x32 L5 dam fixture compares CUDA and Metal
rasters for three RK2 steps (0.003 s total) at epsilon=0.001, with identical
initial active leaf IDs (184). Through step 3, maximum depth error is
2.45e-5 m; maximum Qx error grows from 2.13e-5 at step 1 to 4.17e-5 at
step 2 and 1.17e-3 at step 3. Qy maximum error stays below 2.75e-8. The
regression and CUDA snapshots are in
`tests_8.4/run_metal_mwdg2_cuda_parity.sh` and
`tests_8.4/fixtures/mwdg2_cuda_z1_l5_e001/`. This short raster-level
comparison does not establish long-run coefficient or trajectory parity.

The same dam fixture was followed for 100 fixed steps, for 1 s and for
2,000 s; see "Moving-water dam" in the current handover above. The short
trajectory diagnostic is `tests_8.4/run_metal_mwdg2_cuda_trajectory.sh`.

The 32x32/L5 lake-at-rest hill case is now compared directly against saved
CUDA output in `tests_8.4/run_metal_mwdg2_cuda_hill.sh`. After fixing the Metal
limiter's persistent eta representation, removing pressure corrections not
present in CUDA, and correcting topography significance and raster projection,
the case selects the same 16-leaf tree and remains stationary through 0.2 s:
depth agrees exactly at the saved precision, with maximum Qx/Qy differences
below 1.9e-7 m2/s. This closes that specific hill case; it does not establish
moving-water trajectory, coefficient, or basin-scale parity.

Short moving-water comparisons agree closely; at 2,000 s Metal depth is
within 1.4-3.2 % of single-precision CUDA, while the sloshing flow field
differs as much as CUDA single from CUDA double. The Alzette validation below
remains unresolved, so these results do not establish basin-scale validation.

Production gaps remain: user boundary files and point sources, rainfall masks,
channels/SGC/structures/routing, porosity and lat/long metrics are rejected.
CUDA uses double precision for cumulative Green-Ampt F while Metal uses two
float components. Full adaptive leaf-set and long-trajectory parity remain
unverified. CUDA adaptive MWDG2 has no rainfall input at all, so Metal
rain follows the fixed-grid Metal DG2 semantics rather than adaptive CUDA
output. A 10-second Alzette smoke run completed, but does not validate basin
behavior: it used the full finest mesh, had a signed leaf-volume change of
about -466 m3, and reconstructed negative depths totaling about 769,349 m3.
The prepared Alzette `N 75641 75666 FREE` segment maps to an empty range under
the current input index conversion, so that run used reflecting boundaries.
The six-hour Alzette run and matched CUDA basin comparison remain unresolved;
the CUDA reference itself previously collapsed its timestep near 11 seconds.

### Where things are, and keeping 8.3 untouched

The exact folders on the development machines (the 8.3 and 8.4 sources and
zips, the Metal and CUDA builds, the reference outputs and prepared inputs on
the HPC, the archive) are listed in `HANDOVER_LOCAL.md`, which is kept next to
this source and out of the release. In general:

- **8.3 is frozen.** Its release, its git history, its build used for the
  8.3 paper, its test runs and its reference outputs are read only: read,
  copy and compare, never edit, rebuild in place or write into them.
- **Work in a copy of 8.4**, with new build and output folders, and new run
  labels; never reuse a label of an existing reference run.
- **Compare on the same inputs**: copy the prepared test inputs of the
  reference runs (with their `.prepared` marker) instead of regenerating them.
- **The CUDA references** are the 8.3 builds and their outputs; the CPU
  references are the CPU solvers of the same version.

### Rules that held throughout

- Without the new keywords, results must stay identical to 8.3, byte for
  byte (`tests/run_suite.sh` + `hash_outputs.sh` + `compare_hashes.py`).
- Small test first, then the full set; check mass balance and repeatability.
- Compare against a reference on the same inputs, and report the spread
  between existing back ends (CPU vs CUDA) next to the new one.
- Defects found in the existing solvers are reported, not silently changed
  (list above).

## Plan

1. Constant-rate infiltration and evaporation in CUDA FV1, FV2, DG2 and ACC (this
   version), tested as above.
2. Done: Green-Ampt infiltration (`infiltration greenampt`), with
   soil-property and initial-moisture grids, tested as above.
   Next for real runs: Ks = 0 on river, lake and sealed cells; parameter
   grids from the soil map and soil moisture (Geoportal or WRF `SMOIS`).
   Possible later: recovery between rain events (SWMM-style, drying rate
   from evapotranspiration).
3. Done: every GPU solver keyword runs on Metal, and infiltration, evaporation
   and Green-Ampt work in each GPU path. MWDG2 now has a native adaptive Metal
   driver plus rain and losses, with the unsupported modes and parity gaps
   listed above; ACC_NUGRID and HWFV1 remain dense compatibility paths.
4. Done: infiltration, evaporation and Green-Ampt in CUDA MWDG2, HWFV1 and
   non-uniform-grid ACC. Continue MWDG2 CUDA parity, implement remaining
   boundary/source support, and complete basin validation before claiming full
   solver support.

Known limits, as in the CPU solvers: water shallower than `DepthThresh` does
not drain; infiltration is a constant rate (step 2 addresses this).
