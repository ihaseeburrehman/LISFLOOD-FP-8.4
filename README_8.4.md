# LISFLOOD-FP 8.4: how it was built and tested

`README.md` describes 8.4 and how to build and use it; this file records how
8.4 was developed and tested, and the results.
**8.4 adds two things to 8.3:** floodplain infiltration and evaporation for the
CUDA and Metal FV1, FV2, DG2 and ACC solvers, and Metal (Apple Silicon)
versions of the FV2, DG2 and ACC solvers, so that every solver 8.3 runs on an
NVIDIA GPU also runs on a Mac (see "Metal back end" below). Nothing else
changes.

## Why

In 8.2 and 8.3 only the classic CPU solvers (and the sub-grid solver) remove
water by infiltration and evaporation (`infevap.cpp`). All GPU solvers, and the
CPU FV1/FV2/DG2 solvers, read the keywords and silently ignore them. For a
flood such as the July 2021 Alzette event, water that soaks into the ground
cannot be represented on the GPU at all.

## What 8.4 does

The CUDA and Metal FV1, FV2, DG2 and ACC solvers apply the CPU rules at every time step: every wet
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
- Solvers that do not apply them (CUDA ACC-nugrid and adaptive; CPU
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

## Metal back end: FV2, DG2 and ACC (Apple Silicon)

8.3 ran only FV1 on Metal. 8.4 adds the other three GPU solvers. As in 8.3, a
`.par` file with `cuda` (or `metal`) and `fv2`, `dg2` or `acceleration` runs on
the Mac's GPU when LISFLOOD-FP is built on macOS; the build compiles one
metallib per solver (`source/metal/{fv1,fv2,dg2,acc}`).

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
  `source/metal/losses` (one kernel for the four solvers). Metal GPUs have no
  double precision, which CUDA uses for the Green-Ampt depth F; Metal keeps F
  as a pair of floats (exact two-sum) and computes each step's increment in a
  form without cancellation. ACC applies the losses where the CPU ACC solver
  does (after the dry check, before the depth update); FV1, FV2 and DG2 after
  the step, as CUDA.
- With `-v` the Metal solvers print one `t= dt=` line per time step, as the
  CUDA solvers do (the 8.4 tests rebuild the expected depths from it).

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

## Handover: what to do next

The four uniform-grid GPU solvers (FV1, FV2, DG2, ACC) apply infiltration,
evaporation and Green-Ampt on CUDA and Metal. Not yet covered: the CUDA
adaptive solvers MWDG2 and HWFV1 (`source/cuda/adaptive`, keywords `mwdg2`,
`hwfv1`), the CUDA ACC solver on a non-uniform grid (`source/cuda/acc_nugrid`),
and the sub-grid channel model, which exists only on the CPU. In that order:

### 1. Losses in CUDA MWDG2 and HWFV1

What is different from the uniform solvers:
- `lis::cuda::adaptive::Simulation::run(argc, argv)` parses the `.par` file
  itself; it does not receive `Fnames`, `States` and `Pars`. Pass them from
  `lisflood.cpp` (as for the other solvers) or read the loss keywords with the
  solver's own `read_keyword_*` functions.
- The solution is an `AssembledSolution` of active cells of a multiresolution
  mesh (`h0, qx0, qy0`, for MWDG2 also the slopes `h1x, h1y, ...`; `levels`,
  `act_idcs`, `length`) that changes every step. A cell at level `l` covers
  `2^(L-l) x 2^(L-l)` cells of the finest grid, which is the DEM padded to
  `2^L x 2^L`.
- Its outputs and mass log (`write_cumulative_data`) are its own.

Plan:
1. Keep the parameters (rate, `Ks`, `psi`, `dtheta`, soil-depth cap) and the
   state (`residual`, and `F` in double) on the **finest grid**, so they do
   not depend on the mesh. Read them with `cuda::Losses`' parser
   (`host_grid`) into the padded `2^L x 2^L` layout (NoData and padding: 0).
2. After `fv1_update` / `dg2_update` and before the time step is computed for
   the next step (`cuda_adaptive_simulate.cu`, after the solver branch), launch
   one kernel over the active cells: for an active cell, take the mean
   parameters of its finest children, apply `depth_after_loss` /
   `depth_after_greenampt` (copy them from `cuda_losses.cu`) to `h0`, scale
   `qx0, qy0` (FV1 rule) or, for MWDG2, apply the slope rule of
   `apply_losses_slopes` with face value `sqrt(3)`; then add the same `dF` and
   residual to every finest child, so that `F` survives when the mesh changes.
3. Volume removed = depth change x cell area (`dx_finest^2 x 4^(L-l)`); sum it
   (block sum + `atomicAdd` on a double, as in `cuda_losses.cu`) and report it
   in the solver's mass log, so that its volume error does not count it.
4. Tests: run the T012 variants and the Green-Ampt tests of `tests_8.4` with
   `mwdg2` and `hwfv1` (add the solvers to `make_inputs.py`; with `epsilon` 0
   the mesh is adapted only at the first step and should stay at the finest
   level, so the expected depths of `check.py` should apply unchanged; check
   this first), then with `epsilon` > 0 on a case whose mesh
   adapts (check the volume balance and that `F` is continuous when cells
   merge and split). Without the keywords the outputs must be identical to
   8.3, byte for byte.

### 2. Losses in CUDA ACC on a non-uniform grid (`acc_nugrid`)

Same idea; its cells also have several sizes (`NonUniformNeighbours`,
`AssembledSolution` in `cuda/acc_nugrid`). ACC changes only the depth.

### 3. MWDG2 and HWFV1 on Metal

A large port (the adaptive solver has 56 CUDA kernels, and 11 of its files
use CUB for sorting, scans and compaction of the Morton-ordered mesh). Suggested
order: HWFV1 first (finite volume, no slopes), then MWDG2.
- Replace CUB: Metal has no equivalent library. Write a radix sort and an
  exclusive scan in Metal (or use a well-tested open-source one); compaction
  is a scan plus a scatter. Check each against the CUDA result on random input
  before porting anything else.
- Port the multiresolution analysis (`mra/`: encode, threshold, decode) and
  the neighbour search (`neighbours/`) kernel by kernel, and compare the mesh
  (`levels`, `act_idcs`) with CUDA after every step on a small case: it must be
  identical, or nothing after it can be.
- Then the update operators (`operators/fv1_update`, `dg2_update`, friction,
  limiter) and the time step, as for FV2 and DG2 here: same boundary and step
  order as CUDA, each face flux computed once, no fast math, deterministic
  reductions (no float atomics in anything that feeds the solution).
- Losses: reuse `source/metal/losses` with the finest-grid state of step 1.
- Verify as in this file: identical to CUDA on the small cases where possible,
  the rest with the same input files and with the CPU or CUDA spread as
  baseline; every run must repeat bit for bit.

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
3. Done: FV2, DG2 and ACC on Metal, and infiltration, evaporation and
   Green-Ampt in the four Metal solvers. Possibly later: gridded evaporation.

Known limits, as in the CPU solvers: water shallower than `DepthThresh` does
not drain; infiltration is a constant rate (step 2 addresses this).
