# LISFLOOD-FP 8.4: how it was built and tested

## Native Metal for every GPU solver (11 October 2026)

All seven GPU solver keywords now run natively on Apple Silicon:
FV1, FV2, DG2 and ACC on the uniform grid, ACC_NUGRID on its static
non-uniform mesh, and HWFV1 and MWDG2 on an adaptive mesh. Each applies
infiltration, evaporation and Green-Ampt, and accepts boundary files, point
sources and stage files as its CUDA counterpart does.

| keyword | Metal implementation | ported from |
|---|---|---|
| `hwfv1` | `source/metal/adaptive`: Haar wavelets on the cell means (`hwfv1_encode_level`), first-order update (`hwfv1_update`) | `cuda/adaptive` with `solver_type HWFV1` (`fv1_update.cu`, `encode_and_thresh_flow.cu`) |
| `mwdg2` | `source/metal/adaptive`: multiwavelets, DG2 residual (`mwdg2_rhs`), SSP-RK2, limiter | `cuda/adaptive` (`dg2_update.cu`, `limit_slopes.cu`) |
| `acc_nugrid` | `source/metal/acc_nugrid`: mesh, neighbour records and interfaces built once on the host; discharge, depth, time step, rain and losses on the GPU | `cuda/acc_nugrid` (`compute_q.cu`, `update_h.cu`, `find_nonuniform_neighbours.cu`, `find_interfaces.cu`) |

Both adaptive solvers share one driver (`metal_mwdg2_simulate.mm`); the mesh
is selected on the GPU every step (`metal_mwdg2_mesh.mm`).

### How Metal follows CUDA

Read from the CUDA sources and confirmed case by case against the
single-precision A100 build of 8.4:

- **Neighbours.** A leaf reads its own coefficients after friction and slope
  limiting, but its neighbours from the tree loaded before them
  (`load_soln_and_nghbr_coeffs` before `limit_slopes`). Friction is applied to
  the neighbour copies only when the reading leaf itself is above `tol_h` and
  `tol_s` (`friction_implicit.cu`); such a leaf only has its mean discharge
  zeroed. `tol_s = 1e-9`.
- **SSP-RK2** averages with the limited stage-1 state (the
  `copy_to_buf_assem_sol` copy is overwritten by stage 1).
- **Wet leaves.** Only leaves whose own or a neighbour's mean depth reaches
  `tol_h` are updated. FREE ghosts are built from the cell means.
- **Time step.** The first step is 0.001 s (`initial_tstep` is not read); a
  dry leaf caps the next step at `initial_tstep`; no step is shortened to hit
  an output time (outputs follow the first step at or after each time).
- **Domain edge.** The raster's last row and column are the edge
  (`find_neighbours.cu`: north/east from the leaf's north-east finest cell,
  south/west from its south-west one); DG2 does not update leaves centred in
  the padding. With no boundary file the edge ghost copies the depth and
  zeroes the normal discharge, so water leaves wherever the edge cell has
  outward flow; the `.mass` balance counts this flow.
- **Boundaries and sources** (`classes/Boundary.h`, `PointSources.h`): one
  segment per side, cells `int((lower-origin)/dx)` to
  `int((upper-origin)/dx)-1`; HFIX/HVAR set the edge cell's depth to
  `max(0, level-z)`, QFIX/QVAR to `q dt/dx`; point sources set or raise the
  finest cell's level before each remeshing; boundary, source and stage cells
  stay finest. `refine_wall`/`ref_thickness` keep the north and east strips
  finest.
- **ACC_NUGRID** builds its mesh from the DEM and its DG2 slope files with the
  multiwavelet transform, its neighbour records (one per neighbouring face;
  the face length is the neighbour's cell size) and interfaces exactly as the
  CUDA kernels. NoData and padding are `nodata_elevation` walls (default
  1e7 m), as `read_raster_file` sets them. Its uniform-rain row stepping,
  dynamic-rain tile mapping and stage/velocity/VTK outputs follow CUDA; its
  `.mass` holds real volumes (CUDA's holds zeros).

Where Metal deliberately differs: the remeshing keeps the input bed on the
finest level (it never smooths it); the padding repeats the raster's edge
cells (see the defects below); rain is integrated over the step actually
taken; in the MWDG2 input a NoData corner of a basin cell takes the cell's
own values and NoData cells stay walls (CUDA averages the wall height in;
`LISFLOOD_ADAPTIVE_CUDA_NODATA_INPUT=1` restores it); losses are booked as
the water that actually left the stored level (the state holds z + h in
float32, so on a high bed a small step can round away and is carried to the
next one).

### Results against CUDA

Same cases on both back ends (`tests_8.4/adaptive_parity.py`; CUDA:
`build_head84`, single precision, A100). Depth difference: normalized L2 over
the raster at each output.

| case | solver | Metal vs CUDA |
|---|---|---|
| dam break, friction, limiter, 0.5 s (29 steps) | MWDG2 | 0.004 % at worst, same steps |
| same, 0.1 s of 1 ms steps | MWDG2 | below 0.0001 % |
| dam break, 20 s | HWFV1 | 0.02 %, same 405 steps |
| wet/dry hill, 30 s | HWFV1 | 0.06 %, same 601 steps |
| HFIX inflow + FREE outflow / QVAR inflow / point sources / QFIX + infiltration, 30 s | HWFV1 | 0.10 / 0.11 / 0.06 / 0.015 % at the end |
| same four, 30 s | MWDG2 | 0.2-2.3 % at the end |
| lake at rest on a hill | MWDG2 | depth identical; discharge round-off (3e-6 m2/s) on both |
| dam break, compressed mesh, 20 s | ACC_NUGRID | identical 28-leaf mesh, 0.13 % |
| CUDA's own acc_nugrid loss suite (`tests_8.4/acc_nugrid.py`) | ACC_NUGRID | 106 of 106 checks |

On wet/dry hills MWDG2 and ACC_NUGRID agree with CUDA to round-off for the
first 30-40 steps and then drift. Adding 1e-5 m to one start cell moves
Metal against itself by as much (MWDG2 frictionless bowl: 8-14 %; ACC_NUGRID
hill: 18-33 %), so these flows are sensitive on both back ends; the 2,000 s
frictionless sloshing dam differs between CUDA in single and double precision
as much as between Metal and CUDA.

`python3 tests_8.4/adaptive_parity.py check WORK BIN` reruns eight of these
cases and compares them with the saved CUDA outputs in
`tests_8.4/fixtures/adaptive_parity` (no GPU cluster needed).

### Found in CUDA while matching it (to report, not copied unless stated)

- MWDG2/HWFV1 with no boundary file: the edge is not a wall (copied).
- MWDG2 with `limitslopes`: the two sides of a face use different traces,
  so mass is not conserved exactly (copied; 0.013 % in the limiter test).
- MWDG2/HWFV1: `initial_tstep` is not read (first step 0.001 s); no rainfall
  input; with `epsilon 0` the written depth raster is the domain mean.
- MWDG2/HWFV1 on rectangular rasters: padding is `wall_height` without water
  (`read_raster_file`), so coarse leaves next to the raster edge read
  ancestors that mix wall and water and a lake at rest starts to flow; the
  input projection also duplicates node row `2^L - ysz` (row 76 for the
  Alzette basin). Metal pads with the edge cells and does not copy the row.
- HWFV1 (`fv1_update.cu`) updates padding cells, which take water from the
  raster edge that the edge cell never loses (copied for parity).
- When a dry coarse leaf is refined, its children get the parent's water
  level over their own beds, giving small positive and negative depths
  (seen as +/-0.02 m on the bowl test). Metal copies it for parity;
  `LISFLOOD_ADAPTIVE_POSITIVE_EXPAND=1` instead gives each child the
  parent's mean depth over its own bed (no negative depths, same volume) but
  changes the MWDG2 steps, so it is a diagnostic, not the default.
- MWDG2/HWFV1 scale `epsilon` by the largest bed and water level in the
  padded raster, which includes the `wall_height` given to NoData and padding
  cells (`preflag_topo`, `get_max_scale_coeffs`). With `wall_height 10000`,
  `epsilon 0.001` allows 10 m of bed or level variation inside a leaf. On
  the Alzette 100 m basin (Metal HWFV1, closed outlet, no rain, 1 h; budget
  from `LISFLOOD_MWDG2_TRACE_MASS=1`) that coarsens across river banks:
  children of wet leaves get depths down to -12 m and the flux update loses
  39 % of the water, because face fluxes are not conservative where leaf
  sizes change (a coarse leaf reads the mean of its finer neighbours, the
  fine leaves read the coarse leaf). With `epsilon 1e-4` the error is +1.4 %
  and depths reach -1.2 m; with `1e-5` or `0` it is +0.4 % (friction applied
  differently by wet and dry readers) and no depth is below -0.003 m.
  Remeshing itself changes the volume by under 20 m3. Small cases match CUDA's
  total volume to 0.1 %; a CUDA Alzette run at `epsilon 0.001`
  (`build/alzette_hwfv1_mass.sbatch`) is still to be done. Use an `epsilon`
  of the wanted level tolerance divided by `wall_height`.
- MWDG2 input (`read_and_project_modes_dg2.cu`): each cell's modes come
  from four corner samples (its own and its east, north-east and southern
  neighbours'), with NoData read as `wall_height`. A basin cell beside
  NoData gets a bed thousands of metres high and NoData cells beside water
  get water; on the Alzette basin this gives a -60 m cell (Metal, copied)
  and is the likely cause of CUDA's collapse there. The corner sampling also
  shifts and smooths the start depths by half a cell (kept: parity).
- ACC_NUGRID: a fine cell's face record uses the coarse neighbour's size as
  the face length, so the two sides of a coarse/fine face do not balance;
  with sloping beds the scheme gains water and becomes unstable (closed hill:
  80 -> 829 m3 on CUDA). Its `.mass` file holds zeros; dynamic rain is mapped
  from the south edge; the uniform-rain row stepping scales the time twice
  after a row change; it needs a stage file.
- The Alzette outlet segment (`N 75641 75666 FREE`) is narrower than one cell
  and maps to an empty range in the CUDA adaptive and non-uniform readers.

`README.md` describes 8.4 and how to build and use it; this file records how
8.4 was developed and tested, and the results.
**The intended 8.4 contribution over the 8.3 release baseline** is to make
floodplain water losses available in the GPU solvers (uniform infiltration,
cell-varying infiltration, evaporation and Green-Ampt), and to provide native
Metal implementations of the GPU solver family. This describes the release
baseline and this code line; it is not a claim that no LISFLOOD-FP research
extension has ever implemented Green-Ampt.

For example, the 2023 LISFLOOD-FP-vGA thesis describes a separate research
extension with spatial Green-Ampt and a Parlange explicit approximation. Its
existence means Green-Ampt is not new to every LISFLOOD-derived code. It is not
the standard 8.2/8.3 release implementation, and it does not establish that
its equation, parameter handling or published test outputs are ground truth
for this 8.4 implementation. Our CUDA/Metal path instead solves the ponded
Green-Ampt step with an implicit per-step update. Cross-model comparisons must
align the equation, wetting assumptions, parameters, time step and water
balance before interpreting a difference as a defect.

## Why

In the 8.2/8.3 release sources checked for this branch, the classic CPU solvers
(and the sub-grid solver) remove water by the existing uniform-rate
infiltration and evaporation path (`infevap.cpp`). The standard GPU solvers,
and CPU FV1/FV2/DG2, do not provide the 8.4 loss behavior. Separate research
extensions can differ from this release baseline. For a flood such as the
July 2021 Alzette event, water that soaks into the ground cannot be represented
on the GPU at all in that baseline.

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

### Experimental multi-GPU CUDA FV1 (14 October 2026)

#### Clean ECMWF version layout

The minimal comparison layout is installed in purge-safe ECMWF permanent
storage at `/perm/lux0804/LISFLOOD_versions`:

| Directory | Source version | Output directory |
|---|---|---|
| `LISFLOOD-8.3` | LISFLOOD-FP 8.3 CUDA build | `LISFLOOD-8.3/output` |
| `LISFLOOD-8.4-single` | commit `5d50b13`, the final 8.4 state before multi-GPU FV1 | `LISFLOOD-8.4-single/output` |
| `LISFLOOD-8.4-multigpu` | commit `24b159c`, final validated MPI FV1 optimization with the `GPUs` parameter | `LISFLOOD-8.4-multigpu/output` |

All three use `/perm/lux0804/LISFLOOD_versions/input`; there are no
private input copies inside the installations. The shared files are the 10 m
DEM and Manning raster, `Border.bci`, the 96-hour WRF rainfall, and
`13_july_18hr.wd`. No file called `18_july_18hr.wd` was present in scratch or
permanent storage when the layout was created. `13_july_18hr.wd` is consistent
with the rainfall window, which begins at 18:00 UTC on 13 July 2021; it replaces
the earlier, opaque `6hr-0000.wd` path in the supplied parameter files.

Each version directory has `source`, `build/lisflood`, `run.par`, and its own
`output`. Reusable copies of the parameter files and layout instructions are in
`tests_8.4/hpc_layout`. To return to 8.4 without multi-GPU support, check out
commit `5d50b13`. The multi-GPU work began at `9c554b8`; use validated commit
`24b159c` for the final optimized implementation and the checked `GPUs 1`/`GPUs 4`
parameter behavior. Commit `3be328a` is the earlier validated implementation before
the row-local and timestep optimizations.

The installation, shared inputs, and empty per-version output directories are
kept under `/perm`; they are not dependent on the scratch copy. A 96-hour 10 m
run can produce about 27 GB per configuration, so for disposable benchmarks it
is reasonable to copy `run.par` and point only `dirroot` back to scratch while
leaving the executable and inputs in permanent storage.

CUDA FV1 can run with MPI ranks on separate NVIDIA GPUs when configured with
`-DLISFLOOD_ENABLE_CUDA_MPI=ON`. The option is off by default. Each rank owns
a contiguous band of raster rows, updates that band on its assigned CUDA GPU,
and exchanges one row of flow state with its neighbours. State rasters are
replicated on each rank; output snapshots are assembled across ranks. This is
an experimental single-node path for **FV1 only**. It does not distribute
raster memory, and host-staged halo exchange and collectives add overhead.

Set the requested rank/GPU count in the parameter file after `cuda`:

```text
fv1
cuda
GPUs 4
```

`GPUs` is checked against the MPI process count, so it also documents and
guards the intended launch. The parameter does not start MPI processes or
request hardware from the scheduler: the batch allocation and launch command
must use the same count. For example, use `#SBATCH --ntasks=4`, request four
GPUs, and launch with `mpirun -np 4 lisflood case.par`. For one GPU, set
`GPUs 1` and launch directly or with `mpirun -np 1`. If the counts do not
match, FV1 stops with an error instead of silently using a different number.
Multi-GPU/multi-rank operation applies to CUDA FV1 and Apple Silicon Metal FV1;
the other CUDA and Metal solvers remain single-GPU.

For Metal FV1 on Apple Silicon, configure with `-DLISFLOOD_ENABLE_METAL_MPI=ON`
and OpenMP support:

```sh
cmake -S source -B build/macos_mpi -DLISFLOOD_ENABLE_METAL_MPI=ON -DOpenMP_ROOT=/opt/homebrew/opt/libomp
cmake --build build/macos_mpi --target lisflood -j
```

Use `tests_8.4/cuda_fv1_mpi_alzette.sbatch` as the ECMWF example for CUDA. The allocation
requests four A100s, and the case file must contain both `cuda` and `fv1`;
`fv1` alone runs the CPU implementation. The example uses `mpirun` with four
ranks, and checks that MPI reports ranks 0–3 before running. Build with CUDA,
MPI C++ and the normal LISFLOOD dependencies loaded:

```sh
cmake -S source -B build-mpi -DLISFLOOD_ENABLE_CUDA_MPI=ON
cmake --build build-mpi -j
```

On 14 October, ECMWF job 38422144 ran a six-hour, 100 m Alzette case with
hourly outputs on one A100 and four A100s. Both the loss-free and Green-Ampt
cases passed. Logs report rank 0/4 through rank 3/4 on distinct A100s and
non-overlapping row bands. The checker compares every depth, Qx, Qy, elevation
and maximum-depth raster, plus mass rows. All saved rasters matched exactly
(maximum absolute difference 0); maximum scaled mass difference was
`5.424e-7` without losses and `0` with Green-Ampt. The small case took about
2 s on one GPU and 3 s on four GPUs, so this validates parallel execution and
numerical parity, not a speedup. Run the checker with
`python3 tests_8.4/compare_cuda_fv1_mpi.py SINGLE_DIR MULTI_DIR`.

The implementation also reduces the global CFL step, mass and loss totals,
keeps the row ownership consistent for rain, boundaries, NoData draining and
maximum-depth output, and synchronizes snapshot scheduling across ranks. The
snapshot schedule fix was found from a live MPI stack trace: non-root ranks
were entering a gather one output early while rank 0 had advanced to halo
exchange. `MPI_Finalize` is called when FV1 initializes MPI itself. The Alzette
regression covers dynamic rainfall and Green-Ampt; other boundary/source and
loss combinations still need dedicated multi-GPU tests before broader claims.

A 96-hour extension was run on 15 October (ECMWF job 38522270). For the
no-loss case, every hourly raster still matched exactly, but mass diagnostics
diverged after the early outputs (including wet area and discharge totals),
and the strict mass checker failed. The four-GPU run took 67.02 s wall / 65 s
solver loop versus 34.63 s / 32 s on one GPU, so it was slower on this basin.
That was the 100 m case. It stopped before Green-Ampt and does not establish
long-duration parity for that setup; see the dated entry in
`HANDOVER_LOCAL.md`.

These timings cannot be compared with the 8.3 large-grid benchmark: this test
used 100 m cells (about 0.19 million raster cells) and 51,369 solver steps,
whereas the benchmark used 10 m cells (10.64 million cells) and a much smaller
time step. The older README also mislabeled its benchmark: archived parameter
files show the 9.4-minute CUDA FV1 run simulated 2 h, not 24 h; the 24 h,
63.2-minute archived run was CUDA ACC with an initial 0.5 s time step. The
benchmark statement in `README.md` has been corrected accordingly.

To measure the matching large grid directly, ECMWF job 38525312 ran the
10 m, 10,642,080-cell, WRF-driven Alzette FV1 case for 24 h on one A100 using
the current 8.4 CUDA binary. It completed 182,412 steps and wrote all 25
hourly depth snapshots through 86,400.383 s. The solver loop took 802 s
(13 min 22 s); whole-process wall time was 821.53 s (13 min 42 s). Startup
confirmed `CUDA FV1` bound to `NVIDIA A100-SXM4-40GB`. The historical 8.3 FV1
record was only a 2 h run (496 s); the new 24 h time is the valid current
24-hour FV1 measurement.

The same 10 m, 24 h case was rerun on 15 October with one and four A100s in
ECMWF job 38530439. One GPU took 720 s solver / 734.49 s wall; four GPUs took
566 s / 586.62 s, a 20.1% lower wall time (1.25x throughput). Each run reached
86,400.383 s and wrote 25 hourly depth snapshots. Every hourly `.wd`, `.Qx`,
`.Qy`, `.elev` and `.max` raster matched exactly. Across 145 `.mass` rows, the
maximum relative Qout difference was 7.71e-5 (0.0077%); the maximum residual
difference scaled by stored water volume was 5.15e-7. This passes the 3%
acceptance threshold, while the measured speedup applies to this case and
hardware only. Reproduce with
`tests_8.4/cuda_fv1_mpi_alzette_10m_24h.sbatch` and
`tests_8.4/compare_cuda_fv1_mpi.py`.

The same 10 m case was extended to the full 96-hour WRF window on 15 October
(ECMWF job 38547028). The 96-hour rainfall file was present on HPC at
`/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_all_models_96h/inputs_18utc/rain_wrf1p3_afterda_96h.nc`;
it covers 13 July 2021 18:00 UTC through 17 July 18:00 UTC, matching the
initial depth file. Both runs reached 345600.406 s and wrote 97 hourly depth
snapshots. One A100 took 2641.13 s wall (2626 s solver); four A100s took
2162.15 s wall (2143 s solver), an 18.1% lower wall time (1.22x throughput).
All 389 paired `.wd`, `.Qx`, `.Qy`, `.elev` and `.max` raster outputs were
exactly equal at saved precision. Across 577 mass rows, maximum relative
differences were 0.0144% for Area, 0.00145% for Vol and 0.00743% for Qout;
Qerror and Verror differences scaled by stored water volume were `3.89e-9`
and `6.32e-6`. These are below the stated 3% acceptance threshold.

The original Slurm job used the generic checker's `1e-6` mass tolerance and
therefore exited nonzero on the 500 m2 Area diagnostic difference (0.0144%),
even though all saved rasters were exact. The mass-only rerun with the
documented 3% acceptance tolerance passed across all 577 rows. The 96-hour
batch example now uses that separate 3% mass tolerance while retaining the
`1e-6` raster tolerance and stricter `1e-4` Qout tolerance. Reproduce with
`tests_8.4/cuda_fv1_mpi_alzette_10m_96h.sbatch`; compact logs and mass tables
are archived at `/perm/lux0804/lf84/fv1_mpi_validation_10m_96h_38547028/`.

### FV1 multi-GPU synchronization and decomposition optimization (15 October 2026)

Commit `24b159c` is the final validated implementation. Each rank still holds a
replicated raster allocation, but all timestep-critical FV1 kernels now launch
only over that rank's contiguous row stripe. This includes fluxes, friction,
rain, losses, NoData drainage, maximum depth, and the CFL field/reduction. A
true local-only allocation would reduce memory use, but it would not remove the
per-step global CFL agreement or halo dependency that now limit scaling.

Halo exchange packs H, HU and HV for both neighbours on the GPU, performs one
pinned device-to-host transfer, posts the north and south MPI operations
concurrently, then performs one host-to-device transfer and unpacks on the GPU.
Friction is evaluated on the already-received halo rows because it is a
cell-local operation; this removes the former second halo exchange without
changing the FV1 update. Mass volume and wet-area calculations reduce each
rank's stripe to scalars and combine those scalars with MPI. Raster bands are
gathered only for an hourly snapshot or enabled stage output.

The CFL minimum is reduced into normal device memory, copied to a pinned host
scalar, combined with `MPI_Allreduce`, and copied back for the CUDA kernels.
Host time accounting uses the returned scalar and no longer reads the managed
CUDA timestep twice per iteration. The ECMWF build was independently checked:
it is a `Release` build targeting A100 `sm_80`; it was not using `sm_61`. The
example batch script requests two logical CPUs per MPI rank and binds each rank
to a physical core.

The final 24-hour, 10 m Alzette test is ECMWF job `38672520`. Four A100s took
**5.0020 minutes wall** (283 s solver loop), versus the existing one-A100
reference at **10.1047 minutes wall** (590 s solver loop). This is **2.02x
throughput**, a **50.5% wall-time reduction**. All 101 hourly `.wd`, `.Qx`,
`.Qy`, `.elev` and `.max` raster files matched exactly. Across 145 mass rows,
the largest scaled difference was `6.17017338e-05` (0.0062%), and the scaled
residual difference was `5.2227503e-07`; both pass the 3% acceptance limit.
The four-GPU-only reproducer is
`tests_8.4/cuda_fv1_mpi_alzette_10m_24h_multionly.sbatch`; it compares against
the preserved one-GPU output instead of rerunning it.

For comparison, the earlier implementation took 8.9092 minutes on four GPUs.
Row-local work, scalar output reductions, one halo exchange, device timestep
staging and CPU affinity therefore reduced the four-GPU wall time by 43.9%. A
pre-timestep version of these changes took 5.4837 minutes in job `38668706`
and also passed exact raster comparison.

A direct CUDA IPC halo prototype was measured rather than assumed. It opened
peer allocations successfully on four A100s, despite UCX reporting no CUDA IPC,
GDR-copy or CUDA-copy transport. It took **8.8370 minutes** and was rejected;
the direct peer copies and synchronization were slower than the batched pinned
host path on this node. The IPC code is not in commit `24b159c`.

Four GPUs cannot deliver 4x for this explicit adaptive-timestep case while
preserving the numerical sequence. Every one of the approximately 182,000
timesteps requires a global minimum and neighbour-halo agreement, kernel launch
and synchronization costs do not divide by four, and hourly output remains a
rank-0 operation. The measured 2.02x is the supported result. Reaching 3x would
require a larger redesign, such as a persistent multi-GPU execution model that
o longer returns to MPI/CPU every timestep; stripe-only allocation by itself
does not remove those synchronization points.

### Independent Green-Ampt equation check (14 October 2026)

To avoid treating either this implementation or another LISFLOOD extension as
truth, CUDA FV1 was run on the ECMWF NVIDIA A100 against the continuous
Green-Ampt equation. The flat 10 x 10 grid had 1 m cells, 0.09 m initial
ponding, and Rawls sandy-loam parameters reported in the LISFLOOD-FP-vGA
thesis: Ks 10.9 mm/h, suction 110.1 mm, saturated moisture 0.412 and initial
moisture 0.041. With no rain, ponding falls as infiltration proceeds. At the
1, 2 and 3 hour outputs, GPU depth differs from the independently solved
continuous equation by 0.0095, 0.0076 and 0.0066 mm; the maximum over all ten
hourly outputs is 0.0096 mm. The first 0.532 s snapshot differs by 0.143 mm;
it is listed separately because it is the first finite step from F=0. Both are
dry by hour 4 (the continuous solution dries at 3.3704 h). The run log
confirms `GPU enabled` and `Using CUDA
device`; it completed in 4 s of solver loop time. Recheck saved outputs with
`python3 tests_8.4/greenampt_implicit_reference.py WORK` (the checker also
requires CUDA-device evidence in `run.log`).

This is a controlled equation test, not a watershed validation. The thesis's
separate LISFLOOD-FP-vGA test is not a clean target for direct output matching:
it uses a Parlange approximation and describes wetting-time handling, and its
reported test-1 endpoint water volume (0.018 m3) does not reconcile with its
plotted cumulative infiltration endpoint (about 0.0416 m for a 1 m2 cell and
0.09 m initial ponding). So we record the difference as an unresolved
reference-definition issue rather than changing 8.4 to force agreement. A
rainfall-runoff case with complete inputs, a consistent published water
budget, and observation data is still needed for hydrological validation.

### Independent field-curve check (14 October 2026)

As a second external check, `tests_8.4/greenampt_field_check.py` compares the
classical Green-Ampt equation with the TA1 cumulative-infiltration observations
published by Zheng et al. (2025), Table 1. The study reports Ks = 0.0136 cm/min
and moisture contents 0.03 and 0.46. To avoid fitting and scoring on the same
data, only the 10-minute measurement calibrates A = psi (theta_sat - theta_init);
the other six measurements are held out. This gives equivalent suction
psi = 23.791 cm and, on the held-out observations, MAPE 1.68%, maximum relative
error 4.03%, RMSE 0.0702 cm and R2 = 0.9913. The script passes its declared
gates of 2% MAPE and 5% maximum relative error.

This is empirical evidence for the equation's cumulative-infiltration curve,
not a full LISFLOOD-FP simulation or watershed validation: it does not test
rainfall-runoff routing, surface ponding dynamics, GPU parity, or basin water
budgets. The suction is calibrated from the first point, and the study's
hydraulic setup is not reproduced as a LISFLOOD case. It complements the A100
controlled GPU equation test above, but a published catchment case with
available forcing, terrain, and observed hydrographs is still needed to
validate the coupled flood model.

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

8.3 ran only FV1 on Metal. 8.4 adds FV2, DG2, ACC, ACC_NUGRID, HWFV1 and
MWDG2. A `.par` file with `cuda` (or `metal`)
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

### ACC_NUGRID, HWFV1 and MWDG2

See "Native Metal for every GPU solver" at the top of this file.

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
| 16 | The adaptive and non-uniform CUDA solvers read many inputs and keywords themselves, with their own rules (boundary ranges, padding, NoData walls, first step, output times) | Metal results differed in ways the LISFLOOD documentation does not explain | read each CUDA reader and kernel, and confirm every rule with a small case run on both back ends (`tests_8.4/adaptive_parity.py`) |
| 17 | Agreement with CUDA after many steps on wet/dry flows looked poor (up to 20 %) | long runs drifted while short runs matched | measure the flow's own sensitivity: the same back end with 1e-5 m added to one start cell drifts by as much; compare single and double precision CUDA too |
| 18 | A function-local 12x12 filter matrix in the Metal MWDG2 kernels was rebuilt on every call | 0.3 s per step on the Alzette tree | a constant table with the same values; GPU copies instead of CPU memcpy (5.7 times faster) |
| 19 | CUDA reference outputs made by a modified double-precision debug build | a 21 % "failure" of Metal MWDG2 | references from the unmodified single-precision release build only |
| 20 | HWFV1 on Alzette lost 39 % of its water in an hour at `epsilon 0.001` | a mesh, flux or rain fault was suspected | a per-step budget (`LISFLOOD_MWDG2_TRACE_MASS=1`) split the flux update, remeshing, losses and rain: only the flux update changes the volume, and only with coarse leaves; the mesh dump (`LISFLOOD_DIAG_DUMP_STEP`) showed the threshold is scaled by the 10,000 m wall height, so `epsilon 1e-5` is the equivalent of a 0.1 m tolerance |
| 21 | HWFV1 on Alzette booked 113,000 m3 of infiltration but kept the water | the losses looked applied | compare the booked volume with the stored water: the adaptive state holds z + h in float32 and a 2 mm/h step rounds away on a 300 m bed; the residual now carries it and only removed water is booked (test: losses on a 300 m bed) |
| 22 | MWDG2 on Alzette had a -60 m cell from the start | an unstable flow was suspected | the first output already differed from the start file: CUDA's corner-sampled input averages the NoData wall height into basin cells; NoData corners now use the cell's own values (test: lake at rest beside NoData) |

## Handover: what remains

Every GPU solver runs on CUDA and natively on Metal, with infiltration,
evaporation and Green-Ampt. What remains: report the CUDA defects listed at
the top; a faster MWDG2 Metal step for large basins (about 30 ms per step on
the 512 x 512 Alzette tree; each kernel still waits for the previous one,
`LISFLOOD_MWDG2_PROFILE=1` prints where the time goes); and the CUDA runs of
the rectangular-raster comparison cases (`rectlake_*`, `rectdam_*` in
`tests_8.4/adaptive_parity.py`), which were written but not yet run on the
A100. The sub-grid channel model remains CPU-only, on both back ends.

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

### Native adaptive and non-uniform Metal solvers: tests

- `sh tests_8.4/run_metal_mwdg2_app.sh BIN`: lake at rest on rectangular and
  mixed meshes, CFL steps, uniform and grid infiltration, Green-Ampt with its
  soil cap across remeshing, evaporation, uniform and NetCDF rain, NoData,
  `.mass` and `.loss`.
- `run_metal_mwdg2_{boundary_app,limiter_app}.sh BIN`: FREE edge, stage
  output, `drain_nodata`; slope limiter.
- `run_metal_mwdg2_cuda_{hill,parity,trajectory,1s}.sh BIN`: saved CUDA
  outputs (lake at rest; 3, 100 and 1000 steps of a dam break).
- `run_metal_mwdg2_{input,selection,mra,scan,mesh,live_step,adaptive_minicase}.sh
  DIR`: the components (input projection, selection, transforms, scan, mesh
  and neighbours, one step, an adaptive mini-case).
- `python3 tests_8.4/adaptive_parity.py check WORK BIN`: MWDG2, HWFV1 (with
  boundaries, sources and losses) and ACC_NUGRID against saved CUDA outputs.
- `python3 tests_8.4/acc_nugrid.py BIN BIN WORK`: CUDA's ACC_NUGRID loss
  suite, 106 checks (with the Metal binary as both arguments on a Mac, which
  has no 8.3 ACC_NUGRID to compare against).

All pass on the Apple M1 Max (11 October 2026).

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
3. Done: every GPU solver runs natively on Metal, with infiltration,
   evaporation and Green-Ampt, boundary files and point sources.
4. Done: infiltration, evaporation and Green-Ampt in CUDA MWDG2, HWFV1 and
   non-uniform-grid ACC.

Known limits, as in the CPU solvers: water shallower than `DepthThresh` does
not drain; infiltration is a constant rate (step 2 addresses this).
