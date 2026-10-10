# LISFLOOD-FP 8.4

8.4 adds **GPU infiltration and evaporation** and **expanded Apple Metal
support** to 8.3. Infiltration can use a uniform rate, a per-cell rate grid,
or Green-Ampt; evaporation uses a time series and can accompany either
infiltration model. The table below separates loss support from solver and
mesh support.

8.3 improved CUDA execution and introduced Metal FV1. 8.2 had no Metal
backend. Existing CPU loss support is unchanged in 8.4; CPU FV1/FV2/DG2
still do not apply these losses. Thus the new coverage is every **GPU solver
selection**, not every CPU solver in LISFLOOD-FP.

| GPU solver | CUDA flow solver in 8.2/8.3 | CUDA losses in 8.2/8.3 | CUDA losses in 8.4 | Metal flow solver in 8.3 | Metal in 8.4, including losses |
|---|---|---|---|---|---|
| FV1 | Available | None | All three loss options | Available, without losses | Native uniform grid; losses added |
| FV2 | Available | None | All three loss options | Unavailable | New native uniform-grid solver and losses |
| DG2 | Available | None | All three loss options | Unavailable | New native uniform-grid solver and losses |
| ACC | Available | None | All three loss options | Unavailable | New native uniform-grid solver and losses |
| ACC_NUGRID | Static non-uniform mesh | None | All three loss options | Unavailable | Dense ACC compatibility mode and losses; no mixed-size mesh |
| HWFV1 | Adaptive mesh | None | All three loss options | Unavailable | Dense FV1 compatibility mode and losses; no adaptation |
| MWDG2 | Adaptive mesh | None | All three loss options | Unavailable | Dense DG2 compatibility mode and losses; no adaptation |

**All three loss options** means constant-rate infiltration (uniform or
gridded), Green-Ampt infiltration, and evaporation. Uniform-rate and
Green-Ampt infiltration are alternatives; evaporation can be combined with
either. A supported keyword does not imply identical meshes or results across
backends. Native Metal adaptation for the last three rows remains unfinished.
The 28-run dense Metal Alzette matrix completed; the CUDA MWDG2 basin run
stalled, so the complete cross-backend matrix is not yet validated. See
`README_8.4.md` for the evidence and handover.

The version number 8.4 is a working name until agreed with the LISFLOOD-FP
developers.

## Contents

```
README.md                    this file
README_8.4.md                how 8.4 was built and tested, with the results
LICENSE                      GNU General Public License v3.0
source/                      the complete source, ready to compile
lisflood-8.2-to-8.3.patch    the 8.3 changes as one patch to the official 8.2
lisflood-8.3-to-8.4.patch    the 8.4 changes as one patch to 8.3
config.float.cmake           build settings: single precision, as the GPU solvers use
tests/                       LISFLOOD-FP's own test cases: quick check, full suite, comparison of two builds
tests_8.4/                   the infiltration and evaporation tests of 8.4
paper/                       the cases of the 8.3 paper and the script that runs them
```

`source/` is the official LISFLOOD-FP 8.2
([Zenodo 10.5281/zenodo.13121102](https://doi.org/10.5281/zenodo.13121102))
without the files the CMake build does not use (Windows and Intel library
bundles, the old makefiles, prebuilt binaries, manuals, test data and its git
history), plus the 8.3 and 8.4 changes. To apply them to a full 8.2 instead,
from its top folder: `patch -p1 -E --binary < lisflood-8.2-to-8.3.patch`, then
`patch -p1 -E --binary < lisflood-8.3-to-8.4.patch`.

## What is new in 8.4

### 1. Infiltration and evaporation on CUDA

In 8.2 and 8.3 only the classic CPU solvers (and the sub-grid solver) remove
water by infiltration and evaporation; the GPU solvers read the keywords and
ignore them. In 8.4 all seven GPU solver selections apply them on CUDA and
Metal: FV1, FV2, DG2, ACC, ACC_NUGRID, HWFV1 and MWDG2, with the Metal mesh
limitations shown above. On CUDA the existing uniform, non-uniform and
adaptive flow implementations are retained.

Add the following to a case's `.par` file for CUDA FV1 with 2 mm/h uniform
infiltration and evaporation; keep the case's terrain, boundaries, forcing
and output settings:

```text
cuda
fv1
infiltration 0.00000055555556
evaporation evap.dat
```

An `evap.dat` specifying a constant 3 mm/day is:

```text
Evaporation 3 mm/day
2 days
3 0
3 10
```

For Green-Ampt, replace the `infiltration <rate>` line with:

```text
infiltration greenampt
greenampt_ks 10
greenampt_psi 110
greenampt_porosity 0.43
greenampt_theta_init 0.30
```

These values illustrate file syntax; they are not calibrated basin parameters.
Choose one solver keyword: `fv1`, `fv2`, `dg2`, `acceleration`, `acc_nugrid`,
`hwfv1`, or `mwdg2`. Adaptive/non-uniform input requirements are described below.

**Constant rate**, as in the CPU solvers: every wet cell (deeper than
`DepthThresh`, not NoData) loses `(infiltration rate + evaporation rate) x dt`
of depth, never below zero.

| `.par` keyword | meaning |
|---|---|
| `infiltration <rate>` | uniform infiltration rate, m/s |
| `infilfile <grid>` | infiltration rate per cell, ESRI ASCII grid in mm/h, same size as the DEM (like `manningfile`); NoData and negative values count as 0; used instead of `infiltration` |
| `evaporation <file>` | evaporation time series in mm/day, uniform in space (the CPU's file and loader) |

**Green-Ampt**, a separate choice (`infiltration greenampt` instead of a rate;
both in one file stop with an error). Each parameter is a number for the whole
domain or a grid file like the Manning file:

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
infiltrated so far), stepped with implicit Euler from `F = 0`; never more than
the water present, and not beyond `soil depth x dtheta`. Evaporation works
with either model. The soil does not recover between rain events.
Rivers, lakes and sealed surfaces should get `Ks = 0`, and soils already
flooded at the start an initial moisture close to saturation; otherwise they
infiltrate as if dry.

Details:

- FV1 scales its discharges with the depth (velocity kept); ACC changes only
  the depth; FV2 and DG2 lower the average depth, keep the depth slopes and
  scale the discharge with the depth, and make a cell first order where a face
  would become dry.
- For uniform-grid solvers the volume removed is summed on the GPU: in the `.mass` file `Verror` no
  longer counts it as an error, and the last column (`Rain-(Inf+Evap)`) shows
  minus the volume removed since the start.
- Each cell carries the part of its loss that single-precision rounding
  dropped to its next step, so small rates are not lost.
- CUDA MWDG2/HWFV1 keep soil state at the finest raster resolution when the
  mesh changes; `acc_nugrid` uses the same approach on its static mixed-size
  mesh. Soil-parameter changes and NoData boundaries retain refinement. For
  each active cell, the losses of its finest children are averaged. Adaptive
  discharges and MWDG2 slopes follow the FV1/DG2 rules above; `acc_nugrid`
  changes depth only.
  Loss-enabled runs write `<resroot>.loss` at `massint`, with CSV columns
  `time,integrated_time,loss_volume,water_volume` (seconds and m3).
  `integrated_time` is the double-precision sum of the actual time steps.
  This diagnostic does not account for boundary fluxes or rainfall; the
  existing `.cumu` timing file is unchanged. Use raster inputs and a padded
  mesh of at least 32 x 32 cells: `max_ref_lvl >= 5` for MWDG2/HWFV1 and
  `L >= 5` for `acc_nugrid`.
- CPU FV1/FV2/DG2 do not apply them and print a warning.

### 2. Additional Metal solvers and their losses

8.4 adds native Metal FV2, DG2 and ACC to the Metal FV1 introduced in 8.3.
All four apply the same infiltration and evaporation options described above.
On an Apple Silicon build, replace `cuda` with `metal` in the example; retain
the loss keywords and the solver's required input files. The three remaining
GPU solver keywords are accepted through the dense compatibility modes below.

With `cuda` (or `metal`) in the `.par` file, FV1, FV2, DG2, ACC, ACC_NUGRID,
HWFV1 and MWDG2 run on the Apple GPU. Metal FV2 and DG2 follow the CUDA solvers and
Metal ACC follows the CPU ACC solver. HWFV1 and MWDG2 use their corresponding
Metal FV1 and DG2 update operators on the complete finest-level DEM grid. This
overcomes Metal's lack of the CUB sort/scan primitives used to build the CUDA
adaptive tree, but it does not compress the mesh: `epsilon`, `grading` and
`refine_wall` do not change the Metal mesh. A notice is printed at startup so
this limitation cannot be mistaken for a native adaptive run.
ACC_NUGRID likewise uses the dense Metal ACC operator at the finest level;
`L` and `epsilon` are accepted but do not make a mixed-size Metal mesh.
Outputs follow the uniform Metal path; the adaptive-only `.cumu` and `.loss`
files are not produced because the mesh does not change.

MWDG2 accepts ordinary adaptive DEM and start rasters. If DG2 modal `1x`/`1y`
files are absent, it uses the conservative piecewise-constant projection
(zero initial unresolved slopes); the RKDG2 kernel then evolves all modes.
Infiltration, evaporation and Green-Ampt use the same Metal loss kernels as
FV1/DG2. See `README_8.4.md` for implementation details and validation scope.

`lisflood -version` and normal startup print one minimal version line:
`LISFLOOD-FP 8.4`.

## Earlier changes

### 8.3 (changes from 8.2)

| | in 8.2 | in 8.3 | solvers |
|---|---|---|---|
| 1 | the maximum-depth grid is kept in CPU memory, read and written over PCIe every time step, and allocated without the ghost cells the GPU writes: large grids stop with `cudaErrorIllegalAddress` | kept on the GPU at full size, copied to the CPU only when the `.max` file is written | FV1, FV2, DG2 |
| 2 | every NoData cell adds to one shared counter every time step, even when dry | dry NoData cells skip the addition | FV1, FV2, DG2, ACC |
| 3 | the flux kernels do not synchronise their threads at the end of each pass, so on grids wider or taller than 1920 cells repeated runs give different results | they do | FV1, FV2, DG2 |
| 4 | no GPU support on Apple computers | Metal back end | FV1 |
| | does not compile with CUDA 12 on Linux (an unused global `sync()` clashes with the system's) | renamed | build |
| | a stray one-line file `format` in the top folder hides the C++ header `<format>` on macOS | removed | build |
| | `generateDG2DEM` and `generateDG2start` use an uninitialised NoData value | initialised as in `lisflood` | DG2 input tools |
| | prints its internal version number, 8.1.0 | prints `LISFLOOD-FP 8.3` | start-up |

What was checked for 8.3 (details in the 8.3 paper):

- On the test cases that come with LISFLOOD-FP, wherever 8.2 runs, 8.3 writes
  the same output files, byte for byte, for FV1, FV2, DG2 and ACC (except the
  last digit of the mass-balance log, which varies between any two runs).
- 8.2 stops with `cudaErrorIllegalAddress` under FV1, FV2 and DG2 on the
  Carlisle and EA test 5 cases (0.58 and 1.7 million cells) and on a 5.2
  million-cell grid; 8.3 runs them. Largest grids tested: 5.2 million cells
  (FV1, FV2, DG2) and 10.6 million (FV1).
- NVIDIA Compute Sanitizer finds invalid memory accesses and shared-memory
  hazards in 8.2 and none in 8.3.
- On a 24 h, 10.6 million-cell flood simulation on an NVIDIA A100, 8.3 needs
  9.4 min where 8.2 (with the maximum-depth grid allocated at full size, the
  minimum for it to run) needed 105 min. The saving depends on the grid: on a
  grid without NoData cells it is 72 % at 0.5 million cells and 85-89 % at 2 to
  8 million, and a few seconds on small grids.
- The Metal FV1 solver takes the same time steps as the CPU solver on the three
  cases of the paper and agrees with it to within single-precision rounding.
- The other GPU solvers of 8.2 (`acc_nugrid` and the adaptive MWDG2/HWFV1) are
  unchanged apart from the build fix, and were not tested.

Tested on: Red Hat Enterprise Linux 8.10, GCC 8.5, CUDA 12.6, NVIDIA A100 (40 GB);
macOS 27, Xcode 27, Apple M1 Max (32 GPU cores, 64 GB).

## Build on Linux with an NVIDIA GPU

Needs a C++14 compiler (tested with GCC 8.5), CUDA 11.4 or later (tested with 12.6),
CMake 3.13 or later, NetCDF-C, OpenMP and libnuma. On a cluster, load them as
modules, for example at ECMWF:

```bash
module load cuda/12.6 netcdf4/4.9.3 cmake/3.28.3
```

Then, from this folder, with `CMAKE_CUDA_ARCHITECTURES` set to your GPU (70
V100, 80 A100, 89 L40S, 90 H100):

```bash
cmake -S source -B build -DCMAKE_BUILD_TYPE=Release -D_CONFIG=$PWD/config.float.cmake -DCMAKE_CUDA_ARCHITECTURES=80
```

```bash
cmake --build build -j 16
```

If CMake does not find NetCDF, add
`-DNetCDF_INCLUDE_DIR=<netcdf>/include -DNetCDF_LIBRARY=<netcdf>/lib/libnetcdf.so`.

On a cluster:

- Compile on the login node, but run the check below on a GPU node, e.g.
  `srun --gres=gpu:1 --time=00:20:00 tests/quick_test.sh build/lisflood` (add
  your partition or account if the cluster needs them).
- Run `nvidia-smi` on the GPU node to see the GPU model, and make sure
  `CMAKE_CUDA_ARCHITECTURES` matches it; with the wrong number the build
  succeeds but the runs fail.
- If `lisflood` fails to start with a library error, a `LD_LIBRARY_PATH` set
  in `~/.bashrc` (e.g. for another compiler) may override the loaded modules:
  run `unset LD_LIBRARY_PATH`, then load the modules again.

## Build on a Mac with Apple Silicon

Needs Xcode with its command-line tools and, from Homebrew,
`brew install cmake libomp netcdf`:

```bash
cmake -S source -B build -DCMAKE_BUILD_TYPE=Release -D_CONFIG=$PWD/config.float.cmake -DOpenMP_CXX_FLAGS="-Xclang -fopenmp -I/opt/homebrew/opt/libomp/include" -DOpenMP_C_FLAGS="-Xclang -fopenmp -I/opt/homebrew/opt/libomp/include" -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_C_LIB_NAMES=omp -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib
```

```bash
cmake --build build -j 8 --target lisflood generateDG2DEM generateDG2start
```

The build writes five Metal libraries next to `build/lisflood`
(`metal_fv1_kernels`, `metal_fv2_kernels`, `metal_dg2_kernels`,
`metal_acc_kernels` and `metal_losses_kernels`, each `.metallib`); keep them
together with the executable.

## Check

```bash
build/lisflood -version
```

prints `LISFLOOD-FP 8.4` among its start-up lines. Then:

```bash
tests/quick_test.sh build/lisflood
```

on an NVIDIA GPU, or `GPU=metal tests/quick_test.sh build/lisflood` on a Mac,
ends with `QUICK TEST PASSED`. See `tests/README.md` for the full suite,
`tests_8.4/run_tests.sh` for the infiltration and evaporation tests and
`paper/README.md` to run the cases of the 8.3 paper.

## Use

As 8.2. Put `cuda` in the `.par` file, or pass `-cuda`, to run on the GPU; on a
Mac `metal` / `-metal` means the same, and FV1, FV2, DG2, ACC, ACC_NUGRID,
HWFV1 and MWDG2 run on the Apple GPU. The three non-uniform/adaptive solver
keywords currently use the dense finest-level mode described above. Add the
keywords above for infiltration and evaporation.

## Licence

GNU General Public License v3.0 (`LICENSE`), the licence cited in the
LISFLOOD-FP 8.0 and 8.2 papers and agreed with the LISFLOOD-FP developers.
