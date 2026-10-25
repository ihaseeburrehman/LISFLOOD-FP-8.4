# LISFLOOD-FP 8.4

2D flood model with every GPU solver on NVIDIA (CUDA) and Apple Silicon (Metal),
infiltration and evaporation on every GPU solver, and multi-GPU runs over MPI.

## Install

```bash
./install_lisflood_8.4.sh --check
```

This lists what the machine has and what is missing (CMake, a C++ compiler,
CUDA or Xcode/Metal, NetCDF, NUMA on Linux, and MPI for several GPUs). For
each missing item it prints the command that installs it on this system. Then
build and test:

```bash
./install_lisflood_8.4.sh
```

It checks again and asks before building. It then runs `example/example.par`
on the GPU and prints `TEST PASSED`. Options: `--precision double`,
`--mpi off`, `--cuda-arch 90` (H100), `--prefix ~/lisflood` (copies the
programs to `~/lisflood/bin`), `--yes` (no questions). If a library is in a
non-standard place, point to its folder: `NETCDF_DIR=/opt/netcdf`,
`CUDA_HOME=/usr/local/cuda-12.6`, `NUMA_DIR`, `MPI_HOME`. On an HPC system,
`module load` the compilers and libraries first.

### Building by hand

```bash
./build.sh
```

The installer calls this script. It picks the back end itself: Metal on macOS,
CUDA on Linux when `nvcc` is found. It switches multi-GPU on when an MPI
compiler is found. The result is `build/lisflood`; keep the `.metallib` files
next to it on a Mac.

| Setting | Default | Example |
|---|---|---|
| `PRECISION` | `float` | `PRECISION=double ./build.sh` |
| `MPI` | `auto` | `MPI=off ./build.sh` |
| `CUDA_ARCH` | detected with `nvidia-smi`, else `80` | `CUDA_ARCH=90 ./build.sh` (H100) |
| `BUILD` | `build` | `BUILD=build_double ./build.sh` |

**Requirements.** On Linux: CMake ≥ 3.13, CUDA ≥ 11.4, NetCDF-C, libnuma and
OpenMP, plus MPI for multi-GPU. At ECMWF:
`module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3`. On a Mac:
Xcode and `brew install cmake libomp netcdf`, plus MacPorts `mpich` for
multi-rank (Open MPI 5 hangs at start-up on macOS).

**Check.** `tests/quick_test.sh build/lisflood` on CUDA, or
`GPU=metal tests/quick_test.sh build/lisflood` on a Mac, should print
`QUICK TEST PASSED`. `python3 tests_8.4/losses/horton_free_test.py build WORK`
checks Horton and the FREE boundary on all seven solvers in a few minutes.

## Run

Start from `example/example.par`. It runs as it is (a 3 h storm on a small
synthetic valley, a few seconds on a GPU) and explains every keyword it uses.

```bash
cd example && ../build/lisflood example.par
```

The keywords that choose the solver and the hardware:

```text
fv1            # or fv2, dg2, acceleration, acc_nugrid, hwfv1, mwdg2
cuda           # or metal: the same switch, the build decides
GPUs 4         # optional: FV1, FV2 and DG2 only
```

```bash
build/lisflood case.par
```

```bash
mpirun -np 4 build/lisflood case.par
```

```bash
mpirun -np 4 tests_8.4/multigpu/rank_gpu.sh build/lisflood case.par   # FV2, DG2
```

Use the first form for one GPU and `mpirun` for several. With `mpirun`, `-np`
must equal the `GPUs` value in the `.par` file. Each rank owns a band of raster
rows and one GPU.

Losses, one infiltration model at a time (each value a number or a grid file):

```text
infiltration 1e-6          # constant rate, m/s (or: infilfile rates_mm_per_h.asc)
infiltration greenampt     # greenampt_ks (mm/h), greenampt_psi (mm),
                           # greenampt_porosity, greenampt_theta_init [, greenampt_soil_depth (m)]
infiltration horton        # horton_f0, horton_fc (mm/h), horton_k (1/h)
evaporation evap.dat       # mm/day time series, with any of the above
```

## What is new since 8.3

| | 8.3 | 8.4 |
|---|---|---|
| CUDA solvers | FV1, FV2, DG2, ACC, ACC_NUGRID, HWFV1, MWDG2, without losses | all seven, with losses |
| Metal solvers | FV1 only, without losses | all seven native, with losses |
| Losses (CUDA and Metal) | none on GPU | uniform or gridded constant-rate infiltration, Green-Ampt infiltration, Horton infiltration, evaporation (combines with any of them) |
| FREE boundary | lets water in where flow at the edge points into the domain | outflow only: inward momentum is reflected (CPU FV1/DG2, CUDA and Metal FV1/FV2/DG2); the CPU FV1 wall keeps its tangential momentum |
| Multi-GPU | none | `GPUs N` + MPI for CUDA FV1, FV2 and DG2; multi-rank Metal FV1, FV2 and DG2 |
| Metal adaptive solvers | none | HWFV1 and MWDG2 adapt the mesh every step with CUDA's multiresolution analysis; uniform and NetCDF rain accepted |
| CUDA FV2 uniform rain (`rainfall`) | ignored | applied |
| CUDA per-step work | rain file read every step; separate kernels for time step, drain, losses and maximum depth | rain file read once; time step reduced in one kernel; drain, losses and maximum depth in one kernel; launch grids sized to the domain |
| Install | several platform-specific CMake lines | `./install_lisflood_8.4.sh` (checks, builds, tests) or `./build.sh` |
| Example | none | `example/example.par`: a runnable case that explains every keyword |
| Version line | `LISFLOOD-FP 8.3` | `LISFLOOD-FP 8.4` |

The per-step changes give identical depth rasters (byte for byte) and make
rain-driven CUDA runs faster than the first 8.4 build. Run time against that
build, on an A100:

| Case | First 8.4 build | Now |
|---|---:|---:|
| Halandri FV1, 9 h, uniform rain | 37.9 s | 7.5 s |
| Halandri ACC, uniform rain | 20.6 s | 5.5 s |
| ACC_NUGRID, uniform rain | 33.1 s | 14.4 s |
| UK FV2 | 10.1 s | 7.7 s |
| Alzette 10 m FV1, 12 h, 4 GPUs (double) | 173.3 s | 156.3 s |
| Alzette 10 m FV1, 12 h, 1 GPU (double) | 331.6 s | 279.9 s |

Fixes in the same work, for CUDA FV2 and DG2 on several GPUs:
- The host time step is now the value of the step itself. Before, a stale copy
  made DG2 non-deterministic and could deadlock ranks.
- Losses are applied only on the rows each rank owns. Before, infiltration was
  counted twice.
- Friction is now applied on the halo rows too.
- The DG2 slope fields are gathered before output.

Depths are now byte-identical across GPU counts: FV1 and DG2 on 1, 2 and
4 GPUs, FV2 on 1 and 2.

The `.mass` water budget now closes, without changing any depth:
- CUDA: rain is counted (in `Verror` and in the last column, as in the CPU
  solvers); `Vol` and `Area` cover the real cells only (before, FV1, FV2 and
  DG2 also counted the ghost-cell ring); ACC counts the flow through the
  domain edges; FV2 and DG2 on several GPUs count each edge once.
- Metal FV1, FV2 and DG2 count each edge face once (faces on threadgroup
  seams were counted twice), and FV1, FV2, DG2 and ACC write `.stage` depths.

The CPU solvers are unchanged. The full source change from 8.3 is
`patches/lisflood-8.3-to-8.4.patch`. To apply it to the 8.3 `source/` folder,
run `git apply --binary` from that folder.

## Multi-GPU results (Alzette basin, NVIDIA A100 40 GB)

CUDA FV1, single precision, wall time:

| Case | 1 GPU | 4 GPUs | Speed-up |
|---|---:|---:|---:|
| 10 m, 24 h (warm start) | 10.10 min | 5.00 min | 2.02× |
| 10 m, 24 h (dry start) | 5.31 min | 1.95 min | 2.72× |
| 10 m, 96 h (dry start) | 31.73 min | 10.51 min | 3.02× |
| 5 m, 24 h (dry start) | 29.40 min | 9.19 min | 3.20× |
| 5 m, 96 h (dry start) | 181.44 min | 52.70 min | 3.44× |

Depth rasters from four GPUs are identical to those from one GPU in every
case that was compared (10 m/24 h both starts, 5 m/24 h, 5 m/96 h).

Speed-up grows with grid size and run length because every step needs a global CFL minimum,
a halo exchange and rank-0 output.

CUDA FV2 and DG2, single precision, one GPU: FV2 5 m/24 h dry takes
1037.8 min, DG2 10 m/24 h dry 542.5 min. On four GPUs they must be started
through `tests_8.4/multigpu/rank_gpu.sh`, which gives each rank only its own
GPU: in a 120 s FV2 5 m test this gives 144 s against 637 s on one GPU,
whereas without it four GPUs take 729 s. The full 24 h four-GPU timings with
the wrapper have not been run yet.

## Limits

- Multi-GPU covers FV1, FV2 and DG2. ACC, ACC_NUGRID, HWFV1 and MWDG2 stop
  with an error if `GPUs` is greater than 1.
- CUDA FV2 and DG2 multi-GPU give the depths of one GPU, but
  are slow unless each rank sees only its own GPU (use `rank_gpu.sh`); FV1
  does not need this.
- On a Mac all Metal ranks share one GPU. Multi-rank there checks correctness;
  it gives no speed-up. Metal multi-rank depth rasters match one rank, but its
  `.mass` diagnostics are not yet correct.
- Channels, structures, routing, porosity and lat/long grids are not part of
  the GPU solvers, as in 8.3.
- ACC_NUGRID, HWFV1 and MWDG2 need `max_ref_lvl` with 2^L at least the larger
  of the DEM's columns and rows; they stop with a message giving the value.
- CUDA HWFV1 and MWDG2 take no rain, as in 8.3; they stop with a message if
  rain is given. The Metal versions take rain.
- The CPU FV1 and DG2 take rain only from `dynamicrainfile`.
- FREE in FV1, FV2 and DG2 copies the edge cell outward, as in 8.3: moving
  water leaves, but water at rest against the edge stays, so a valley that
  ends at a FREE edge can keep a pond there. ACC's FREE uses the bed slope and
  drains it.
- The `.mass` budget is now complete for CUDA FV1, FV2, DG2 and ACC and for
  Metal FV1, FV2, DG2 and ACC (see "What is new"). It shows two solver
  behaviours, not changed: CUDA ACC gains water while a dry, sloping domain
  first wets (on `example/` 12 % of the rain within 5 min, after which the
  budget closes), and DG2, on CUDA and Metal alike, loses thin films of rain
  (2 % of the rain on `example/`, 0.02 % on the Halandri case).
- ACC (local inertial) suits floodplains, not steep terrain: on the steep
  Halandri catchment (10 m) it becomes unstable on CUDA and Metal alike
  (depths of 56-158 m where FV1 gives at most 4.7 m). Use FV1 there.
- CUDA ACC_NUGRID writes no water budget to `.mass`, and on `example/` water
  ponds at its FREE outlet; not investigated.

## Status (25 October 2026)

| Goal | State |
|---|---|
| All seven solvers on CUDA and Metal, with losses | done, tested |
| Installer (`./install_lisflood_8.4.sh`) and `example/example.par` | done, tested on macOS (Metal + MPI) and ECMWF A100 (CUDA + MPI, float and double) |
| Clean, self-explanatory layout; this README | done |
| CUDA FV1 multi-GPU: validated and timed | done (table above) |
| CUDA FV2/DG2 multi-GPU: correct | done (byte-identical depths across GPU counts) |
| CUDA FV2/DG2 multi-GPU: timed | open: rerun the 4-GPU Alzette cases with `rank_gpu.sh` |
| `.mass` water budget (CUDA and Metal FV1, FV2, DG2, ACC) | done |
| Metal multi-rank `.mass` | open: fix on branch `claude/blissful-murdock-d236cf` (FV1 done, FV2/DG2 in progress) |
| 8.4 paper | updated except the FV2/DG2 multi-GPU rows |

The results behind every number here, and the HPC installation, are in
`/perm/lux0804/LISFLOOD_versions` on the ECMWF HPC (see its `README.md`).

## Layout

```
install_lisflood_8.4.sh   check the machine, build, test, install
build.sh               one-command build (called by the installer)
config.float.cmake     single-precision build settings
example/               example.par (every keyword explained) and its inputs
source/                complete source (CPU, source/cuda, source/metal)
patches/               8.2 → 8.3 and 8.3 → 8.4 source patches
tests/                 LISFLOOD-FP test cases (quick_test.sh)
tests_8.4/losses/      infiltration and evaporation checks (run_tests.sh, horton_free_test.py)
tests_8.4/metal/       Metal solver tests and CUDA-parity checks
tests_8.4/multigpu/    multi-GPU Slurm jobs, comparison scripts, Metal multi-rank cases
paper_8.3/             cases of the 8.3 paper
docs/DEVELOPMENT_LOG.md   how 8.4 was built and validated, in full
```

## Licence

GNU General Public License v3.0 (`LICENSE`).
