# LISFLOOD-FP 8.4

2D flood model with every GPU solver on NVIDIA (CUDA) and Apple Silicon (Metal),
infiltration and evaporation on every GPU solver, and multi-GPU runs over MPI.

## Build

```bash
./build.sh
```

The script picks the back end itself: Metal on macOS, CUDA on Linux when `nvcc`
is found. It switches multi-GPU on when an MPI compiler is found. The result is
`build/lisflood`; keep the `.metallib` files next to it on a Mac.

| Setting | Default | Example |
|---|---|---|
| `PRECISION` | `float` | `PRECISION=double ./build.sh` |
| `MPI` | `auto` | `MPI=off ./build.sh` |
| `CUDA_ARCH` | detected with `nvidia-smi`, else `80` | `CUDA_ARCH=90 ./build.sh` (H100) |
| `BUILD` | `build` | `BUILD=build_double ./build.sh` |

**Requirements.** On Linux: CMake ≥ 3.13, CUDA ≥ 11.4, NetCDF-C and OpenMP, plus
MPI for multi-GPU. At ECMWF:
`module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3`. On a Mac:
Xcode and `brew install cmake libomp netcdf`, plus MacPorts `mpich` for
multi-rank (Open MPI 5 hangs at start-up on macOS).

**Check.** `tests/quick_test.sh build/lisflood` on CUDA, or
`GPU=metal tests/quick_test.sh build/lisflood` on a Mac, should print
`QUICK TEST PASSED`. `python3 tests_8.4/losses/horton_free_test.py build WORK`
checks Horton and the FREE boundary on all seven solvers in a few minutes.

## Run

```text
fv1            # or fv2, dg2, acc, acc_nugrid, hwfv1, mwdg2
cuda           # or metal
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
| Build | several platform-specific CMake lines | `./build.sh` |
| Version line | `LISFLOOD-FP 8.3` | `LISFLOOD-FP 8.4` |

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
- CUDA FV2 and DG2 multi-GPU give the same `.mass` record as one GPU, but
  are slow unless each rank sees only its own GPU (use `rank_gpu.sh`); FV1
  does not need this.
- On a Mac all Metal ranks share one GPU. Multi-rank there checks correctness;
  it gives no speed-up. Metal multi-rank depth rasters match one rank, but its
  `.mass` diagnostics are not yet correct.
- Channels, structures, routing, porosity and lat/long grids are not part of
  the GPU solvers, as in 8.3.

## Status (19 October 2026)

| Goal | State |
|---|---|
| All seven solvers on CUDA and Metal, with losses | done, tested |
| One-command build (`./build.sh`) | done, tested on macOS (Metal + MPI) and ECMWF (CUDA + MPI) |
| Clean, self-explanatory layout; this README | done |
| CUDA FV1 multi-GPU: validated and timed | done (table above) |
| CUDA FV2/DG2 multi-GPU: correct | done (same `.mass` as one GPU) |
| CUDA FV2/DG2 multi-GPU: timed | open: rerun 4-GPU cases with `rank_gpu.sh`; then fix device binding in `source/cuda/{fv2,dg2}/cuda_*_simulate.cu` |
| Metal multi-rank `.mass` | open: fix on branch `claude/blissful-murdock-d236cf` (FV1 done, FV2/DG2 in progress) |
| 8.4 paper | updated except the FV2/DG2 multi-GPU rows |
| Git | reorganisation and the FV2/DG2 fix are not committed yet |

The results behind every number here, and the HPC installation, are in
`/perm/lux0804/LISFLOOD_versions` on the ECMWF HPC (see its `README.md`).

## Layout

```
build.sh               one-command build
config.float.cmake     single-precision build settings
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
