# LISFLOOD_versions (ECMWF HPC)

Updated 19 October 2026. Load the modules first:

```sh
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3
```

## Folders

| Folder | What it is | Use it for |
|---|---|---|
| `LISFLOOD-8.4/` | current 8.4 source (identical to the local repo, branch `codex/fv1-multi-gpu` + uncommitted reorganisation and FV2/DG2 fix of 16 Oct); `build/` is the float, CUDA-MPI build | all new 8.4 runs (FV1, FV2, DG2, ACC, ACC_NUGRID, HWFV1, MWDG2, losses, multi-GPU) |
| `LISFLOOD-8.4-fv1-paper-build/` | frozen 8.4 build (commit 24b159c) that produced the published FV1 multi-GPU timings | reproducing those numbers only |
| `LISFLOOD-8.3/` | 8.3 CUDA build | baseline comparisons |
| `input/` | shared Alzette inputs | see below |
| `results_8.4/` | curated results behind the 8.4 paper | see `results_8.4/README.md` |

Build `LISFLOOD-8.4` again with one command (float + MPI by default):

```sh
cd LISFLOOD-8.4 && ./build.sh            # PRECISION=double ./build.sh for double
```

Run on one GPU, or on four with MPI (`-np` must equal `GPUs` in the `.par`):

```sh
LISFLOOD-8.4/build/lisflood case.par
mpirun -np 4 ./rank_gpu.sh LISFLOOD-8.4/build/lisflood case.par
```

**Multi-GPU FV2/DG2: give each rank only its own GPU.** With all four GPUs
visible, FV2 on 4 GPUs runs slower than on 1 GPU (probe job 30077187, see
`results_8.4/fv2_dg2_multigpu/gpu_visibility_probe`). `rank_gpu.sh` is:

```sh
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
exec "$@"
```

## input/

| File | Grid | Notes |
|---|---|---|
| `dem_10m.asc` | 10 m | clean (same md5 as `clean_dem/alzette_10m_dem_original.asc`) |
| `manning_10m.asc`, `Border.bci` | 10 m | |
| `rain_wrf1p3_d03_00utc_102h.nc` | 10 m | WRF d03 rain from 13 Jul 2021 00 UTC, 0–102 h: dry-start runs |
| `rain_wrf1p3_96h.nc`, `13_july_18hr.wd` | 10 m | rain from 13 Jul 18 UTC + matching warm start: the older warm-start runs |
| `alzette_5m_dry/` | 5 m | DEM (restored to the clean original on 19 Oct), Manning, rain from 13 Jul 00 UTC |
| `clean_dem/` | 5 m, 10 m | read-only originals with `MD5SUMS` |

`generateDG2DEM` **overwrites the DEM it is given**. Always run it on a copy
inside the run folder (as `LISFLOOD-8.4/tests_8.4/multigpu/cuda_mpi_timing.sbatch`
does). It corrupted the shared 5 m DEM on 16 Oct; the damaged file is in
`/perm/lux0804/to_delete_after_review/`.
