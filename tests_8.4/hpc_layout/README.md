# ECMWF HPC layout

The curated installation root is:

`/ec/res4/scratch/lux0804/LISFLOOD_versions`

It contains one shared `input` directory and three independent installations:

- `LISFLOOD-8.3`: the 8.3 CUDA build and its own `output` directory.
- `LISFLOOD-8.4-single`: 8.4 commit `5d50b13`, before multi-GPU FV1, and its own `output` directory.
- `LISFLOOD-8.4-multigpu`: 8.4 commit `3be328a`, with `GPUs 4` validation, and its own `output` directory.

Each installation contains `source`, `build/lisflood`, `run.par`, and `output`.
All three `run.par` files use the same files in the top-level `input` directory.
The time-consistent initial state available on HPC is `13_july_18hr.wd`; the
96-hour WRF rainfall also begins at 18:00 UTC on 13 July 2021.

Load the normal ECMWF modules before running. The single-GPU commands are:

```sh
LISFLOOD-8.3/build/lisflood LISFLOOD-8.3/run.par
LISFLOOD-8.4-single/build/lisflood LISFLOOD-8.4-single/run.par
```

The four-GPU version requires a four-GPU Slurm allocation and four MPI ranks:

```sh
mpirun --bind-to none -np 4 \
  LISFLOOD-8.4-multigpu/build/lisflood \
  LISFLOOD-8.4-multigpu/run.par
```
