#!/bin/bash
# Give each MPI rank only its own GPU: mpirun -np 4 ./rank_gpu.sh lisflood case.par
# Needed for CUDA FV2/DG2 multi-GPU speed (see README, Limits).
export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${SLURM_LOCALID:-0}}}
exec "$@"
