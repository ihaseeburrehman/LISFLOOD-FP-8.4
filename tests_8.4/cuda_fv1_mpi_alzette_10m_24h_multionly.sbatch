#!/bin/bash
# Benchmark an optimized four-GPU build against the saved 24 h one-GPU result.
#SBATCH --job-name=lf84_fv1_24h_ipc
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:ga100:4
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/fv1_multi_gpu_src/slurm-%j.out
set -euo pipefail

unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3

BASE=/ec/res4/scratch/lux0804/lf84/fv1_multi_gpu_src
CASE=/ec/res4/scratch/lux0804/metal_vs_cuda_10m_24h
REFERENCE=$BASE/alzette_10m_24h_gpu_compare_optimized/single_gpu/results
WORK=$BASE/alzette_10m_24h_four_gpu_device_dt
BIN=$BASE/build/lisflood
mkdir -p "$WORK/results"
cd "$WORK"

test -s "$REFERENCE/1hr.mass"
cat > run.par <<EOF
resroot 1hr
dirroot $WORK/results
DEMfile /ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/Alzette_sub_basin_10m_bridge_burn.asc
manningfile /ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/manning.n.ascii
dynamicrainfile $CASE/rain_wrf1p3_24h_from_14jul18.nc
startfile /ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_all_models_96h/wrf1p3_afterda_18utc/results/6hr-0004.wd
fv1
cuda
GPUs 4
sim_time 86400
initial_tstep 10
massint 600
saveint 3600
bcifile /ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/Border.bci
nodata_elevation -9999
drain_nodata
qoutput
EOF

/usr/bin/time -f '%e' -o wall_seconds \
  mpirun --map-by ppr:4:node --bind-to core -np 4 "$BIN" run.par > run.log 2>&1
grep -E 'CUDA FV1 MPI rank|loop time' run.log | tee timing
awk '{printf "wall_minutes=%.4f\n", $1/60}' wall_seconds | tee -a timing
python3 "$BASE/tests_8.4/compare_cuda_fv1_mpi.py" \
  "$REFERENCE" "$WORK/results" | tee comparison.log
echo 'optimized four-GPU 24-hour comparison passed'
