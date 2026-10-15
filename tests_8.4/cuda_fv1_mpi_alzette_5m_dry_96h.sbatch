#!/bin/bash
#SBATCH --job-name=lf84_fv1_5m_96h
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:ga100:4
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/fv1_multi_gpu_src/slurm-%j.out
set -euo pipefail
unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3
BASE=/ec/res4/scratch/lux0804/lf84/fv1_multi_gpu_src
INPUT=/perm/lux0804/LISFLOOD_versions/input/alzette_5m_dry
BIN=/perm/lux0804/LISFLOOD_versions/LISFLOOD-8.4-multigpu/build/lisflood
WORK=$BASE/alzette_5m_dry_96h_gpu_compare
rm -rf "$WORK"
mkdir -p "$WORK/single_gpu/results" "$WORK/four_gpu/results"
make_par() {
  local ranks=$1; local run_dir=$2
  {
    echo "resroot 6hr"; echo "dirroot $run_dir/results"
    echo "DEMfile $INPUT/Alzette_sub_basin_bridge_burned.asc"
    echo "manningfile $INPUT/manning_5m_dem_aligned.asc"
    echo "dynamicrainfile $INPUT/rain_5m.nc"
    echo fv1; echo cuda; echo "GPUs $ranks"; echo "sim_time 345600"
    echo initial_tstep 10; echo massint 3600; echo saveint 21600
    echo nodata_elevation -9999; echo drain_nodata
  } > "$run_dir/run.par"
}
make_par 1 "$WORK/single_gpu"; make_par 4 "$WORK/four_gpu"
cd "$WORK/single_gpu"
/usr/bin/time -f '%e' -o wall_seconds mpirun --map-by ppr:1:node --bind-to core -np 1 "$BIN" run.par > run.log 2>&1
awk '{printf "single_gpu_wall_minutes=%.4f\n", $1/60}' wall_seconds | tee timing
grep -E 'CUDA FV1 MPI rank|loop time' run.log | tee -a timing
cd "$WORK/four_gpu"
/usr/bin/time -f '%e' -o wall_seconds mpirun --map-by ppr:4:node --bind-to core -np 4 "$BIN" run.par > run.log 2>&1
awk '{printf "four_gpu_wall_minutes=%.4f\n", $1/60}' wall_seconds | tee timing
grep -E 'CUDA FV1 MPI rank|owns rows|loop time' run.log | tee -a timing
python3 "$BASE/tests_8.4/compare_cuda_fv1_mpi.py" "$WORK/single_gpu/results" "$WORK/four_gpu/results" 1e-6 0.03 | tee "$WORK/comparison.log"
python3 - "$WORK" <<'PY'
from pathlib import Path
import sys
w=Path(sys.argv[1]); one=float((w/'single_gpu/wall_seconds').read_text())/60; four=float((w/'four_gpu/wall_seconds').read_text())/60
print(f'one_gpu_minutes={one:.4f}'); print(f'four_gpu_minutes={four:.4f}'); print(f'speedup={one/four:.4f}x')
PY
