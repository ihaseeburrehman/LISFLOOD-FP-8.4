#!/bin/bash
# Compare one and four A100s on the 33.89-million-cell, initially dry 5 m case — FV2 solver.
#SBATCH --job-name=lf84_fv2_5m_dry
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:ga100:4
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/fv2_multi_gpu/slurm-%j.out
set -euo pipefail

unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3

SRC=/perm/lux0804/LISFLOOD_versions/LISFLOOD-8.4-multigpu
BUILD=/ec/res4/scratch/lux0804/lf84/build_fv2_${SLURM_JOB_ID}
INPUT=/perm/lux0804/LISFLOOD_versions/input/alzette_5m_dry
BASE=/ec/res4/scratch/lux0804/lf84/fv2_multi_gpu
WORK=$BASE/alzette_5m_dry_24h_${SLURM_JOB_ID}
mkdir -p "$BASE" "$WORK/single_gpu/results" "$WORK/four_gpu/results"

# Build with CUDA MPI enabled
echo "=== Building LISFLOOD with CUDA MPI ==="
cmake -B "$BUILD" -S "$SRC/source" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DLISFLOOD_ENABLE_CUDA_MPI=ON \
  -DCMAKE_CXX_FLAGS="-O3" 2>&1
cmake --build "$BUILD" --target lisflood -j8 2>&1
BIN="$BUILD/lisflood"
echo "=== Build complete: $BIN ==="

make_par() {
  local ranks=$1
  local run_dir=$2
  cat > "$run_dir/run.par" <<PAR
resroot 6hr
dirroot $run_dir/results
DEMfile $INPUT/Alzette_sub_basin_bridge_burned.asc
manningfile $INPUT/manning_5m_dem_aligned.asc
dynamicrainfile $INPUT/rain_5m.nc
fv2
cuda
GPUs $ranks
sim_time 86400
initial_tstep 10
massint 3600
saveint 21600
nodata_elevation -9999
drain_nodata
PAR
}

make_par 1 "$WORK/single_gpu"
make_par 4 "$WORK/four_gpu"

echo "=== Running FV2 single GPU ==="
cd "$WORK/single_gpu"
/usr/bin/time -f '%e' -o wall_seconds \
  mpirun --map-by ppr:1:node --bind-to core -np 1 "$BIN" run.par > run.log 2>&1
awk '{printf "single_gpu_wall_minutes=%.4f\n", $1/60}' wall_seconds | tee timing
grep -E 'CUDA FV2 MPI rank|loop time' run.log | tee -a timing

echo "=== Running FV2 four GPUs ==="
cd "$WORK/four_gpu"
/usr/bin/time -f '%e' -o wall_seconds \
  mpirun --map-by ppr:4:node --bind-to core -np 4 "$BIN" run.par > run.log 2>&1
awk '{printf "four_gpu_wall_minutes=%.4f\n", $1/60}' wall_seconds | tee timing
grep -E 'CUDA FV2 MPI rank|owns rows|loop time' run.log | tee -a timing

echo "=== Comparing results ==="
python3 "$SRC/tests_8.4/compare_cuda_fv2_mpi.py" \
  "$WORK/single_gpu/results" "$WORK/four_gpu/results" 0.03 1e-4 \
  | tee "$WORK/comparison.log"

python3 - "$WORK" <<'PY'
from pathlib import Path
import sys
w = Path(sys.argv[1])
one = float((w/'single_gpu/wall_seconds').read_text()) / 60
four = float((w/'four_gpu/wall_seconds').read_text()) / 60
print(f'one_gpu_minutes={one:.4f}')
print(f'four_gpu_minutes={four:.4f}')
print(f'speedup={one/four:.4f}x')
print(f'wall_reduction={(1-four/one)*100:.2f}%')
PY
