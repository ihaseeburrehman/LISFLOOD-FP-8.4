#!/bin/bash
# Timing: 1 vs 4 A100s, FV2 solver, 10 m Alzette warm-start 24 h — float precision.
# Matches the FV1 10m/24h timing case so speedups are directly comparable.
#SBATCH --job-name=lf84_fv2_10m24h
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:ga100:4
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/fv2_multi_gpu/slurm-%j.out
set -euo pipefail

unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3

SRC=/perm/lux0804/LISFLOOD_versions/LISFLOOD-8.4
CONFIG=$SRC/config.float.cmake
BUILD=/ec/res4/scratch/lux0804/lf84/build_fv2_float_${SLURM_JOB_ID}
DEM=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/Alzette_sub_basin_10m_bridge_burn.asc
MANNING=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/manning.n.ascii
RAIN=/ec/res4/scratch/lux0804/metal_vs_cuda_10m_24h/rain_wrf1p3_24h_from_14jul18.nc
START=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_all_models_96h/wrf1p3_afterda_18utc/results/6hr-0004.wd
BCI=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/Border.bci
BASE=/ec/res4/scratch/lux0804/lf84/fv2_multi_gpu
WORK=$BASE/alzette_10m_24h_float_${SLURM_JOB_ID}
mkdir -p "$BASE" "$WORK/single_gpu/results" "$WORK/four_gpu/results"

echo "=== Building LISFLOOD FV2 float with CUDA MPI ==="
cmake -B "$BUILD" -S "$SRC/source" \
  -DCMAKE_BUILD_TYPE=Release \
  -D_CONFIG="$CONFIG" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DLISFLOOD_ENABLE_CUDA_MPI=ON \
  -DCMAKE_CXX_FLAGS="-O3" 2>&1
cmake --build "$BUILD" --target lisflood -j8 2>&1
BIN="$BUILD/lisflood"
echo "=== Build complete: $BIN ==="

make_par() {
  local ranks=$1 run_dir=$2
  cat > "$run_dir/run.par" <<PAR
resroot 1hr
dirroot $run_dir/results
DEMfile $DEM
manningfile $MANNING
dynamicrainfile $RAIN
startfile $START
bcifile $BCI
fv2
cuda
GPUs $ranks
sim_time 86400
initial_tstep 10
massint 3600
saveint 3600
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

python3 - "$WORK" <<'PY'
from pathlib import Path
import sys
w = Path(sys.argv[1])
one = float((w/'single_gpu/wall_seconds').read_text()) / 60
four = float((w/'four_gpu/wall_seconds').read_text()) / 60
print(f'one_gpu_minutes={one:.4f}')
print(f'four_gpu_minutes={four:.4f}')
print(f'speedup={one/four:.4f}x')
PY

echo "=== FV2 10 m 24 h float timing complete ==="
