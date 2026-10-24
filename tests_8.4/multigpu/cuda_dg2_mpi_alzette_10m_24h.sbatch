#!/bin/bash
# Timing: 1 vs 4 A100s, DG2 solver, 10 m Alzette dry-start 6 h — float precision.
# Dry start used because no DG2-format startfile exists (FV1 startfiles lack slope files).
# 6 h keeps dt large (~10 s, pre-rain) so the run completes quickly; speedup is still valid.
# Generates DG2 DEM slope files (once) before the runs.
#SBATCH --job-name=lf84_dg2_10m6h
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:ga100:4
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/dg2_multi_gpu/slurm-%j.out
set -euo pipefail

unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3

SRC=/perm/lux0804/LISFLOOD_versions/LISFLOOD-8.4
CONFIG=$SRC/config.float.cmake
BUILD=/ec/res4/scratch/lux0804/lf84/build_dg2_float_${SLURM_JOB_ID}
DEM=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/Alzette_sub_basin_10m_bridge_burn.asc
MANNING=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/manning.n.ascii
RAIN=/ec/res4/scratch/lux0804/metal_vs_cuda_10m_24h/rain_wrf1p3_24h_from_14jul18.nc
BCI=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/Border.bci
BASE=/ec/res4/scratch/lux0804/lf84/dg2_multi_gpu
WORK=$BASE/alzette_10m_24h_float_${SLURM_JOB_ID}
mkdir -p "$BASE" "$WORK/single_gpu/results" "$WORK/four_gpu/results"

echo "=== Building LISFLOOD DG2 float with CUDA MPI ==="
cmake -B "$BUILD" -S "$SRC/source" \
  -DCMAKE_BUILD_TYPE=Release \
  -D_CONFIG="$CONFIG" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DLISFLOOD_ENABLE_CUDA_MPI=ON \
  -DCMAKE_CXX_FLAGS="-O3" 2>&1
cmake --build "$BUILD" --target lisflood generateDG2DEM -j8 2>&1
BIN="$BUILD/lisflood"
GENDG2="$BUILD/generateDG2DEM"
echo "=== Build complete: $BIN ==="

# Generate DG2 DEM slope files for the 10 m grid (idempotent)
DEM_1X="${DEM%.asc}.asc1x"
if [ ! -f "$DEM_1X" ]; then
  echo "=== Generating DG2 DEM slope files for 10 m grid ==="
  cat > /tmp/gen_dg2dem_${SLURM_JOB_ID}.par <<PAR
DEMfile $DEM
nodata_elevation -9999
PAR
  "$GENDG2" /tmp/gen_dg2dem_${SLURM_JOB_ID}.par
  echo "=== DG2 slope files generated ==="
else
  echo "=== DG2 slope files already exist, skipping ==="
fi

make_par() {
  local ranks=$1 run_dir=$2
  cat > "$run_dir/run.par" <<PAR
resroot 1hr
dirroot $run_dir/results
DEMfile $DEM
manningfile $MANNING
dynamicrainfile $RAIN
bcifile $BCI
dg2
cuda
GPUs $ranks
sim_time 21600
initial_tstep 10
massint 1800
saveint 3600
nodata_elevation -9999
drain_nodata
PAR
}

make_par 1 "$WORK/single_gpu"
make_par 4 "$WORK/four_gpu"

echo "=== Running DG2 single GPU ==="
cd "$WORK/single_gpu"
/usr/bin/time -f '%e' -o wall_seconds \
  mpirun --map-by ppr:1:node --bind-to core -np 1 "$BIN" run.par > run.log 2>&1
awk '{printf "single_gpu_wall_minutes=%.4f\n", $1/60}' wall_seconds | tee timing
grep -E 'CUDA DG2 MPI rank|loop time' run.log | tee -a timing

echo "=== Running DG2 four GPUs ==="
cd "$WORK/four_gpu"
/usr/bin/time -f '%e' -o wall_seconds \
  mpirun --map-by ppr:4:node --bind-to core -np 4 "$BIN" run.par > run.log 2>&1
awk '{printf "four_gpu_wall_minutes=%.4f\n", $1/60}' wall_seconds | tee timing
grep -E 'CUDA DG2 MPI rank|owns rows|loop time' run.log | tee -a timing

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

echo "=== DG2 10 m 6 h dry float timing complete ==="
