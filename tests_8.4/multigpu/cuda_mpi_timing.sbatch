#!/bin/bash
# Wall time of one CUDA multi-GPU run on the Alzette basin: full 24 h, dry start
# at 13 July 2021 00:00 UTC, single precision. One run per job, so the 1-GPU and
# 4-GPU runs of a case can go in parallel on identical A100 nodes.
#
#   sbatch --export=ALL,SOLVER=fv2,GRID=5m,GPUS=1 --gres=gpu:ga100:1 --ntasks=1 --time=40:00:00 cuda_mpi_timing.sbatch
#   sbatch --export=ALL,SOLVER=fv2,GRID=5m,GPUS=4 cuda_mpi_timing.sbatch
#
# SOLVER: fv1 | fv2 | dg2    GRID: 5m | 10m    GPUS: 1 | 4
# Every rank sees only its own GPU (rank_gpu.sh): with all GPUs visible, FV2
# and DG2 on 4 GPUs run slower than on 1 (probe job 30077187).
#SBATCH --job-name=lf84_timing
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:ga100:4
#SBATCH --time=40:00:00
#SBATCH --mem=128G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/timing/slurm-%j.out
set -euo pipefail

unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3

: "${SOLVER:?set SOLVER}" "${GRID:?set GRID}" "${GPUS:?set GPUS}"
SRC=/perm/lux0804/LISFLOOD_versions/LISFLOOD-8.4
BASE=/ec/res4/scratch/lux0804/lf84/timing
BUILD=$BASE/build_${SLURM_JOB_ID}
WORK=$BASE/${SOLVER}_${GRID}_24h_${GPUS}gpu_${SLURM_JOB_ID}
mkdir -p "$WORK/results" "$WORK/inputs"
# The DEM is always a private copy of a read-only original: generateDG2DEM
# overwrites the DEM it is given, and once rewrote the shared input files.
CLEAN=/perm/lux0804/LISFLOOD_versions/input/clean_dem
(cd "$CLEAN" && md5sum -c --quiet MD5SUMS)

case $GRID in
  5m)
    IN=/perm/lux0804/LISFLOOD_versions/input/alzette_5m_dry
    cp "$CLEAN/alzette_5m_dem_original.asc" "$WORK/inputs/dem.asc"
    ln -sfn "$IN/manning_5m_dem_aligned.asc" "$WORK/inputs/manning.asc"
    ln -sfn "$IN/rain_5m.nc" "$WORK/inputs/rain.nc"
    BCI_LINE="" ;;
  10m)
    IN=/perm/lux0804/LISFLOOD_versions/input
    cp "$CLEAN/alzette_10m_dem_original.asc" "$WORK/inputs/dem.asc"
    ln -sfn "$IN/manning_10m.asc" "$WORK/inputs/manning.asc"
    ln -sfn /ec/res4/scratch/lux0804/lf84/inputs_10m_00utc/rain_wrf1p3_d03_00utc_102h.nc "$WORK/inputs/rain.nc"
    BCI_LINE="bcifile $IN/Border.bci" ;;
  *) echo "GRID must be 5m or 10m"; exit 1 ;;
esac

echo "=== build: $SOLVER, float, CUDA MPI ==="
cmake -S "$SRC/source" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -D_CONFIG="$SRC/config.float.cmake" -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DLISFLOOD_ENABLE_CUDA_MPI=ON > "$WORK/build.log" 2>&1
cmake --build "$BUILD" --target lisflood generateDG2DEM -j8 >> "$WORK/build.log" 2>&1

cd "$WORK"
if [ "$SOLVER" = dg2 ]; then
  # generateDG2DEM rewrites inputs/dem.asc and writes its slope files beside
  # it; both stay inside this work folder.
  printf 'DEMfile inputs/dem.asc\nnodata_elevation -9999\n' > gen_dg2dem.par
  "$BUILD/generateDG2DEM" gen_dg2dem.par > gen_dg2dem.log 2>&1
fi

cat > run.par <<PAR
resroot 6hr
dirroot $WORK/results
DEMfile inputs/dem.asc
manningfile inputs/manning.asc
dynamicrainfile inputs/rain.nc
$BCI_LINE
$SOLVER
cuda
GPUs $GPUS
sim_time 86400
initial_tstep 10
massint 3600
saveint 21600
nodata_elevation -9999
drain_nodata
PAR

echo "=== run: $SOLVER $GRID 24 h on $GPUS GPU(s) ==="
/usr/bin/time -f '%e' -o wall_seconds \
  mpirun --map-by ppr:$GPUS:node --bind-to core -np "$GPUS" \
  "$SRC/tests_8.4/multigpu/rank_gpu.sh" "$BUILD/lisflood" run.par > run.log 2>&1
python3 - "$WORK" <<'PY'
from pathlib import Path
import sys
w = Path(sys.argv[1])
rows = (w/'results/6hr.mass').read_text().split('\n')[1:]
t_end = max(float(r.split()[0]) for r in rows if r.split())
assert t_end >= 86400, f'simulation ended at {t_end} s'
print(f'final_time={t_end:.3f}s')
print(f'wall_minutes={float((w/"wall_seconds").read_text())/60:.4f}')
PY
