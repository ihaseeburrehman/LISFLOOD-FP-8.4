#!/bin/bash
# FV2/DG2 single-GPU memcheck: dry start, no boundary file (as the 5 m timing
# case), on the 100 m grid and then 60 s of the 5 m grid. Reuses the builds of
# the sanitize job given in BUILDS.
#SBATCH --job-name=lf84_fv2_san5m
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:ga100:1
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/fv2_multi_gpu/slurm-%j.out
set -uo pipefail
unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3
SAN=$CUDA_HOME/compute-sanitizer/compute-sanitizer
BUILDS=${BUILDS:-/ec/res4/scratch/lux0804/lf84/fv2_multi_gpu/sanitize_39161177}
WORK=/ec/res4/scratch/lux0804/lf84/fv2_multi_gpu/sanitize5m_${SLURM_JOB_ID}
I5=/perm/lux0804/LISFLOOD_versions/input/alzette_5m_dry
mkdir -p "$WORK"; cd "$WORK"
SRC=/perm/lux0804/LISFLOOD_versions/LISFLOOD-8.4
if [ ! -x "$BUILDS/build_float/lisflood" ]; then
  mkdir -p "$BUILDS/data"
  for f in /ec/res4/scratch/lux0804/lf84/alzette_all_compare/case/data/*; do ln -sfn "$f" "$BUILDS/data/"; done
  cmake -S "$SRC/source" -B "$BUILDS/build_float" -DCMAKE_BUILD_TYPE=Release \
    -D_CONFIG="$SRC/config.float.cmake" -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DLISFLOOD_ENABLE_CUDA_MPI=ON -DCMAKE_CUDA_FLAGS="-lineinfo" > build.log 2>&1
  cmake --build "$BUILDS/build_float" --target lisflood -j8 >> build.log 2>&1 || { echo build failed; exit 1; }
fi
for grid in 100m 5m; do
  if [ $grid = 100m ]; then
    DEM=$BUILDS/data/dem_100m.asc; MAN=$BUILDS/data/manning_100m.asc; RAIN=$BUILDS/data/rain_100m.nc; T=600
  else
    DEM=$I5/Alzette_sub_basin_bridge_burned.asc; MAN=$I5/manning_5m_dem_aligned.asc; RAIN=$I5/rain_5m.nc; T=60
  fi
  for prec in float; do
    for solver in fv2 dg2; do
      run=${solver}_${prec}_${grid}
      cat > $run.par <<EOF
DEMfile $DEM
manningfile $MAN
dynamicrainfile $RAIN
$solver
cuda
GPUs 1
resroot r
dirroot $run
sim_time $T
initial_tstep 10
massint 30
saveint $T
nodata_elevation -9999
drain_nodata
EOF
      echo "=== $run (plain) ==="
      mpirun -np 1 "$BUILDS/build_$prec/lisflood" $run.par > ${run}_plain.log 2>&1; echo "exit=$?"
      grep -v UCX ${run}_plain.log | grep -E 'error|ERROR|loop time' | head -5
      echo "=== $run (memcheck) ==="
      mpirun -np 1 "$SAN" --tool memcheck --print-limit 10 "$BUILDS/build_$prec/lisflood" $run.par > ${run}_san.log 2>&1; echo "exit=$?"
      grep -E 'Invalid|at 0x|by thread|is out of bounds|ERROR SUMMARY|loop time|CUDA error|ERROR' ${run}_san.log | head -25
    done
  done
done
