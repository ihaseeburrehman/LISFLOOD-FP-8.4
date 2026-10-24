#!/bin/bash
# Locate CUDA FV2/DG2 MPI memory errors: short 100 m Alzette runs under
# compute-sanitizer, float and double builds, one and two ranks.
#SBATCH --job-name=lf84_fv2_sanitize
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:ga100:2
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/fv2_multi_gpu/slurm-%j.out
set -uo pipefail

unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3

SRC=/perm/lux0804/LISFLOOD_versions/LISFLOOD-8.4
CASE=/ec/res4/scratch/lux0804/lf84/alzette_all_compare/case
WORK=${WORK:-/ec/res4/scratch/lux0804/lf84/fv2_multi_gpu/sanitize_${SLURM_JOB_ID}}
SAN=$CUDA_HOME/compute-sanitizer/compute-sanitizer
mkdir -p "$WORK"; cd "$WORK"
mkdir -p data; for f in "$CASE"/data/*; do ln -sfn "$f" data/; done
ln -sfn "$CASE/Border.bci" Border.bci

for prec in float double; do
  case $prec in float) CONFIG=$SRC/config.float.cmake ;; double) CONFIG=$SRC/source/config.default.cmake ;; esac
  B=$WORK/build_$prec
  if [ ! -x "$B/lisflood" ]; then
  cmake -S "$SRC/source" -B "$B" -DCMAKE_BUILD_TYPE=Release -D_CONFIG="$CONFIG" \
    -DCMAKE_CUDA_ARCHITECTURES=80 -DLISFLOOD_ENABLE_CUDA_MPI=ON \
    -DCMAKE_CUDA_FLAGS="-lineinfo" > build_$prec.log 2>&1
  cmake --build "$B" --target lisflood generateDG2DEM -j8 >> build_$prec.log 2>&1 || { echo "build $prec failed"; continue; }
  fi
  for solver in fv2 dg2; do
    for n in 1 2; do
      run=${solver}_${prec}_${n}
      cat > $run.par <<EOF
DEMfile data/dem_100m.asc
startfile data/start_100m.wd
manningfile data/manning_100m.asc
dynamicrainfile data/rain_100m.nc
bcifile Border.bci
$solver
cuda
GPUs $n
resroot r
dirroot $run
sim_time 600
initial_tstep 10
massint 300
saveint 300
nodata_elevation -9999
drain_nodata
EOF
      if [ $solver = dg2 ] && [ ! -f data/dem_100m.asc1x ]; then
        printf 'DEMfile data/dem_100m.asc\nnodata_elevation -9999\n' > gen.par
        "$B/generateDG2DEM" gen.par > gen_dg2.log 2>&1 || echo "generateDG2DEM failed"
      fi
      echo "=== $run ==="
      mpirun --map-by ppr:$n:node -np $n "$SAN" --tool memcheck \
        --print-limit 20 "$B/lisflood" $run.par > $run.log 2>&1
      echo "exit=$?"
      grep -m 12 -E 'Invalid|at 0x|by thread|Address|ERROR SUMMARY|loop time|CUDA error' $run.log
    done
  done
done
