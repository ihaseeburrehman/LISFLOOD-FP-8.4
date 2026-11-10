#!/bin/bash
# sbatch [--gpus N] run_case.sbatch CASE SOLVER SIM_TIME [NGPU] [BUILD]
#   CASE      folder under cases/ (from make_case.py)
#   SOLVER    acc | fv1 | fv2 | dg2
#   SIM_TIME  seconds (depths saved every 3600 s, as the reference frames)
#SBATCH --job-name=fcb_run
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/run-%j.out
set -e
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3
CASE=$1; SOLVER=$2; T=$3; NGPU=${4:-1}; BUILD=${5:-build}
LF=/perm/lux0804/LISFLOOD_versions/LISFLOOD-8.4/$BUILD/lisflood
cd /ec/res4/scratch/lux0804/lf84/floodcastbench/cases/$CASE
OUT=${SOLVER}_${NGPU}gpu_${T}s
case $SOLVER in
  acc) S="acceleration
theta 0.7
cfl 0.7" ;;
  *) S=$SOLVER ;;
esac
cat > $OUT.par <<PAR
DEMfile dem.asc
manningfile manning.asc
startfile start.wd
dynamicrainfile rain.nc
bcifile free.bci
resroot r
dirroot $OUT
cuda
$S
fpfric 0.035
sim_time $T
saveint 3600
massint 600
GPUs $NGPU
PAR
[ -f indus.bdy ] && echo "bdyfile indus.bdy" >> $OUT.par
echo "== $CASE $SOLVER $NGPU GPU, $T s, $LF  ($(date))"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
START=$(date +%s)
if [ "$NGPU" -gt 1 ]; then
  mpirun -np $NGPU /perm/lux0804/LISFLOOD_versions/LISFLOOD-8.4/rank_gpu.sh $LF $OUT.par
else
  $LF $OUT.par
fi
echo "WALL $(( $(date +%s) - START )) s"
