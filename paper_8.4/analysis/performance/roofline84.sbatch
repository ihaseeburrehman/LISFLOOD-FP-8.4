#!/bin/bash
# Nsight Compute roofline of the CUDA FV1 time step, LISFLOOD-FP 8.4 release build (double precision).
#   sbatch roofline84.sbatch 10m | 5m
# Kernels: the default FV1 step (apply_friction_batched, update_flow_variables_xm, update_flow_variables_ym).
# Sampling: 16 consecutive launches of each kernel at a wet state on every GPU (Nsight counts launches over all
#   ranks together, so on 4 GPUs both the skip and the count are 4 times the per-GPU values).
#   10m: Alzette July 2021 paper case (radar rain, wet start), sampled from hour 30 (flood phase).
#   5m:  Alzette 5 m, WRF rain, dry start, sampled from hour 24; the step index comes from a run without the profiler.
# Kernel replay: the application runs once; only the sampled launches are replayed.
#SBATCH --job-name=roof84
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:ga100:4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/paper84/logs/%x-%j.out
set -euo pipefail
unset LD_LIBRARY_PATH
export HOME=/ec/res4/scratch/lux0804   # Nsight deploys its section files under HOME
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3
CASE=$1
P=/ec/res4/scratch/lux0804/lf84/paper84
BIN=$P/build_double/lisflood
NCU=/usr/local/apps/nvidia/24.11/Linux_x86_64/24.11/profilers/Nsight_Compute/ncu
KERNELS='^(apply_friction_batched|update_flow_variables_xm|update_flow_variables_ym)$'
NK=3; COUNT=16
ROOT=$P/roofline/${CASE}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)
mkdir -p $ROOT/reports $ROOT/runs
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
$NCU --version | tail -1
md5sum $BIN

if [ $CASE = 10m ]; then
  SRC=$P/alzette/L84_R_1km_5min_noinf_dpn_1g      # the paper run (final build): its .mass gives the step index
  T=108000
  STEP=$(awk -v t=$T '$1+0 >= t {print $4; exit}' $SRC/results/1hr.mass)
  par() {   # name ranks
    mkdir -p $ROOT/runs/$1
    grep -vE '^(dirroot|GPUs|sim_time|saveint|qoutput)' $SRC/run.par > $ROOT/$1.par
    printf 'dirroot %s\nGPUs %s\nsim_time %s\nsaveint 345600\n' $ROOT/runs/$1 $2 $((T + 120)) >> $ROOT/$1.par
  }
else
  I=/perm/lux0804/LISFLOOD_versions/input/alzette_5m_dry
  T=86400
  par() {   # name ranks
    mkdir -p $ROOT/runs/$1
    printf 'resroot 5m\ndirroot %s\nDEMfile %s\nmanningfile %s\ndynamicrainfile %s\nfv1\ncuda\nGPUs %s\nsim_time %s\ninitial_tstep 10\nmassint 600\nsaveint %s\nnodata_elevation -9999\ndrain_nodata\n' \
      $ROOT/runs/$1 $I/Alzette_sub_basin_bridge_burned.asc $I/manning_5m_dem_aligned.asc $I/rain_5m.nc $2 $3 $4 > $ROOT/$1.par
  }
  # step index at hour 24 from a run without the profiler (FV1 steps are identical on 1 and 4 GPUs)
  par steps_4gpu 4 $T $T
  t0=$(date +%s); mpirun --map-by ppr:4:node --bind-to core -np 4 $BIN $ROOT/steps_4gpu.par > $ROOT/runs/steps_4gpu.log 2>&1
  echo "steps run (4 GPUs, no profiler): $(( $(date +%s) - t0 )) s"
  STEP=$(awk -v t=$T '$1+0 >= t {print $4; exit}' $ROOT/runs/steps_4gpu/5m.mass)
  [ -n "$STEP" ] || STEP=$(tail -n 1 $ROOT/runs/steps_4gpu/5m.mass | awk '{print $4}')
  par0() { par $1 $2 $((T + 120)) 345600; }
fi
SKIP=$(( (STEP - 1) * NK ))
echo "sample from step $STEP (sim time $T s): launch-skip $SKIP, launch-count $((NK * COUNT)) per GPU"

profile() {   # name ranks
  local name=$1 ranks=$2
  if [ $CASE = 10m ]; then par $name $ranks; else par0 $name $ranks; fi
  local ncu=($NCU --target-processes all --replay-mode kernel
    --kernel-name-base function --kernel-name "regex:$KERNELS"
    --launch-skip $((SKIP * ranks)) --launch-count $((NK * COUNT * ranks))
    --section SpeedOfLight --section SpeedOfLight_RooflineChart --section SpeedOfLight_HierarchicalDoubleRooflineChart
    --export $ROOT/reports/${name}_pid_%p --force)
  local t0=$(date +%s)
  if [ $ranks -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=0 "${ncu[@]}" $BIN $ROOT/$name.par > $ROOT/runs/$name.log 2>&1
  else
    "${ncu[@]}" mpirun --map-by ppr:4:node --bind-to core -np 4 $BIN $ROOT/$name.par > $ROOT/runs/$name.log 2>&1
  fi
  echo "PROFILED $name ranks=$ranks wall=$(( $(date +%s) - t0 )) s reports=$(ls $ROOT/reports | grep -c "^${name}_")"
}
profile ${CASE}_4gpu 4
profile ${CASE}_1gpu 1
for r in $ROOT/reports/*.ncu-rep; do
  $NCU --import $r --page raw --csv > ${r%.ncu-rep}.csv
done
echo "reports in $ROOT/reports"
