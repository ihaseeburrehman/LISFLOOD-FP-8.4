#!/bin/bash
# Table 7 (scaling), 1 then 4 GPUs on one node: sbatch p_scale.sbatch BIN CASE PAR1 PAR4
#SBATCH --job-name=p_scale
#SBATCH --gres=gpu:ga100:4
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/paper84/logs/%x-%j.out
unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3
P=/ec/res4/scratch/lux0804/lf84/paper84
BIN=$1; O=$P/scaling/$2; rm -rf $O; mkdir -p $O/1gpu $O/4gpu
sed -e "s#^dirroot .*#dirroot $O/1gpu/results#" -e "s/^GPUs .*/GPUs 1/" $3 > $O/1gpu/run.par
sed -e "s#^dirroot .*#dirroot $O/4gpu/results#" -e "s/^GPUs .*/GPUs 4/" $4 > $O/4gpu/run.par
t0=$(date +%s.%N); CUDA_VISIBLE_DEVICES=0 $BIN $O/1gpu/run.par > $O/1gpu/run.log 2>&1
echo "RUN $2 1gpu rc=$? wall=$(echo "$(date +%s.%N) - $t0" | bc)"
t0=$(date +%s.%N); mpirun --map-by ppr:4:node --bind-to core -np 4 $BIN $O/4gpu/run.par > $O/4gpu/run.log 2>&1
echo "RUN $2 4gpu rc=$? wall=$(echo "$(date +%s.%N) - $t0" | bc)"
n=0; d=0; for f in $O/1gpu/results/*; do b=$(basename $f); case $b in *.mass) continue;; esac; n=$((n+1)); cmp -s $f $O/4gpu/results/$b || d=$((d+1)); done
echo "COMPARE $2 1 vs 4 GPUs: $n files, $d differ"
