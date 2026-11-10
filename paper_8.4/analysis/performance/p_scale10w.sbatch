#!/bin/bash
# Table 7, 10 m 24 h stored start: the par of the earlier runs has no GPUs line, so it is added here
#SBATCH --job-name=p_scale
#SBATCH --gres=gpu:ga100:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --mem=128G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/paper84/logs/%x-%j.out
unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3
P=/ec/res4/scratch/lux0804/lf84/paper84; BIN=$P/build/lisflood; S=/perm/lux0804/LISFLOOD_versions/results_8.4/fv1_multigpu/10m_24h_warm/four_gpu.par
O=$P/scaling/10m_24h_warm; rm -rf $O; mkdir -p $O/1gpu $O/4gpu
for n in 1 4; do (sed -e "s#^dirroot .*#dirroot $O/${n}gpu/results#" -e "/^GPUs/d" $S; echo "GPUs $n") > $O/${n}gpu/run.par; done
t0=$(date +%s.%N); CUDA_VISIBLE_DEVICES=0 $BIN $O/1gpu/run.par > $O/1gpu/run.log 2>&1
echo "RUN 10m_24h_warm 1gpu rc=$? wall=$(echo "$(date +%s.%N) - $t0" | bc)"
t0=$(date +%s.%N); mpirun --map-by ppr:4:node --bind-to core -np 4 $BIN $O/4gpu/run.par > $O/4gpu/run.log 2>&1
echo "RUN 10m_24h_warm 4gpu rc=$? wall=$(echo "$(date +%s.%N) - $t0" | bc)"
n=0; d=0; for f in $O/1gpu/results/*; do b=$(basename $f); case $b in *.mass) continue;; esac; n=$((n+1)); cmp -s $f $O/4gpu/results/$b || d=$((d+1)); done
echo "COMPARE 10m_24h_warm 1 vs 4 GPUs: $n files, $d differ"
