#!/bin/bash
#SBATCH --job-name=p_noinf4
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:ga100:4
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/paper84/logs/%x-%j.out
unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3
t0=$(date +%s.%N); mpirun --map-by ppr:4:node --bind-to core -np 4 /ec/res4/scratch/lux0804/lf84/paper84/build_double/lisflood /ec/res4/scratch/lux0804/lf84/paper84/alzette/noinf_dpn_4g/run.par > /ec/res4/scratch/lux0804/lf84/paper84/alzette/noinf_dpn_4g/run.log 2>&1
echo "RUN noinf_dpn_4g rc=$? wall=$(echo "$(date +%s.%N) - $t0" | bc) loop=$(awk "/^loop time/{print \$3}" /ec/res4/scratch/lux0804/lf84/paper84/alzette/noinf_dpn_4g/run.log | tail -1)"
n=0; d=0; for f in /ec/res4/scratch/lux0804/radar_scale_2026/runs/L84_R_1km_5min_noinf_dpn_1g/results/*; do b=$(basename $f); case $b in *.mass) continue;; esac; n=$((n+1)); cmp -s $f /ec/res4/scratch/lux0804/lf84/paper84/alzette/noinf_dpn_4g/results/$b || d=$((d+1)); done
echo "COMPARE 4 GPUs vs the 1-GPU paper run: $n files, $d differ"
