#!/bin/bash
# Alzette July 2021 (Table 6, Fig. 3), double precision: sbatch p_alz4.sbatch BIND RUN
#SBATCH --job-name=p_alz4
#SBATCH --gres=gpu:ga100:4
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/paper84/logs/%x-%j.out
unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3
P=/ec/res4/scratch/lux0804/lf84/paper84
cmpdir() { local n=0 d=0 m=0; for f in $1/*; do b=$(basename $f); case $b in *.mass|*.log|*.md5) continue;; esac; n=$((n+1)); if [ ! -f $2/$b ]; then m=$((m+1)); elif ! cmp -s $f $2/$b; then d=$((d+1)); fi; done; echo "COMPARE $3: $n files, $d differ, $m missing"; }
BIN=$1; R=/ec/res4/scratch/lux0804/radar_scale_2026/runs/$2; O=$P/alzette/$2; rm -rf $O; mkdir -p $O
sed -e "s#^dirroot .*#dirroot $O/results#" $R/run.par > $O/run.par
t0=$(date +%s.%N)
if [ 4 -gt 1 ]; then mpirun --map-by ppr:4:node --bind-to core -np 4 $BIN $O/run.par > $O/run.log 2>&1
else CUDA_VISIBLE_DEVICES=0 $BIN $O/run.par > $O/run.log 2>&1; fi
echo "RUN $2 rc=$? wall=$(echo "$(date +%s.%N) - $t0" | bc) loop=$(awk "/^loop time/{print \$3}" $O/run.log | tail -1)"
cmpdir $R/results $O/results $2
