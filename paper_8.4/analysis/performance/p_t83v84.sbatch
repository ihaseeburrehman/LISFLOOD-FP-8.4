#!/bin/bash
# 8.3 against the final 8.4, single precision, one A100, runs alternating on the same node (Sect. 5, first paragraph)
#SBATCH --job-name=p_t83v84
#SBATCH --gres=gpu:ga100:1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/paper84/logs/%x-%j.out
unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3
export CUDA_VISIBLE_DEVICES=0
P=/ec/res4/scratch/lux0804/lf84/paper84; T=/ec/res4/scratch/lux0804/lf84/timing_83_vs_84; F=/ec/res4/scratch/lux0804/lf84/floodcastbench/cases
B83=/perm/lux0804/LISFLOOD_versions/LISFLOOD-8.3/build/lisflood; B84=$P/build/lisflood
md5sum $B83 $B84
for c in "alz_fv1 $T/alzette fv1.par" "uk_acc $F/uk_paper_closed acc_1gpu_259200s.par" "uk_fv1 $F/uk_paper_closed fv1_1gpu_259200s.par" \
         "aus_acc $F/australia_paper_closed acc_1gpu_864000s.par" "aus_fv1 $F/australia_paper_closed fv1_1gpu_864000s.par" \
         "moz_acc $F/mozambique_paper_closed acc_1gpu_518400s.par" "moz_fv1 $F/mozambique_paper_closed fv1_1gpu_518400s.par"; do
  set -- $c
  for v in 83 84; do
    B=$B83; [ $v = 84 ] && B=$B84
    O=$P/timing83v84/$1_$v; rm -rf $O; mkdir -p $O
    sed -e "s#^dirroot .*#dirroot $O#" -e "/^GPUs/d" $2/$3 > $O/run.par
    t0=$(date +%s.%N); (cd $2 && $B $O/run.par > $O/run.log 2>&1); rc=$?
    echo "RUN $1 $v rc=$rc wall=$(echo "$(date +%s.%N) - $t0" | bc)"
  done
  n=0; d=0; for f in $P/timing83v84/$1_83/*.wd $P/timing83v84/$1_83/*.max; do [ -f "$f" ] || continue; n=$((n+1)); cmp -s $f $P/timing83v84/$1_84/$(basename $f) || d=$((d+1)); done
  echo "DEPTHS $1 8.3 vs 8.4: $n files, $d differ"
done
