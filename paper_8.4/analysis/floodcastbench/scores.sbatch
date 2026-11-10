#!/bin/bash
#SBATCH --job-name=fcb_scores
#SBATCH --qos=nf
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/scores-%j.out
module load python3
cd /ec/res4/scratch/lux0804/lf84/floodcastbench
mkdir -p exact_series
S="python3 -W ignore tools/fcb_scores.py"
for s in acc fv1; do
  $S UK cases/uk_exact/${s}_1gpu_259200s rerun/uk_clip0 exact_series/uk_$s.csv &
  $S Australia cases/australia_exact/${s}_1gpu_864000s rerun/aus_clip0 exact_series/australia_$s.csv &
  $S Mozambique cases/mozambique_exact/${s}_1gpu_518400s rerun/moz_main_noclip exact_series/mozambique_$s.csv &
  $S Pakistan cases/pakistan_exact/${s}_1gpu_1209600s rerun/pak_clip0 exact_series/pakistan_$s.csv &
done
$S UK rerun/uk_clip1 exact_series/uk_gen.csv &
$S Australia rerun/aus_clip1 exact_series/australia_gen.csv &
$S Mozambique rerun/moz_main_clip exact_series/mozambique_gen.csv &
$S Pakistan rerun/pak_clip1 exact_series/pakistan_gen.csv &
$S UK rerun/uk_clip0 exact_series/uk_gen_noclip.csv &
$S Australia rerun/aus_clip0 exact_series/australia_gen_noclip.csv &
$S Mozambique rerun/moz_main_noclip exact_series/mozambique_gen_noclip.csv &
$S Pakistan rerun/pak_clip0 exact_series/pakistan_gen_noclip.csv &
wait
