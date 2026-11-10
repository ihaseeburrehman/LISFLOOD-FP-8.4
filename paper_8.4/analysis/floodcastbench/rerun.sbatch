#!/bin/bash
# sbatch rerun.sbatch EVENT TOP BOTTOM LEFT RIGHT HOURS CLIP OUT [F]
#SBATCH --job-name=fcb_rerun
#SBATCH --qos=ng
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --account=luuni
#SBATCH --output=logs/rerun-%j.out
module load python3
cd /ec/res4/scratch/lux0804/lf84/floodcastbench
python3 -W ignore tools/fcb_rerun.py "$@"
