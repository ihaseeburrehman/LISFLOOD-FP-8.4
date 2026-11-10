#!/bin/bash
#SBATCH --job-name=fcb_bexact
#SBATCH --qos=nf
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_exact-%j.out
set -e
module load python3
cd /ec/res4/scratch/lux0804/lf84/floodcastbench
python3 -W ignore tools/make_case_exact.py ${2:-Mozambique} cases/$1
