#!/bin/bash
#SBATCH --job-name=p_build
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --gres=gpu:ga100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/paper84/logs/%x-%j.out
set -eo pipefail
unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3
P=/ec/res4/scratch/lux0804/lf84/paper84
cp /perm/lux0804/LISFLOOD_versions/LISFLOOD-8.4/build.sh $P/src/
cd $P/src && PRECISION=float BUILD=$P/build JOBS=16 ./build.sh 2>&1 | tail -2
cd $P/src && PRECISION=double BUILD=$P/build_double JOBS=16 ./build.sh 2>&1 | tail -2
ls -la $P/build/lisflood $P/build_double/lisflood
