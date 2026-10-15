#!/bin/bash
# Compare 1- and 4-GPU CUDA FV1 on the 10 m, 96 h WRF Alzette case.
# Rainfall and initial state both start at 2021-07-13 18:00 UTC.
#SBATCH --job-name=lf84_fv1_10m96h_mpi
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:ga100:4
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/fv1_multi_gpu_src/slurm-%j.out
set -euo pipefail

unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3

BASE=/ec/res4/scratch/lux0804/lf84/fv1_multi_gpu_src
CASE=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_all_models_96h
INPUTS=$CASE/inputs_18utc
START=$CASE/wrf1p3_afterda_18utc/results/6hr-0000.wd
WORK=$BASE/alzette_10m_96h_gpu_compare_optimized
BIN=$BASE/build/lisflood
mkdir -p "$WORK"
cd "$WORK"

test -s "$INPUTS/rain_wrf1p3_afterda_96h.nc"
test -s "$START"
probe_output=$(mpirun --bind-to none -np 4 "$BASE/mpi_probe")
printf '%s\n' "$probe_output"
python3 -c 'import re,sys; rows=re.findall(r"probe rank=(\d+) size=(\d+)",sys.stdin.read()); assert sorted(int(r) for r,n in rows)==[0,1,2,3] and all(int(n)==4 for r,n in rows), rows' <<< "$probe_output"

make_par() {
  local name=$1
  cat > "$name.par" <<EOF
resroot 1hr
dirroot $WORK/$name/results
DEMfile /ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/Alzette_sub_basin_10m_bridge_burn.asc
manningfile /ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/manning.n.ascii
dynamicrainfile $INPUTS/rain_wrf1p3_afterda_96h.nc
startfile $START
fv1
cuda
GPUs $ranks
sim_time 345600
initial_tstep 10
massint 600
saveint 3600
bcifile /ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/Border.bci
nodata_elevation -9999
drain_nodata
qoutput
EOF
}

run_timed() {
  local name=$1 ranks=$2
  mkdir -p "$WORK/$name/results"
  make_par "$name"
  date -u '+start_utc=%Y-%m-%dT%H:%M:%SZ' > "$name.timing"
  /usr/bin/time -f '%e' -o "$name.wall_seconds" \
    mpirun --bind-to none -np "$ranks" "$BIN" "$name.par" > "$name.log" 2>&1
  grep -E 'CUDA FV1 MPI rank|loop time' "$name.log" | tee -a "$name.timing"
  printf 'wall_seconds=%s\n' "$(cat "$name.wall_seconds")" | tee -a "$name.timing"
  python3 - "$name" <<'PY'
from pathlib import Path
import sys
name=sys.argv[1]
root=Path(name)/'results'
rows=(root/'1hr.mass').read_text().splitlines()
times=[float(x.split()[0]) for x in rows[1:] if x.split()]
depths=list(root.glob('1hr-*.wd'))
assert times and max(times)>=345600, (name,max(times) if times else None)
assert len(depths)==97, (name,len(depths))
print(f'{name}: final_mass_time={max(times):.3f}s, hourly_depth_snapshots={len(depths)}')
PY
}

run_timed single_gpu 1
run_timed four_gpu 4
python3 "$BASE/tests_8.4/compare_cuda_fv1_mpi.py" \
  "$WORK/single_gpu/results" "$WORK/four_gpu/results" \
  1e-6 0.03 \
  | tee comparison.log
echo '10 m Alzette 96-hour single/four-GPU CUDA FV1 comparison completed'
