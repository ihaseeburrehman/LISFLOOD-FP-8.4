#!/bin/bash
# Same Alzette single-/four-GPU CUDA FV1 test as the 6 h test, extended to 96 h.
# Keep outputs separate so the earlier validation remains intact.
#SBATCH --job-name=lf84_fv1_96h
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:ga100:4
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/fv1_multi_gpu_src/slurm-%j.out
set -euo pipefail

unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3

BASE=/ec/res4/scratch/lux0804/lf84/fv1_multi_gpu_src
CASE=/ec/res4/scratch/lux0804/lf84/alzette_all_compare/case
WORK=$BASE/alzette_96h_validation
BIN=$BASE/build/lisflood
mkdir -p "$WORK"
ln -sfn "$CASE/data" "$WORK/data"
ln -sfn "$CASE/Border.bci" "$WORK/Border.bci"
cd "$WORK"

probe_output=$(mpirun --bind-to none -np 4 "$BASE/mpi_probe")
printf '%s\n' "$probe_output"
python3 -c 'import re,sys; rows=re.findall(r"probe rank=(\d+) size=(\d+)",sys.stdin.read()); assert sorted(int(r) for r,n in rows)==[0,1,2,3] and all(int(n)==4 for r,n in rows), rows' <<< "$probe_output"

make_par() {
  local run_name=$1 loss_name=$2
  cat > "$run_name.par" <<EOF
DEMfile data/dem_100m.asc
startfile data/start_100m.wd
manningfile data/manning_100m.asc
dynamicrainfile data/rain_100m.nc
bcifile Border.bci
cuda
resroot r
dirroot $run_name
fv1
sim_time 345600
initial_tstep 10
massint 3600
saveint 3600
nodata_elevation -9999
drain_nodata
qoutput
output_precision 9
EOF
  if [[ $loss_name == green_ampt ]]; then
    cat >> "$run_name.par" <<'EOF'
infiltration greenampt
greenampt_ks 10
greenampt_psi 110
greenampt_porosity 0.43
greenampt_theta_init 0.30
EOF
  fi
}

run_pair() {
  local loss_name=$1
  local one="single_${loss_name}" four="multi_${loss_name}"
  mkdir -p "$WORK/$one" "$WORK/$four"
  make_par "$one" "$loss_name"
  sed "s/^dirroot $one$/dirroot $four/" "$one.par" > "$four.par"
  /usr/bin/time -f '%e' -o "$one.time" mpirun --bind-to none -np 1 \
    "$BIN" "$one.par" > "$one.log" 2>&1
  /usr/bin/time -f '%e' -o "$four.time" mpirun --bind-to none -np 4 \
    "$BIN" "$four.par" > "$four.log" 2>&1
  python3 "$BASE/tests_8.4/compare_cuda_fv1_mpi.py" \
    "$WORK/$one" "$WORK/$four" | tee "$WORK/compare_${loss_name}.log"
  printf '%s single_gpu_wall_seconds=%s four_gpu_wall_seconds=%s\n' \
    "$loss_name" "$(cat "$one.time")" "$(cat "$four.time")" \
    | tee -a "$WORK/timings.txt"
}

run_pair no_loss
run_pair green_ampt
echo "Alzette FV1 96-hour single/multi-GPU cases passed"
