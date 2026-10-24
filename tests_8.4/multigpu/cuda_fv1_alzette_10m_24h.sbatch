#!/bin/bash
# 24-hour, 10 m Alzette CUDA FV1 run on ECMWF, with hourly output snapshots.
# Uses the same WRF-driven case inputs as the archived 8.3 benchmark.
#SBATCH --job-name=lf84_fv1_10m24h
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:ga100:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/fv1_multi_gpu_src/slurm-%j.out
set -euo pipefail

unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3

BASE=/ec/res4/scratch/lux0804/lf84/fv1_multi_gpu_src
CASE=/ec/res4/scratch/lux0804/metal_vs_cuda_10m_24h
OUT=$BASE/alzette_10m_24h_current_fv1
BIN=$BASE/build/lisflood
mkdir -p "$OUT"
cd "$OUT"

cat > alzette_10m_24h_fv1.par <<EOF
resroot 1hr
dirroot $OUT/results
DEMfile /ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/Alzette_sub_basin_10m_bridge_burn.asc
manningfile /ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/manning.n.ascii
dynamicrainfile $CASE/rain_wrf1p3_24h_from_14jul18.nc
startfile /ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_all_models_96h/wrf1p3_afterda_18utc/results/6hr-0004.wd
fv1
cuda
sim_time 86400
initial_tstep 10
massint 600
saveint 3600
bcifile /ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/Border.bci
nodata_elevation -9999
drain_nodata
qoutput
EOF

date -u '+run_start_utc=%Y-%m-%dT%H:%M:%SZ' | tee timing.txt
/usr/bin/time -f 'wall_seconds=%e' -o walltime.txt \
  "$BIN" alzette_10m_24h_fv1.par > run.log 2>&1
cat walltime.txt | tee -a timing.txt
grep -E 'CUDA|GPU enabled|loop time|Writing snapshot' run.log | tail -30 | tee -a timing.txt
python3 - <<'PY'
from pathlib import Path
import re
out=Path('results')
log=Path('run.log').read_text(errors='replace')
depths=sorted(out.glob('1hr-*.wd'))
mass_rows=(out/'1hr.mass').read_text().splitlines()
times=[float(line.split()[0]) for line in mass_rows[1:] if line.split()]
assert times and max(times) >= 86400, f"did not reach 24 h: {max(times) if times else 'no mass records'}"
assert len(depths) == 25, f"expected 25 hourly depth snapshots, found {len(depths)}"
assert re.search(r'loop time\s+[0-9.]+',log), 'missing solver loop time'
assert 'bound to GPU' in log, 'CUDA GPU selection not reported'
print(f'validation=PASS final_mass_time_seconds={max(times):.3f} hourly_depth_snapshots={len(depths)}')
PY
echo 'Alzette 10 m 24-hour CUDA FV1 run completed and validated'
