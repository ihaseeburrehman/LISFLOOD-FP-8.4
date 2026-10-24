#!/bin/bash
# Full-workload Nsight Compute roofline profiling for CUDA FV1.
# Profiles the same 10 m/24 h, 5 m/24 h and 5 m/96 h workloads used for
# scaling, with one and four A100 GPUs. Reports are retained; run outputs are
# temporary scratch data.
#SBATCH --job-name=lf84_ncu_roofline
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:ga100:4
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/fv1_multi_gpu_src/slurm-%j.out
set -euo pipefail

unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3

BASE=/ec/res4/scratch/lux0804/lf84/fv1_multi_gpu_src
SRC=/perm/lux0804/LISFLOOD_versions/LISFLOOD-8.4-fv1-paper-build/source
PROF_BUILD=$BASE/build_cuda116_ncu_${SLURM_JOB_ID}
BIN=$PROF_BUILD/lisflood
# Include the Slurm job ID so simultaneously submitted cases never share
# reports, logs, parameter files, or output directories.
ROOT=$BASE/ncu_roofline_full_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT/reports" "$ROOT/logs" "$ROOT/runs"
CASE_SET=${CASE_SET:-all}

wanted() {
  [[ "$CASE_SET" == all || ",$CASE_SET," == *",$1,"* ]]
}

cmake -S "$SRC" -B "$PROF_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DLISFLOOD_ENABLE_CUDA_MPI=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/apps/cuda/11.6/bin/nvcc \
  > "$ROOT/logs/build.log" 2>&1
cmake --build "$PROF_BUILD" --parallel 8 >> "$ROOT/logs/build.log" 2>&1
test -x "$BIN"

PROFILE_SECTIONS=(
  SpeedOfLight_RooflineChart
)
NCU=/usr/local/apps/cuda/11.6/nsight-compute-2022.1.1/ncu
# Profile the dominant FV1 flux/friction kernels throughout each complete
# workload.  The simulation is not shortened or spatially reduced; filtering
# only avoids replaying thousands of small setup/output kernels that do not
# represent the solver's sustained arithmetic/memory path.
KERNEL_REGEX='^(update_flow_variables_x|update_flow_variables_y|apply_friction)$'

make_par() {
  local name=$1 ranks=$2 sim=$3 dem=$4 manning=$5 rain=$6 save=$7 out=$8
  mkdir -p "$out"
  {
    echo 'resroot 6hr'
    echo "dirroot $out"
    echo "DEMfile $dem"
    echo "manningfile $manning"
    echo "dynamicrainfile $rain"
    echo fv1
    echo cuda
    echo "GPUs $ranks"
    echo "sim_time $sim"
    echo 'initial_tstep 10'
    echo 'massint 3600'
    echo "saveint $save"
    echo 'nodata_elevation -9999'
    echo drain_nodata
  } > "$ROOT/$name.par"
}

run_profile() {
  local name=$1 ranks=$2 par=$3
  # %p is Nsight Compute's target-process PID macro and is unique for both
  # direct one-rank launches and MPI-launched four-rank runs.
  local report="$ROOT/reports/${name}_pid_%p"
  local ncu=("$NCU" --target-processes all \
    --kernel-name-base function --kernel-name "regex:${KERNEL_REGEX}" \
    --launch-skip 0 --launch-count 16)
  if [ "$ranks" -eq 1 ]; then
    # Application replay is reliable for a single process and provides the
    # complete workload replay used by the single-GPU roofline measurements.
    ncu+=(--replay-mode application --app-replay-match name)
  else
    # MPI ranks have different row-partition grids. Kernel replay avoids
    # cross-rank application-replay matching while still profiling the full
    # multi-GPU simulation.
    ncu+=(--replay-mode kernel)
  fi
  for section in "${PROFILE_SECTIONS[@]}"; do
    ncu+=(--section "$section")
  done
  ncu+=(--export "$report" --force)
  echo "START $name ranks=$ranks $(date -Is)" | tee -a "$ROOT/logs/driver.log"
  if [ "$ranks" -eq 1 ]; then
    "${ncu[@]}" "$BIN" "$par" > "$ROOT/logs/${name}.log" 2>&1
  else
    # One Nsight launcher must wrap the MPI application.  Launching one ncu
    # instance per MPI rank makes each replayed rank call MPI_Init independently
    # and fails with ORTE permission/startup errors.
    "${ncu[@]}" mpirun --map-by ppr:4:node --bind-to core -np 4 \
      "$BIN" "$par" > "$ROOT/logs/${name}.log" 2>&1
  fi
  echo "END $name $(date -Is)" | tee -a "$ROOT/logs/driver.log"
}

DEM10=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/Alzette_sub_basin_10m_bridge_burn.asc
MAN10=/ec/res4/scratch/lux0804/ai_rerun_2026/lisflood_graphcast_6h_corrected/inputs/manning.n.ascii
RAIN10=/ec/res4/scratch/lux0804/metal_vs_cuda_10m_24h/rain_wrf1p3_24h_from_14jul18.nc
# 10m 96h dry-start rain: WRF d03 1.3 km, starts 2021-07-13 00:00 UTC, 18 steps 0-102h
RAIN10_96H=/ec/res4/scratch/lux0804/lf84/inputs_10m_00utc/rain_wrf1p3_d03_00utc_102h.nc
DEM5=/perm/lux0804/LISFLOOD_versions/input/alzette_5m_dry/Alzette_sub_basin_bridge_burned.asc
MAN5=/perm/lux0804/LISFLOOD_versions/input/alzette_5m_dry/manning_5m_dem_aligned.asc
RAIN5=/perm/lux0804/LISFLOOD_versions/input/alzette_5m_dry/rain_5m.nc

for ranks in 1 4; do
  name="10m_24h_${ranks}gpu"
  if wanted "$name"; then
    make_par "$name" "$ranks" 86400 "$DEM10" "$MAN10" "$RAIN10" 3600 "$ROOT/runs/${name}/results"
    run_profile "$name" "$ranks" "$ROOT/$name.par"
  fi
done

for ranks in 1 4; do
  name="5m_24h_${ranks}gpu"
  if wanted "$name"; then
    make_par "$name" "$ranks" 86400 "$DEM5" "$MAN5" "$RAIN5" 21600 "$ROOT/runs/${name}/results"
    run_profile "$name" "$ranks" "$ROOT/$name.par"
  fi
done

for ranks in 1 4; do
  name="5m_96h_${ranks}gpu"
  if wanted "$name"; then
    make_par "$name" "$ranks" 345600 "$DEM5" "$MAN5" "$RAIN5" 21600 "$ROOT/runs/${name}/results"
    run_profile "$name" "$ranks" "$ROOT/$name.par"
  fi
done

# 10m 96h dry-start: clean comparison baseline matching 5m 96h dry-start
for ranks in 1 4; do
  name="10m_96h_dry_${ranks}gpu"
  if wanted "$name"; then
    make_par "$name" "$ranks" 345600 "$DEM10" "$MAN10" "$RAIN10_96H" 86400 "$ROOT/runs/${name}/results"
    run_profile "$name" "$ranks" "$ROOT/$name.par"
  fi
done

find "$ROOT/reports" -type f -name '*.ncu-rep' -printf '%f\n' | sort > "$ROOT/report_index.txt"
du -sh "$ROOT/reports" | tee "$ROOT/report_size.txt"
echo "ROOT=$ROOT" | tee -a "$ROOT/logs/driver.log"
