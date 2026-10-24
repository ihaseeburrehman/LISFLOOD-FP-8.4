#!/bin/bash
# Why is CUDA FV2 on 4 GPUs slower than on 1 GPU once the 5 m Alzette grid is
# wet? Short runs from the 12 h state of the 1-GPU timing run (job 39431763):
#   a) 1 GPU   b) 4 GPUs   c) 4 GPUs, each rank sees only its own GPU
#   d) 4 GPUs under Nsight Systems
#SBATCH --job-name=lf84_fv2_probe
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:ga100:4
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --output=/ec/res4/scratch/lux0804/lf84/timing/slurm-%j.out
set -uo pipefail

unset LD_LIBRARY_PATH
module load cuda/12.6 hpcx-openmpi/2.14.0-cuda netcdf4/4.9.3 python3

BASE=/ec/res4/scratch/lux0804/lf84/timing
REF=$BASE/fv2_5m_24h_1gpu_39431763
LF=$BASE/build_39431763/lisflood
SIM=${SIM:-120}
WORK=$BASE/fv2_probe_${SLURM_JOB_ID}
mkdir -p "$WORK"; cd "$WORK"
ln -sfn "$REF/inputs" inputs
NSYS=$(command -v nsys || ls -d "$CUDA_HOME"/nsight-systems*/bin/nsys 2>/dev/null | head -1 || true)
nvidia-smi topo -m

cat > rank_gpu.sh <<'SH'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
exec "$@"
SH
chmod +x rank_gpu.sh

par() {  # name gpus sim_time
  cat > $1.par <<PAR
resroot r
dirroot $WORK/$1
DEMfile inputs/dem.asc
manningfile inputs/manning.asc
dynamicrainfile inputs/rain.nc
startfile $REF/results/6hr-0002.wd
fv2
cuda
GPUs $2
sim_time $3
initial_tstep 0.2
massint 30
saveint 100000
nodata_elevation -9999
drain_nodata
PAR
}

run() {  # name gpus sim_time [prefix...]
  local name=$1 n=$2 t=$3; shift 3
  par $name $n $t
  local s=$(date +%s.%N)
  mpirun --map-by ppr:$n:node --bind-to core -np $n "$@" "$LF" $name.par > $name.log 2>&1
  local rc=$? e=$(date +%s.%N)
  local steps=$(awk 'END{print $4}' $name/r.mass)
  echo "$name rc=$rc wall=$(echo "$e - $s" | bc) s steps=$steps t_end=$(awk 'END{print $1}' $name/r.mass)"
}

run a_1gpu 1 $SIM
run b_4gpu 4 $SIM
run c_4gpu_visible 4 $SIM ./rank_gpu.sh
if [ -n "$NSYS" ]; then
  run d_4gpu_nsys 4 30 "$NSYS" profile -t cuda,mpi,nvtx -o "$WORK/nsys_rank%q{OMPI_COMM_WORLD_RANK}" --force-overwrite true
  for f in nsys_rank*.nsys-rep; do
    "$NSYS" stats -q -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,mpi_event_sum -f csv -o "${f%.nsys-rep}" "$f" > /dev/null 2>&1
  done
  ls nsys_rank*
else
  echo "nsys not found"
fi
