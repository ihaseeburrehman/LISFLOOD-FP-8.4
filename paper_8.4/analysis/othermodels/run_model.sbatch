#!/bin/bash
# SERGHEI-SWE or TRITON on the Alzette 10 m case, no infiltration, double precision:
#   sbatch --gres=gpu:ga100:NG --ntasks=NG run_model.sbatch MODEL NG SIM_TIME NAME
#   MODEL = serghei | triton, SIM_TIME in s (345600 = 96 h)
#SBATCH --job-name=om_run
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/othermodels/logs/%x-%j.out
unset LD_LIBRARY_PATH
module load prgenv/gnu gcc/8.5.0 cuda/12.6 hpcx-openmpi/2.14.0-cuda
export LD_LIBRARY_PATH=/usr/local/apps/gcc/12.2.0/lib64:/ec/res4/scratch/lux0804/othermodels/pnetcdf/lib:$LD_LIBRARY_PATH
# EXTRA_LD: e.g. the CUDA 11 runtime that the HPC-X UCX CUDA transport (libuct_cuda) needs
[ -n "${EXTRA_LD:-}" ] && export LD_LIBRARY_PATH=$EXTRA_LD:$LD_LIBRARY_PATH
MODEL=$1; NG=$2; T=$3; NAME=$4
W=/ec/res4/scratch/lux0804/othermodels; C=$W/cases/alzette10m
R=$W/runs/$NAME; rm -rf $R; mkdir -p $R/input
# one GPU per rank
cat > $R/gpu.sh <<'G'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
exec "$@"
G
chmod +x $R/gpu.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

if [ $MODEL = serghei ]; then
  for f in dem roughness hini polygon; do ln -s $C/serghei/$f.input $R/input/$f.input; done
  ln -s $C/${FORCING:-atmforcing_c.nc} $R/input/atmforcing.nc
  cat > $R/input/parameters.input <<P
simLength  : $T
cfl        : 0.5
outFreq    : ${OUTFREQ:-3600.0}
obsFreq    : 3600.0
parNx      : ${PARNX:-1}
parNy      : ${PARNY:-$NG}
outFormat  : NETCDF
BCtype     : REFLECTIVE
nScreen    : 20000
P
  cat > $R/input/sw.input <<P
initialMode : file
initialValue : 0.0
friction : manning
roughness : file
dryDepth : 0.001
P
  cat > $R/input/extbc.input <<P
bccount : 1

id : outlet
bctype : ${BCTYPE:-9}
direction : 0.0 1.0
polygon : polygon.input
P
  cat > $R/input/infiltration.input <<P
InfModel : none
P
  if [ -n "${INFDIR:-}" ]; then rm $R/input/infiltration.input; ln -s $INFDIR/infiltration.input $INFDIR/infiltrationMap.input $R/input/; fi
  cd $R; [ $MODEL = triton ] && mkdir -p output
  s=$(date +%s.%N)
  mpirun --map-by ppr:$NG:node --bind-to none -np $NG ./gpu.sh $W/build_serghei${SERGHEI_SUFFIX:-}/bin/serghei input/ output/ 1 > run.log 2>&1
  rc=$?
else
  cat > $R/input/alzette.cfg <<P
dem_filename="$C/triton/alzette.dem"
output_folder="runs/$NAME/output"
input_format=ASC
outfile_pattern="%s/%s/%s_%02d_%02d"
output_format=BIN
output_option=SEQ
num_sources=0
n_infile="$C/triton/alzette.mann"
num_runoffs=$(head -2 $C/triton/alzette.roff | tail -1 | awk -F, '{print NF-1}')
runoff_filename="$C/triton/alzette.roff"
runoff_map="$C/triton/alzette.rmap"
num_extbc=0
extbc_dir="$C/triton/"
extbc_file="$C/triton/alzette.extbc"
time_series_flag=0
print_option=huv
max_value_print_option=h
sim_start_time=0
sim_duration=$T
checkpoint_id=0
time_increment_fixed=0
time_step=0.01
print_interval=3600
print_observation=3600
it_print=20000
h_infile="$C/triton/alzette.inith"
qx_infile=""
qy_infile=""
it_count=0
courant=0.5
hextra=0.001
gpu_direct_flag=0
domain_decomposition=static
open_boundaries=0
P
  cd $R; [ $MODEL = triton ] && mkdir -p output
  s=$(date +%s.%N)
  mpirun --map-by ppr:$NG:node --bind-to none -np $NG ./gpu.sh $W/build_triton${TRITON_SUFFIX:-}/triton.exe input/alzette.cfg > run.log 2>&1
  rc=$?
fi
echo "RUN $NAME model=$MODEL gpus=$NG sim=$T rc=$rc wall=$(echo "$(date +%s.%N) - $s" | bc)"
tail -5 $R/run.log
exit $rc
