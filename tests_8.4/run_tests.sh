#!/bin/sh
# Run the 8.4 tests on a machine with an NVIDIA GPU.
#
#   tests_8.4/run_tests.sh BIN84 BIN83 WORK [ALZETTE_PAR]
#
#   BIN84        lisflood built from this folder (CUDA)
#   BIN83        lisflood 8.3 (CUDA), for the "feature off = 8.3" checks
#   WORK         output folder
#   ALZETTE_PAR  .par of the paper's Alzette 24 h run (absolute input paths);
#                the Alzette tests run 6 h of it
#
# ONLY=t012 or ONLY=alzette runs one part.
# Then: python3 tests_8.4/check.py WORK
HERE=$(cd "$(dirname "$0")" && pwd)
BIN84=$1; BIN83=$2; WORK=$3; ALZ=$4
[ -x "$BIN84" ] && [ -x "$BIN83" ] && [ -n "$WORK" ] || { sed -n 2,13p "$0"; exit 2; }
mkdir -p "$WORK" && WORK=$(cd "$WORK" && pwd)
python3 "$HERE/make_inputs.py" "$WORK" $ALZ || exit 1

run() {  # bin dir par name
  O=$WORK/out/$4; rm -rf "$O"; mkdir -p "$O"
  (cd "$2" && "$1" -v -dirroot "$O" "$3" > "$O/run.log" 2>&1); rc=$?
  echo "RESULT $4 rc=$rc loop=$(awk '/^loop time/{print $3}' "$O/run.log" | tail -1)"
}

[ "$ONLY" = alzette ] || for p in "$WORK"/t012/*.par; do
  n=$(basename "$p" .par)
  run "$BIN84" "$WORK/t012" "$(basename "$p")" "t012_${n}_84"
done
[ "$ONLY" = alzette ] || for s in cuda_fv1 cuda_acc; do
  run "$BIN83" "$WORK/t012" off_$s.par "t012_off_${s}_83"
done

if [ -n "$ALZ" ] && [ "$ONLY" != t012 ]; then
  for p in "$WORK"/alzette/*.par; do
    n=$(basename "$p" .par)
    run "$BIN84" "$WORK/alzette" "$(basename "$p")" "alz_${n}_84"
  done
  for s in fv1 acceleration; do
    run "$BIN83" "$WORK/alzette" off_$s.par "alz_off_${s}_83"
  done
fi
