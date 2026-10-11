#!/bin/sh
# Lake-at-rest CUDA reference comparison on a level-5 mixed adaptive mesh.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BIN=${1:-${LISFLOOD_BIN:-"$ROOT/build/macos/lisflood"}}
FIX="$ROOT/tests_8.4/fixtures/mwdg2_cuda_lake32_l5"
if [ ! -x "$BIN" ]; then echo "Missing LISFLOOD binary: $BIN" >&2; exit 2; fi
WORK=${TMPDIR:-/tmp}/mwdg2_cuda_hill.$$
mkdir -p "$WORK"
PASSED=0
cleanup() {
  if [ "$PASSED" -eq 1 ]; then rm -rf "$WORK"; else
    echo "CUDA hill comparison artifacts retained at: $WORK" >&2
    [ ! -f "$WORK/run.log" ] || cat "$WORK/run.log" >&2
  fi
}
trap cleanup EXIT HUP INT TERM
cat > "$WORK/case.par" <<PAR
DEMfile $FIX/dem.asc
startfile $FIX/start.asc
resroot $WORK/metal
sim_time 0.2
saveint 0.1
massint 0.1
initial_tstep 0.05
fpfric 0.03
max_ref_lvl 5
epsilon 0.001
mwdg2
cuda
qoutput
limitslopes
tol_Krivo 1
output_precision 12
PAR
"$BIN" -v "$WORK/case.par" > "$WORK/run.log" 2>&1
grep -q 'Using the Metal GPU for native adaptive MWDG2' "$WORK/run.log"
grep -q 'Metal device:' "$WORK/run.log"
grep -q 'Metal MWDG2 native adaptive leaves: 16 / 1024' "$WORK/run.log"
grep -q 'Finished native adaptive MWDG2: leaves=16,' "$WORK/run.log"
python3 - "$WORK" "$FIX" <<'PY'
from pathlib import Path
import sys
import numpy as np
work, fix = map(Path, sys.argv[1:])
for suffix, atol in (('.wd', 1e-6), ('.Qx', 5e-7), ('.Qy', 5e-7)):
    got = np.loadtxt(work / f'metal-0002{suffix}', skiprows=6)
    expected = np.loadtxt(fix / f'cuda-0002{suffix}', skiprows=6)
    error = np.abs(got - expected)
    rmse, maximum = float(np.sqrt(np.mean(error**2))), float(error.max())
    if not np.all(np.isfinite(got)) or maximum > atol:
        raise SystemExit(f'{suffix} differs from CUDA: RMSE={rmse:.9g}, max={maximum:.9g}, tolerance={atol:g}')
    print(f'PASS CUDA hill {suffix}: RMSE={rmse:.3g}, max={maximum:.3g} (tol {atol:g})')
print('PASS level-5 mixed-mesh lake at rest: CUDA and Metal raster outputs agree')
PY
PASSED=1
