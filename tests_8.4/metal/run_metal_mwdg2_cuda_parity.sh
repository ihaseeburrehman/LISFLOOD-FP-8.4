#!/bin/sh
# Three fixed-step adaptive RK2 outputs compared against CUDA 8.4 for the
# 32x32 level-5 dam with an ungraded epsilon=0.001 tree.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
BIN=${1:-${LISFLOOD_BIN:-}}
if [ -z "$BIN" ]; then BIN="$ROOT/build/macos/lisflood"; fi
if [ ! -x "$BIN" ]; then echo "Missing LISFLOOD binary: $BIN" >&2; exit 2; fi
FIX="$ROOT/tests_8.4/metal/fixtures/mwdg2_cuda_z1_l5_e001"
WORK=${TMPDIR:-/tmp}/mwdg2_cuda_parity.$$
mkdir -p "$WORK"
PASSED=0
cleanup() { if [ "$PASSED" -eq 1 ]; then rm -rf "$WORK"; else echo "Parity artifacts retained at $WORK" >&2; fi; }
trap cleanup EXIT HUP INT TERM
cat > "$WORK/case.par" <<PAR
DEMfile $FIX/dem.asc
startfile $FIX/start.asc
sim_time 0.003
initial_tstep 0.001
saveint 0.001
massint 0.001
fpfric 0
max_ref_lvl 5
epsilon 0.001
mwdg2
cuda
qoutput
limitslopes
tol_Krivo 1
output_precision 12
resroot $WORK/metal
PAR
LISFLOOD_MWDG2_DUMP_LEAVES=1 "$BIN" -v "$WORK/case.par" > "$WORK/run.log" 2>&1
python3 - "$WORK" "$FIX" <<'PY'
from pathlib import Path
import re, sys
import numpy as np
work, fix = map(Path, sys.argv[1:])
log=(work/'run.log').read_text()
if 'Using the Metal GPU for native adaptive MWDG2' not in log or 'Metal device:' not in log:
    raise SystemExit('app did not select the native Metal GPU MWDG2 path')
m=re.search(r'MWDG2_LEAF_IDS ([0-9 ]+)',log)
if not m: raise SystemExit('native initial leaf-ID diagnostic missing')
actual=list(map(int,m.group(1).split()))
expected=list(map(int,(fix/'cuda-leaf-ids.txt').read_text().split()))
if actual != expected: raise SystemExit(f'initial active leaves differ from CUDA: {len(actual)} vs {len(expected)}')
if 'Metal MWDG2 native adaptive leaves: 184 / 1024' not in log:
    raise SystemExit('unexpected native initial adaptive leaf count')
limits={'.wd':[3e-5]*3, '.Qx':[3e-5,6e-5,1.3e-3], '.Qy':[1e-7]*3}
for step in range(1,4):
    for suffix in ('.wd','.Qx','.Qy'):
        got=np.loadtxt(work/f'metal-{step-1:04d}{suffix}',skiprows=6)
        want=np.loadtxt(fix/f'cuda-multi-step{step}{suffix}',skiprows=6)
        err=np.abs(got-want); rmse=float(np.sqrt(np.mean(err**2))); maximum=float(err.max())
        atol=limits[suffix][step-1]
        if not np.all(np.isfinite(got)) or maximum>atol:
            raise SystemExit(f'step {step} {suffix} differs from CUDA: RMSE={rmse:.9g}, max={maximum:.9g}, tolerance={atol:g}')
        print(f'PASS CUDA step {step} {suffix}: RMSE={rmse:.3g}, max={maximum:.3g} (tol {atol:g})')
print('PASS CUDA initial topology: exact 184 active leaf IDs')
print('PASS three RK2 steps / 0.003 s: native Metal adaptive output compared with CUDA')
PY
PASSED=1
