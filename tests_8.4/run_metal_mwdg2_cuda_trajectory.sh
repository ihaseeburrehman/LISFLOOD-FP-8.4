#!/bin/sh
# Longer CUDA comparison on the stable 32x32 L5 dam fixture. This is a
# bounded-drift regression through 100 RK2 steps, not a long-run equivalence
# claim; exact values and tolerances are printed for every sampled checkpoint.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BIN=${1:-${LISFLOOD_BIN:-}}
if [ -z "$BIN" ]; then BIN="$ROOT/build/macos/lisflood"; fi
if [ ! -x "$BIN" ]; then echo "Missing LISFLOOD binary: $BIN" >&2; exit 2; fi
FIX="$ROOT/tests_8.4/fixtures/mwdg2_cuda_z1_l5_e001"
WORK=${TMPDIR:-/tmp}/mwdg2_cuda_trajectory.$$
mkdir -p "$WORK"
PASSED=0
cleanup() { if [ "$PASSED" -eq 1 ]; then rm -rf "$WORK"; else echo "Trajectory artifacts retained at $WORK" >&2; fi; }
trap cleanup EXIT HUP INT TERM
cat > "$WORK/case.par" <<PAR
DEMfile $FIX/dem.asc
startfile $FIX/start.asc
sim_time 0.1
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
    raise SystemExit('app did not select native Metal GPU MWDG2')
if 'Metal MWDG2 native adaptive leaves: 184 / 1024' not in log:
    raise SystemExit('unexpected initial Metal topology')
if 'Finished native adaptive MWDG2: leaves=352' not in log:
    raise SystemExit('run did not finish with the expected finite 352-leaf tree')
if not re.search(r't=0\.099000\s+dt=0\.001000',log):
    raise SystemExit('run did not advance through the final scheduled step')
snapshots=(1,2,3,10,25,50,75,100)
# Tolerances record the measured current drift rather than implying exact
# trajectory agreement. Qy is zero in the symmetric CUDA reference; Metal's
# small transverse residual reaches 1.65e-6 RMSE / 8.58e-6 max by step 100.
limits={'.wd':(7e-4,3.5e-3), '.Qx':(1.1e-2,6.1e-2), '.Qy':(1.8e-6,9e-6)}
relative_errors={}
for step in snapshots:
    fields={}
    for suffix in ('.wd','.Qx','.Qy'):
        got=np.loadtxt(work/f'metal-{step-1:04d}{suffix}',skiprows=6)
        ref=np.loadtxt(fix/f'cuda-long-step{step:03d}{suffix}',skiprows=6)
        if not np.all(np.isfinite(got)):
            raise SystemExit(f'non-finite Metal output at step {step} {suffix}')
        diff=np.abs(got-ref); rmse=float(np.sqrt(np.mean(diff**2))); maximum=float(diff.max())
        fields[suffix]=(got,ref,diff)
        rmse_tol,max_tol=limits[suffix]
        if rmse>rmse_tol or maximum>max_tol:
            raise SystemExit(f'step {step} {suffix}: RMSE={rmse:.9g}, max={maximum:.9g}, limits={rmse_tol:g}/{max_tol:g}')
        print(f'PASS bounded trajectory step {step:3d} {suffix}: RMSE={rmse:.3g}, max={maximum:.3g}')
    depth=fields['.wd']; qx=fields['.Qx']; qy=fields['.Qy']
    depth_relative=float(np.linalg.norm(depth[2])/max(np.linalg.norm(depth[1]),1e-30))
    flow_relative=float(np.sqrt(np.sum(qx[2]**2)+np.sum(qy[2]**2))/
                        max(np.sqrt(np.sum(qx[1]**2)+np.sum(qy[1]**2)),1e-30))
    if depth_relative>0.03 or flow_relative>0.03:
        raise SystemExit(f'step {step}: normalized raster L2 difference exceeds 3%: depth={100*depth_relative:.6g}%, flow={100*flow_relative:.6g}%')
    print(f'PASS 3% normalized raster L2 step {step:3d}: depth={100*depth_relative:.4g}%, flow vector={100*flow_relative:.4g}%')
    if step==100:
        a=np.loadtxt(work/'metal-0099.wd',skiprows=6)
        b=np.loadtxt(fix/'cuda-long-step100.wd',skiprows=6)
        delta=abs(float(a.sum()-b.sum()))
        if delta>0.2: raise SystemExit(f'signed raster volume delta {delta:.6g} m3 exceeds 0.2 m3')
        print(f'PASS step 100 signed raster volume delta: {delta:.6g} m3')
print('PASS 100-step / 0.1 s finite bounded-drift diagnostic; not a long-basin parity claim')
PY
PASSED=1
