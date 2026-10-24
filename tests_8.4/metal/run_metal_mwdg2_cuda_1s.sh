#!/bin/sh
# Compare the Metal raster at t=.998 s with CUDA job 37108018's last saved
# raster (step 997, also t=.998 s). The CUDA job completed at 1.000 s but its
# save schedule ended two nominal steps earlier. Tolerances record accumulated
# drift; this is a finite-horizon diagnostic, not exact parity.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
BIN=${1:-${LISFLOOD_BIN:-$ROOT/build/macos/lisflood}}
if [ ! -x "$BIN" ]; then echo "Missing LISFLOOD binary: $BIN" >&2; exit 2; fi
FIX="$ROOT/tests_8.4/metal/fixtures/mwdg2_cuda_z1_l5_e001"
WORK=${TMPDIR:-/tmp}/mwdg2_cuda_1s.$$
mkdir -p "$WORK"
PASSED=0
cleanup() { if [ "$PASSED" -eq 1 ]; then rm -rf "$WORK"; else echo "1s comparison artifacts retained at $WORK" >&2; fi; }
trap cleanup EXIT HUP INT TERM
cat > "$WORK/case.par" <<PAR
DEMfile $FIX/dem.asc
startfile $FIX/start.asc
sim_time 1
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
"$BIN" -v "$WORK/case.par" > "$WORK/run.log" 2>&1
python3 - "$WORK" "$FIX" <<'PY'
from pathlib import Path
import re, sys
import numpy as np
work, fix = map(Path,sys.argv[1:])
log=(work/'run.log').read_text()
if 'Using the Metal GPU for native adaptive MWDG2' not in log or 'Metal device: Apple' not in log:
    raise SystemExit('1s case did not use Apple Metal MWDG2')
# The reference comes from a double-precision CUDA build; the adaptive
# topology after 1000 steps (CUDA: 880 leaves) is reported, not required.
leaves=re.search(r'Finished native adaptive MWDG2: leaves=(\d+)',log)
if not leaves: raise SystemExit('t=.998 Metal run did not finish')
print(f'INFO final leaves: Metal {leaves.group(1)}, CUDA 880')
frames=sorted(work.glob('metal-*.wd'))
if len(frames)<1001: raise SystemExit(f'expected scheduled 1s rasters, got {len(frames)}')
# The CUDA job stops at its scheduled t=.998 frame (step 997); the Metal
# 1-second run writes that same time at index 0998, then continues to 1.000.
final='metal-0998'
if not (work/f'{final}.wd').exists(): raise SystemExit('missing Metal t=.998 checkpoint')
# Pass criterion: the 3% normalized L2 limit below. The per-field errors are
# reported only: the reference is a double-precision CUDA build, and 1000
# single-precision steps of a dam break drift by a few mm (see docs/DEVELOPMENT_LOG.md).
for suffix in ('.wd','.Qx','.Qy'):
    got=np.loadtxt(work/f'{final}{suffix}',skiprows=6)
    ref=np.loadtxt(fix/f'cuda-1s-final{suffix}',skiprows=6)
    if not np.isfinite(got).all(): raise SystemExit(f'non-finite t=.998 Metal {suffix}')
    diff=np.abs(got-ref); rmse=float(np.sqrt(np.mean(diff**2))); maximum=float(diff.max())
    print(f'INFO t=.998 CUDA comparison {suffix}: RMSE={rmse:.4g}, max={maximum:.4g}')
depth=np.loadtxt(work/f'{final}.wd',skiprows=6)
depth_ref=np.loadtxt(fix/'cuda-1s-final.wd',skiprows=6)
qx=np.loadtxt(work/f'{final}.Qx',skiprows=6)
qx_ref=np.loadtxt(fix/'cuda-1s-final.Qx',skiprows=6)
qy=np.loadtxt(work/f'{final}.Qy',skiprows=6)
qy_ref=np.loadtxt(fix/'cuda-1s-final.Qy',skiprows=6)
depth_relative=float(np.linalg.norm(depth-depth_ref)/max(np.linalg.norm(depth_ref),1e-30))
flow_relative=float(np.sqrt(np.sum((qx-qx_ref)**2)+np.sum((qy-qy_ref)**2))/
                    max(np.sqrt(np.sum(qx_ref**2)+np.sum(qy_ref**2)),1e-30))
if depth_relative>0.03 or flow_relative>0.03:
    raise SystemExit(f't=.998 exceeds 3% normalized raster L2: depth={100*depth_relative:.6g}%, flow={100*flow_relative:.6g}%')
print(f'PASS 3% normalized raster L2 at t=.998: depth={100*depth_relative:.4g}%, flow vector={100*flow_relative:.4g}%')
ref=depth_ref
delta=float(depth.sum()-ref.sum())
print(f'INFO t=.998 raster signed-volume delta: {delta:.6g} m3 of {ref.sum():.1f}; max |Qy|={np.max(np.abs(qy)):.6g} (CUDA Qy is zero)')
print('PASS matched-time CUDA comparison; accumulated field errors remain approximate, not exact parity')
PY
PASSED=1
