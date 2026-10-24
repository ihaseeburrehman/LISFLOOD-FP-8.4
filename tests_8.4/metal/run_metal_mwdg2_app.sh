#!/bin/sh
# App-level native adaptive MWDG2 regression on a non-square raster. It checks
# that the real LISFLOOD executable chooses Metal, loads native kernels,
# advances at scheduled output intervals without tiny duplicate steps, and
# conserves a lake at rest on the valid raster area.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
BIN=${1:-${LISFLOOD_BIN:-}}
if [ -z "$BIN" ]; then
  for candidate in "$ROOT/build/macos/lisflood" "$ROOT/build/omp/lisflood" "$ROOT/build/lisflood"; do
    if [ -x "$candidate" ]; then BIN=$candidate; break; fi
  done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
  echo "No executable LISFLOOD binary found. Pass its path as the first argument or set LISFLOOD_BIN." >&2
  exit 2
fi
WORK=${TMPDIR:-/tmp}/lisflood_mwdg2_app_test.$$
mkdir -p "$WORK"
PASSED=0
cleanup() {
  if [ "$PASSED" -eq 1 ]; then
    python3 -c 'import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$WORK"
  else
    echo "App regression artifacts retained at: $WORK" >&2
    if [ -f "$WORK/run.log" ]; then cat "$WORK/run.log" >&2; fi
  fi
}
trap cleanup EXIT HUP INT TERM

python3 - "$WORK" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
def write_asc(path, rows):
    h, w = len(rows), len(rows[0])
    with path.open('w') as f:
        f.write(f'ncols {w}\nnrows {h}\nxllcorner 0\nyllcorner 0\ncellsize 10\nNODATA_value -9999\n')
        for row in rows:
            f.write(' '.join(map(str, row)) + '\n')
write_asc(p/'dem.asc', [[100.0]*5 for _ in range(3)])
write_asc(p/'start.asc', [[0.1]*5 for _ in range(3)])
(p/'case.par').write_text('\n'.join([
    f'DEMfile {p}/dem.asc', f'startfile {p}/start.asc', f'resroot {p}/out',
    'sim_time 0.02', 'saveint 0.01', 'massint 0.01', 'initial_tstep 0.01',
    'fpfric 0.03', 'max_ref_lvl 3', 'epsilon 0.1', 'mwdg2', 'cuda', '']))
PY

"$BIN" -v "$WORK/case.par" > "$WORK/run.log"
grep -q 'LISFLOOD-FP 8.4' "$WORK/run.log"
grep -q 'Maximum resolution level changed to 3' "$WORK/run.log"
grep -q 'Using the Metal GPU for native adaptive MWDG2' "$WORK/run.log"
grep -q 'Metal device:' "$WORK/run.log"
grep -q 'Native adaptive MWDG2 kernels:' "$WORK/run.log"
grep -q 'Metal MWDG2 native adaptive leaves:' "$WORK/run.log"
grep -q 'Finished native adaptive MWDG2' "$WORK/run.log"
python3 - "$WORK" <<'PY'
from pathlib import Path
import math, sys
p = Path(sys.argv[1])
rows = (p/'out.mass').read_text().strip().splitlines()[1:]
# CUDA adaptive MWDG2 always starts with a 0.001 s step (initial_tstep is not
# used): one record after that step, then one per massint.
if len(rows) != 3:
    raise SystemExit(f'expected exactly three mass records, got {len(rows)}')
parsed = [[float(x) for x in row.split()] for row in rows]
times = [row[0] for row in parsed]
steps = [row[1] for row in parsed]
volumes = [row[5] for row in parsed]
# As CUDA: no step is shortened to hit an output time; a record follows the
# first step at or past each massint multiple (0.001, 0.011) and the end.
if times != [0.001, 0.011, 0.02]:
    raise SystemExit(f'unexpected output schedule: {times}')
if any(abs(x - y) > 1e-7 for x, y in zip(steps, [0.001, 0.01, 0.009])):
    raise SystemExit(f'unexpected/tiny duplicate timestep: {steps}')
if any(not math.isclose(v, 150.0, rel_tol=0, abs_tol=2e-3) for v in volumes):
    raise SystemExit(f'lake-at-rest volume drift: {volumes}')
for name in ('out-0000.wd', 'out-0001.wd', 'out-0002.wd', 'out.max'):
    if not (p/name).is_file():
        raise SystemExit(f'missing app output {name}')
print('PASS: real LISFLOOD native adaptive MWDG2 app path, rectangular raster, scheduled outputs, lake-at-rest mass')
PY

# A localized bed step drives a genuinely mixed quadtree mesh (neither a
# single coarse leaf nor the dense finest grid). With h+z constant, compare
# all saved depths and volumes over multiple steps to check the lake-at-rest
# C-property through native adaptive regridding.
python3 - "$WORK" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])/'hill'
p.mkdir()
z=[[1.0 if x==3 and y==3 else 0.0 for x in range(8)] for y in range(8)]
def asc(path, rows):
    with path.open('w') as f:
        f.write('ncols 8\nnrows 8\nxllcorner 0\nyllcorner 0\ncellsize 10\nNODATA_value -9999\n')
        for row in rows: f.write(' '.join(f'{v:.8f}' for v in row)+'\n')
asc(p/'dem.asc', [[100+z[y][x] for x in range(8)] for y in range(8)])
asc(p/'start.asc', [[5-z[y][x] for x in range(8)] for y in range(8)])
(p/'case.par').write_text('\n'.join([
    f'DEMfile {p}/dem.asc',f'startfile {p}/start.asc',f'resroot {p}/out',
    'sim_time 0.2','saveint 0.1','massint 0.1','initial_tstep 0.05',
    'fpfric 0.03','max_ref_lvl 3','epsilon 0.001','output_precision 9',
    'mwdg2','cuda','qoutput','']))
PY
"$BIN" -v "$WORK/hill/case.par" > "$WORK/hill.log"
grep -q 'Metal device:' "$WORK/hill.log"
python3 - "$WORK/hill" <<'PY'
from pathlib import Path
import math, re, sys
p=Path(sys.argv[1])
log=(p.parent/'hill.log').read_text()
m=re.search(r'native adaptive leaves: (\d+) / 64', log)
if not m or not 1 < int(m.group(1)) < 64:
    raise SystemExit('hill case did not select a genuinely mixed adaptive mesh')
def raster(path):
    return [[float(x) for x in row.split()] for row in path.read_text().splitlines()[6:]]
files=sorted(p.glob('out-*.wd'))
if len(files)<3: raise SystemExit(f'expected multi-step depth frames, got {len(files)}')
reference=raster(files[0])
for path in files[1:]:
    frame=raster(path)
    err=max(abs(a-b) for ra,rb in zip(reference,frame) for a,b in zip(ra,rb))
    if err>1e-5: raise SystemExit(f'lake-at-rest depth changed by {err} in {path.name}')
for suffix in ('.Qx','.Qy'):
    for path in sorted(p.glob('out-*'+suffix)):
        values=raster(path)
        maximum=max(abs(value) for row in values for value in row)
        if maximum>1e-5:
            raise SystemExit(f'lake-at-rest {suffix} reached {maximum} in {path.name}')
mass=(p/'out.mass').read_text().strip().splitlines()[1:]
if len(mass)<2: raise SystemExit('hill case did not write multiple mass records')
volumes=[float(row.split()[5]) for row in mass]
if any(not math.isclose(v,31900.0,rel_tol=0,abs_tol=2e-2) for v in volumes):
    raise SystemExit(f'mixed-mesh lake-at-rest volume drift: {volumes}')
print(f'PASS: mixed-mesh lake at rest ({m.group(1)} leaves), {len(files)} frames, volume={volumes[-1]:.6f} m3')
PY
python3 "$ROOT/tests_8.4/metal/metal_mwdg2_losses_app.py" "$BIN" "$WORK"
PASSED=1
