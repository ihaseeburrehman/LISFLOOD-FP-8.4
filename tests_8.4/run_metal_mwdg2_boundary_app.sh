#!/bin/sh
# App regression for native adaptive MWDG2's Alzette boundary subset.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BIN=${1:?pass a built lisflood executable}
WORK=${TMPDIR:-/tmp}/lisflood_mwdg2_boundary.$$
mkdir -p "$WORK"
PASSED=0
cleanup() {
  if [ "$PASSED" -eq 1 ]; then rm -rf "$WORK"; else echo "Artifacts retained at $WORK" >&2; fi
}
trap cleanup EXIT HUP INT TERM
python3 - "$WORK" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
# A northward bed slope directs wet flow toward the open N edge.
z=[[100.0]*8 for _ in range(8)]
for y in range(8):
    for x in range(8): z[y][x]=100.0+(7-y)*0.2
# One masked cell checks that NoData remains dry under drain_nodata/rain.
z[0][0]=-9999.0
h=[[0.5]*8 for _ in range(8)]
h[0][0]=-9999.0
def asc(path, rows):
    path.write_text('ncols 8\nnrows 8\nxllcorner 0\nyllcorner 0\ncellsize 10\nNODATA_value -9999\n'+
       '\n'.join(' '.join(map(str,r)) for r in rows)+'\n')
asc(p/'dem.asc',z); asc(p/'start.asc',h)
(p/'stage.txt').write_text('1\n25 75\n')
(p/'open.bci').write_text('N 20 40 FREE\n')
base=['sim_time 1.0','saveint 0.5','massint 0.5','initial_tstep 0.02','fpfric 0',
      'max_ref_lvl 3','epsilon 0','mwdg2','metal','drain_nodata','qoutput']
for case,bci in [('closed',None),('open',p/'open.bci')]:
    lines=[f'DEMfile {p}/dem.asc',f'startfile {p}/start.asc',f'resroot {p}/{case}',
           f'stagefile {p}/stage.txt']
    if bci: lines.append(f'bcifile {bci}')
    (p/f'{case}.par').write_text('\n'.join(lines+base)+'\n')
# A separate reflecting/no-drain case verifies that raster output clamping
# cannot change MWDG2's conserved signed leaf-mean mass.
mass_base=[v for v in base if v != 'drain_nodata']
mass_base=[v.replace('sim_time 1.0','sim_time 0.5').replace('saveint 0.5','saveint 0.25').replace('massint 0.5','massint 0.25') for v in mass_base]
(p/'mass-dem.asc').write_text('ncols 8\nnrows 8\nxllcorner 0\nyllcorner 0\ncellsize 10\nNODATA_value -9999\n'+
    '\n'.join(' '.join(['100']*8) for _ in range(8))+'\n')
(p/'mass-start.asc').write_text('ncols 8\nnrows 8\nxllcorner 0\nyllcorner 0\ncellsize 10\nNODATA_value -9999\n'+
    '\n'.join(' '.join(['0.5' if x<4 else '0' for x in range(8)]) for _ in range(8))+'\n')
(p/'mass.par').write_text('\n'.join([f'DEMfile {p}/mass-dem.asc',f'startfile {p}/mass-start.asc',
    f'resroot {p}/mass',f'stagefile {p}/stage.txt']+mass_base)+'\n')
PY
"$BIN" -v "$WORK/closed.par" > "$WORK/closed.log"
"$BIN" -v "$WORK/open.par" > "$WORK/open.log"
"$BIN" -v "$WORK/mass.par" > "$WORK/mass.log"
grep -q 'reflecting boundaries' "$WORK/closed.log"
grep -q 'north FREE boundary cells 2-3' "$WORK/open.log"
python3 - "$WORK" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
def final_volume(name):
    rows=(p/f'{name}.mass').read_text().strip().splitlines()[1:]
    if len(rows)!=3: raise SystemExit(f'{name}: expected initial and two interval mass records')
    return float(rows[-1].split()[5])
final_volume('closed'); final_volume('open')
mass_rows=(p/'mass.mass').read_text().strip().splitlines()[1:]
if len(mass_rows)!=3: raise SystemExit('mass regression: expected initial and two interval records')
for row in mass_rows:
    fields=row.split()
    if abs(float(fields[10])) > 5e-2:
        raise SystemExit(f'signed mean-volume conservation failed: Verror={fields[10]}')
stage=(p/'open.stage').read_text().splitlines()
if 'Output, depths:' not in stage or not any(line.strip().startswith('0.500') for line in stage):
    raise SystemExit('stage time series header/0.5 s sample missing')
# NoData water remains zero after loop-level drain and raster projection.
frames=sorted(p.glob('open-*.wd'))
if not frames: raise SystemExit('no output depth frames')
for f in frames:
    raster=[[float(v) for v in row.split()] for row in f.read_text().splitlines()[6:]]
    if raster[0][0] != 0.0: raise SystemExit(f'NoData cell became wet in {f.name}')
print('PASS: N-FREE edge selection, stage sampling, and drain_nodata NoData-cell clearing')
PY
PASSED=1
