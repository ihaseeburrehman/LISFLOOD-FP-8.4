#!/bin/sh
# Exercise the native adaptive CUDA-style Krivodonova limiter on a wet dam.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BIN=${1:?pass a built lisflood executable}
WORK=${TMPDIR:-/tmp}/lisflood_mwdg2_limiter.$$
mkdir -p "$WORK"
PASSED=0
cleanup() { if [ "$PASSED" -eq 1 ]; then rm -rf "$WORK"; else echo "Limiter artifacts retained at $WORK" >&2; fi; }
trap cleanup EXIT HUP INT TERM
python3 - "$WORK" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
for name,rows in [('dem',[[100.0]*8 for _ in range(8)]),
                  ('start',[[0.2 if x<4 else 0.8 for x in range(8)] for _ in range(8)])]:
    with (p/f'{name}.asc').open('w') as f:
        f.write('ncols 8\nnrows 8\nxllcorner 0\nyllcorner 0\ncellsize 10\nNODATA_value -9999\n')
        for row in rows: f.write(' '.join(map(str,row))+'\n')
base=[f'DEMfile {p}/dem.asc',f'startfile {p}/start.asc','sim_time 0.5',
      'saveint 0.25','massint 0.25','initial_tstep 0.25','fpfric 0',
      'max_ref_lvl 3','epsilon 0','mwdg2','metal']
(p/'plain.par').write_text('\n'.join(base+[f'resroot {p}/plain','']) )
(p/'limited.par').write_text('\n'.join(base+[f'resroot {p}/limited','limitslopes','tol_Krivo 0','']))
PY
"$BIN" -v "$WORK/plain.par" > "$WORK/plain.log"
"$BIN" -v "$WORK/limited.par" > "$WORK/limited.log"
grep -q 'Using the Metal GPU for native adaptive MWDG2' "$WORK/limited.log"
grep -q 'Adaptive MWDG2 Krivodonova threshold changed to 0' "$WORK/limited.log"
python3 - "$WORK" <<'PY'
from pathlib import Path
import math,sys
p=Path(sys.argv[1])
def read_raster(path): return [[float(x) for x in row.split()] for row in path.read_text().splitlines()[6:]]
for label in ('plain','limited'):
    rows=(p/f'{label}.mass').read_text().strip().splitlines()[1:]
    # CUDA adaptive MWDG2 always starts with a 0.001 s step (initial_tstep is
    # not used), so a record follows that step, then one per massint.
    if len(rows)!=3: raise SystemExit(f'{label}: expected three mass records')
    # Without the limiter the scheme conserves mass. With it, CUDA (and so
    # Metal) limits a leaf's own slopes but reads its neighbours unlimited, so
    # the two sides of a face see different traces and mass is not exactly
    # conserved (0.46 m3 of 3440 m3 here); allow 0.05 % of the volume.
    limit=0.05 if label=='plain' else 5e-4*3440
    for row in rows:
        fields=row.split()
        if abs(float(fields[10]))>limit: raise SystemExit(f'{label}: volume error {fields[10]}')
        if not all(math.isfinite(float(v)) for v in fields): raise SystemExit(f'{label}: nonfinite mass row')
a=read_raster(p/'plain-0001.wd'); b=read_raster(p/'limited-0001.wd')
err=max(abs(x-y) for ra,rb in zip(a,b) for x,y in zip(ra,rb))
if err <= 1e-7: raise SystemExit(f'limiter had no measurable effect on dam case: max difference {err}')
print(f'PASS: native adaptive limiter parsed/applied; closed-basin Verror within 0.05 m3 (plain) / 0.05 % (limited), frame difference={err:.9g} m')
PY
PASSED=1
