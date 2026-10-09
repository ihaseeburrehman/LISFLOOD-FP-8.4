#!/usr/bin/env python3
"""Check adaptive depth carry-over with point sources and a zero loss rate.
Run on a GPU node: python3 adaptive_sources.py BIN84 BIN83 WORK (new WORK only).
"""
import csv
from pathlib import Path
import subprocess
import sys
from adaptive import N, raster, values

binary, reference, work = [Path(p).resolve() for p in sys.argv[1:]]
work.mkdir(parents=True, exist_ok=False)
raster(work/'dem.asc', [[0]*N for _ in range(N)])
raster(work/'start.asc', [[.1]*N for _ in range(N)])
(work/'source.bci').write_text('P 4455 4455 QFIX .1\n')
failed = 0
for solver in ['hwfv1', 'mwdg2']:
    for epsilon in [0, .001]:
        label = f'{solver}_{epsilon}'
        par = work/(label+'.par')
        par.write_text(f'DEMfile dem.asc\nstartfile start.asc\nbcifile source.bci\n'
                       f'dirroot {label}\nresroot r\n{solver}\ncuda\nmax_ref_lvl 5\n'
                       f'epsilon {epsilon}\nwall_height 25\ninitial_tstep 60\n'
                       'sim_time 3600\nsaveint 900\nmassint 150\nfpfric .06\n'
                       'raster_out\ncumulative\nelevoff\ninfiltration 0\n')
        with (work/(label+'.log')).open('w') as log:
            rc = subprocess.run([str(binary), par.name], cwd=work, stdout=log,
                                stderr=subprocess.STDOUT, timeout=120).returncode
        rows = list(csv.DictReader((work/label/'r.loss').open())) if rc==0 else []
        error = float('inf')
        if rows:
            initial = float(rows[0]['water_volume'])
            t0 = float(rows[0]['integrated_time'])
            error = max(abs(float(r['water_volume'])-initial-.1*270*(float(r['integrated_time'])-t0)) for r in rows)/initial
        old_label = label+'_83'
        with (work/(old_label+'.log')).open('w') as log:
            old_rc = subprocess.run([str(reference), '-dirroot', old_label, par.name], cwd=work,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=120).returncode
        old_grids = sorted((work/old_label).glob('r-*.wd'), key=lambda p:int(p.stem.split('-')[-1]))
        difference = float('inf')
        old_error = float('inf')
        if rows and old_rc==0 and old_grids:
            old_water = sum(values(old_grids[-1]))*270**2
            difference = abs(float(rows[-1]['water_volume'])-old_water)/initial
            old_error = abs(old_water-initial-.1*270*(float(rows[-1]['integrated_time'])-t0))/initial
        ok = rc==0 and old_rc==0 and difference<3e-4
        failed += not ok
        print(('PASS ' if ok else 'FAIL ')+f'{label} point-source volume agrees with 8.3 (difference={difference:.3g}; balance error new={error:.3g}, reference={old_error:.3g})', flush=True)
# A small padded mesh must fail clearly before entering legacy tree traversal.
small = work/'small.par'
small.write_text(par.read_text().replace('max_ref_lvl 5','max_ref_lvl 4').replace(f'dirroot {label}','dirroot small'))
result = subprocess.run([str(binary), small.name], cwd=work, capture_output=True, text=True, timeout=30)
ok = result.returncode!=0 and 'max_ref_lvl >= 5' in result.stderr
failed += not ok
print(('PASS ' if ok else 'FAIL ')+'unsupported padded mesh rejected clearly', flush=True)
print(f'{failed} failures', flush=True)
sys.exit(bool(failed))
