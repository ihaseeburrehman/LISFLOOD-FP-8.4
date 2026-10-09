#!/usr/bin/env python3
"""CUDA adaptive loss checks. Run on a GPU node:

    python3 adaptive.py BIN84 BIN83 WORK

Uses new output folders only. Checks closed-basin analytic losses, file versus
uniform parameters, changing meshes, conservation, repeatability, and 8.3
feature-off equality. No third-party Python packages are required.
"""
import csv
import math
from pathlib import Path
import subprocess
import sys


N = 32  # Tree traversal requires at least one full block (N*N/4 >= 256).

def raster(path, rows):
    path.write_text(f'ncols {len(rows[0])}\nnrows {len(rows)}\nxllcorner 0\nyllcorner 0\ncellsize 270\nNODATA_value -9999\n'
                    + '\n'.join(' '.join(map(str, row)) for row in rows) + '\n')


def values(path):
    return [float(x) for x in ' '.join(path.read_text().splitlines()[6:]).split()]


def main():
    binary, reference, work = map(lambda x: Path(x).resolve(), sys.argv[1:])
    work.mkdir(parents=True, exist_ok=True)
    raster(work/'dem.asc', [[0]*N for _ in range(N)])
    raster(work/'start.asc', [[.1]*N for _ in range(N)])
    raster(work/'bump.asc', [[.1+.05*math.sin(math.pi*x/(N-1))**2*math.sin(math.pi*y/(N-1))**2
                            for x in range(N)] for y in range(N)])
    for name, v in [('rate', 3.6), ('ks', 10), ('psi', 110), ('porosity', .43), ('theta', .2)]:
        raster(work/(name+'.asc'), [[v]*N for _ in range(N)])
    raster(work/'rect_dem.asc', [[0]*(N//2) for _ in range(N//4)])
    raster(work/'rect_start.asc', [[.1]*(N//2) for _ in range(N//4)])
    raster(work/'mask_dem.asc', [[-9999 if 12<=x<20 and 12<=y<20 else 0 for x in range(N)] for y in range(N)])
    raster(work/'mask_start.asc', [[0 if 12<=x<20 and 12<=y<20 else .1 for x in range(N)] for y in range(N)])
    # Different rates above and below the midline detect the adaptive/raster y flip.
    raster(work/'half.asc', [[3.6 if y<N//2 else 0]*N for y in range(N)])
    (work/'evap.dat').write_text('Evaporation\n2 days\n3.6 0\n3.6 10\n')
    ga = 'infiltration greenampt\ngreenampt_ks 10\ngreenampt_psi 110\ngreenampt_porosity .43\ngreenampt_theta_init .2\n'
    cases = {
        'off': '', 'moving_off': '', 'constant': 'infiltration .000001\n', 'grid': 'infilfile rate.asc\n',
        'evap': 'evaporation evap.dat\n', 'half': 'infilfile half.asc\n',
        'ga': ga, 'ga_grid': 'infiltration greenampt\ngreenampt_ks ks.asc\ngreenampt_psi psi.asc\ngreenampt_porosity porosity.asc\ngreenampt_theta_init theta.asc\n',
        'cap': ga+'greenampt_soil_depth .2\n',
        'moving_cap': ga+'greenampt_soil_depth .2\n',
        'rectangle_off': '', 'rectangle': 'infiltration .000001\n', 'mask_off': '', 'mask': 'infiltration .000001\n',
        'dry': 'infiltration .001\n',
    }
    failures = []
    def check(ok, text):
        print(('PASS ' if ok else 'FAIL ')+text, flush=True)
        if not ok: failures.append(text)

    def run(solver, name, epsilon, exe, label):
        out = work/label
        if out.exists(): raise RuntimeError('Refusing to overwrite '+str(out))
        out.mkdir()
        start = 'bump.asc' if name in ['moving_cap', 'moving_off'] else 'start.asc'
        dem = 'dem.asc'
        if name.startswith('rectangle'): dem, start = 'rect_dem.asc', 'rect_start.asc'
        if name.startswith('mask'): dem, start = 'mask_dem.asc', 'mask_start.asc'
        par = work/(label+'.par')
        par.write_text(f'DEMfile {dem}\nstartfile {start}\nresroot r\ndirroot {label}\n'
                       f'{solver}\ncuda\nmax_ref_lvl 5\nepsilon {epsilon}\nwall_height 25\n'
                       'initial_tstep 60\nsim_time 86400\nsaveint 21600\nmassint 3600\n'
                       'fpfric .06\nraster_out\ncumulative\nqoutput\nelevoff\n'+cases[name])
        if name=='half':
            par.write_text(par.read_text().replace('sim_time 86400', 'sim_time 3600').replace('saveint 21600', 'saveint 900').replace('massint 3600', 'massint 150'))
        with (out/'run.log').open('w') as log:
            result = subprocess.run([str(exe), str(par)], cwd=work, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        check(result.returncode==0, label+f' exits successfully (rc={result.returncode})')
        grids = sorted(out.glob('r-*.wd'), key=lambda p:int(p.stem.split('-')[-1]))
        if not grids: raise RuntimeError('No grids in '+str(out))
        return out, grids

    def physical(path, name):
        data = values(path)
        if name.startswith('mask'):
            data = [v for i,v in enumerate(data) if not (12<=i%N<20 and 12<=i//N<20)]
        return data

    area = 270**2
    for solver in ['hwfv1', 'mwdg2']:
        for epsilon in [0, .001]:
            prefix = f'{solver}_{epsilon}'
            results = {}
            for name in cases:
                if solver=='mwdg2' and name.startswith(('rectangle','mask')): continue
                out, grids = run(solver, name, epsilon, binary, prefix+'_'+name)
                results[name] = (out, grids)
                final = physical(grids[-1], name)
                check(all(math.isfinite(v) and v>=-1e-7 for v in final), prefix+'_'+name+' nonnegative finite depths')
                if not cases[name]: continue
                rows = list(csv.DictReader((out/'r.loss').open()))
                initial = sum(physical(grids[0], name))*area + float(rows[0]['loss_volume'])
                error = max(abs(float(r['water_volume'])+float(r['loss_volume'])-initial) for r in rows)
                if name not in ['moving_cap', 'rectangle', 'mask']:
                    check(error/initial < 3e-4, prefix+'_'+name+f' closed volume balance ({error/initial:.3g})')
                else:
                    baseline = 'moving_off' if name=='moving_cap' else name+'_off'
                    before = physical(results[baseline][1][0], baseline)
                    initial_off = sum(before)*area
                    reference_drift = max(abs(sum(physical(g, baseline))*area-initial_off) for g in results[baseline][1])/initial_off
                    check(error/initial <= 1.5*reference_drift+3e-4, prefix+f' {name} balance vs loss-free solver (loss={error/initial:.3g}, reference={reference_drift:.3g})')
                if name in ['rectangle', 'mask']:
                    expected_loss = 1e-6*float(rows[-1]['integrated_time'])
                    check(abs(float(rows[-1]['loss_volume'])/(len(final)*area)-expected_loss)<3e-5, prefix+'_'+name+' loss excludes padding/NoData')
                if name in ['constant', 'grid', 'evap']:
                    rate = 1e-6 if name!='evap' else 3.6/86400000
                    expected = .1-rate*float(rows[-1]['integrated_time'])
                    check(max(abs(h-expected) for h in final)<3e-5, prefix+'_'+name+' analytic depth')
                if name=='cap':
                    check(max(abs(h-.054) for h in final)<3e-5, prefix+' cap stops at 46 mm')
                if name=='dry':
                    check(max(final)<=.0011, prefix+' drains to dry threshold')
                if name=='half':
                    check(sum(final[:N*N//4]) < sum(final[-N*N//4:])-.1, prefix+' raster north/south rates have correct orientation')
                if name in ['moving_cap', 'moving_off']:
                    loss = float(rows[-1]['loss_volume'])
                    check(abs(loss/(N*N*area)-.046)<3e-5, prefix+' moving mesh retains soil cap')
                    cumu = list(csv.DictReader((out/'r.cumu').open()))
                    mesh = {r['reduction'] for r in cumu}
                    if epsilon:
                        check(len(mesh)>1, prefix+' mesh changes during moving test')
            for a,b in [('constant','grid'), ('ga','ga_grid')]:
                check([p.read_bytes() for p in results[a][1]]==[p.read_bytes() for p in results[b][1]], prefix+f' {a} == {b}')
            old, old_grids = run(solver, 'off', epsilon, reference, prefix+'_off83')
            check([p.read_bytes() for p in old_grids]==[p.read_bytes() for p in results['off'][1]], prefix+' off == 8.3 depth grids')
            for suffix in ['qx','qy']:
                check([p.read_bytes() for p in sorted(old.glob('*.'+suffix))]==[p.read_bytes() for p in sorted(results['off'][0].glob('*.'+suffix))], prefix+' off == 8.3 '+suffix)
            old_moving, old_moving_grids = run(solver, 'moving_off', epsilon, reference, prefix+'_moving83')
            check([p.read_bytes() for p in old_moving_grids]==[p.read_bytes() for p in results['moving_off'][1]], prefix+' moving off == 8.3 depth grids')
            repeat, repeat_grids = run(solver, 'moving_cap', epsilon, binary, prefix+'_repeat')
            check([p.read_bytes() for p in repeat_grids]==[p.read_bytes() for p in results['moving_cap'][1]], prefix+' repeat grids identical')
    print(f'{len(failures)} failures', flush=True)
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
