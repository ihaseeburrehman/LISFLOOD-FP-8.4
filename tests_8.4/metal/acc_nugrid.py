#!/usr/bin/env python3
"""CUDA acc_nugrid loss checks. Run on a GPU node:

    python3 acc_nugrid.py BIN84 BIN83 WORK

Checks analytic constant-rate, grid, evaporation and Green-Ampt losses,
spatial orientation, NoData/padding, repeatability and feature-off equality.
No third-party Python packages are required.
"""
import csv
import math
from pathlib import Path
import subprocess
import sys


N = 32  # The legacy traversal needs N*N/4 >= one 256-thread block.
CELL = 27000  # Large cells allow short analytic tests to use long CFL steps.
AREA = CELL**2


def raster(path, rows):
    path.write_text(
        f'ncols {len(rows[0])}\nnrows {len(rows)}\nxllcorner 0\nyllcorner 0\n'
        f'cellsize {CELL}\nNODATA_value -9999\n'
        + '\n'.join(' '.join(map(str, row)) for row in rows) + '\n')


def main():
    binary, reference, work = map(lambda x: Path(x).resolve(), sys.argv[1:4])
    reuse = '--reuse' in sys.argv[4:]
    work.mkdir(parents=True, exist_ok=True)
    dem = [[.02 if (x+y) % 2 else 0 for x in range(N)] for y in range(N)]
    start = [[.1-dem[y][x] for x in range(N)] for y in range(N)]
    raster(work/'dem.asc', dem)
    raster(work/'dem.asc1x', [[0]*N for _ in range(N)])
    raster(work/'dem.asc1y', [[0]*N for _ in range(N)])
    raster(work/'start.asc', start)
    raster(work/'rate.asc', [[3.6]*N for _ in range(N)])
    raster(work/'north_rate.asc', [[1800 if y < N//2 else 0 for _ in range(N)]
                                   for y in range(N)])
    for name, value in [('ks', 10), ('psi', 110), ('porosity', .43), ('theta', .2)]:
        raster(work/(name+'.asc'), [[value]*N for _ in range(N)])
    mask = [[-9999 if 12 <= x < 20 and 12 <= y < 20 else dem[y][x]
             for x in range(N)] for y in range(N)]
    mask_start = [[0 if mask[y][x] == -9999 else .1-mask[y][x]
                   for x in range(N)] for y in range(N)]
    raster(work/'mask.asc', mask)
    raster(work/'mask.asc1x', [[0]*N for _ in range(N)])
    raster(work/'mask.asc1y', [[0]*N for _ in range(N)])
    raster(work/'mask_start.asc', mask_start)
    rect = [[.02 if (x+y) % 2 else 0 for x in range(16)] for y in range(8)]
    raster(work/'rect.asc', rect)
    raster(work/'rect.asc1x', [[0]*16 for _ in range(8)])
    raster(work/'rect.asc1y', [[0]*16 for _ in range(8)])
    rect_start = [[.1-rect[y][x] for x in range(16)] for y in range(8)]
    raster(work/'rect_start.asc', rect_start)
    coarse = [[0 for _ in range(N)] for _ in range(N)]
    coarse_start = [[.1-coarse[y][x] for x in range(N)] for y in range(N)]
    raster(work/'coarse.asc', coarse)
    raster(work/'coarse.asc1x', [[0]*N for _ in range(N)])
    raster(work/'coarse.asc1y', [[0]*N for _ in range(N)])
    raster(work/'coarse_start.asc', coarse_start)
    (work/'evap.dat').write_text('Evaporation\n2 days\n86.4 0\n86.4 10\n')
    (work/'stage.txt').write_text(
        f'2\n{CELL/2} {CELL/2}\n{CELL/2} {31*CELL+CELL/2}\n')

    ga = ('infiltration greenampt\ngreenampt_ks 10\ngreenampt_psi 110\n'
          'greenampt_porosity .43\ngreenampt_theta_init .2\n')
    cases = {
        'off': ('dem.asc', 'start.asc', '', 3600),
        'constant': ('dem.asc', 'start.asc', 'infiltration .000001\n', 3600),
        'grid': ('dem.asc', 'start.asc', 'infilfile rate.asc\n', 3600),
        'evap': ('dem.asc', 'start.asc', 'evaporation evap.dat\n', 3600),
        'orient': ('dem.asc', 'start.asc', 'infilfile north_rate.asc\n', 60),
        'ga': ('dem.asc', 'start.asc', ga, 3600),
        'ga_grid': ('dem.asc', 'start.asc',
                    'infiltration greenampt\ngreenampt_ks ks.asc\n'
                    'greenampt_psi psi.asc\ngreenampt_porosity porosity.asc\n'
                    'greenampt_theta_init theta.asc\n', 3600),
        'cap': ('dem.asc', 'start.asc', ga+'greenampt_soil_depth .2\n', 86400),
        'mask': ('mask.asc', 'mask_start.asc', 'infiltration .000001\n', 3600),
        'rectangle': ('rect.asc', 'rect_start.asc', 'infiltration .000001\n', 3600),
        'compressed': ('coarse.asc', 'coarse_start.asc', 'infiltration .000001\n', 3600),
        'dry': ('dem.asc', 'start.asc', 'infiltration .001\n', 600),
    }
    failures = []

    def check(ok, text):
        print(('PASS ' if ok else 'FAIL ') + text, flush=True)
        if not ok:
            failures.append(text)

    def run(name, epsilon, exe, label):
        out = work/label
        if out.exists():
            if reuse and (out/'r.mass').exists():
                check(True, label+' reuses completed outputs')
                return out
            raise RuntimeError('Refusing to overwrite '+str(out))
        out.mkdir()
        dem, start, losses, duration = cases[name]
        par = work/(label+'.par')
        par.write_text(
            f'DEMfile {dem}\nstartfile {start}\nstagefile stage.txt\nresroot r\ndirroot {label}\n'
            f'acc_nugrid\ncuda\nL 5\nepsilon {epsilon}\n'
            f'initial_tstep {min(3600, duration)}\nsim_time {duration}\nsaveint {duration}\n'
            f'massint {min(600, duration)}\n'
            'fpfric .06\nelevoff\nvoutput_stage\noutput_precision 9\n' + losses)
        with (out/'run.log').open('w') as log:
            result = subprocess.run([str(exe), str(par)], cwd=work, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=180)
        check(result.returncode == 0, label+f' exits successfully (rc={result.returncode})')
        return out

    def last_loss(out):
        rows = list(csv.DictReader((out/'r.loss').open()))
        return rows, rows[-1]

    epsilon_arg = next((arg for arg in sys.argv[4:] if not arg.startswith('--')), None)
    epsilons = [float(epsilon_arg)] if epsilon_arg is not None else [0, .001]
    for epsilon in epsilons:
        prefix = f'acc_nugrid_{epsilon:g}'
        results = {}
        for name in cases:
            out = run(name, epsilon, binary, prefix+'_'+name)
            results[name] = out
            if name == 'off':
                continue
            rows, last = last_loss(out)
            loss = float(last['loss_volume'])
            water = float(last['water_volume'])
            elapsed = float(last['integrated_time'])
            cells = 128 if name == 'rectangle' else N*N-64 if name == 'mask' else N*N
            if name == 'rectangle':
                initial = sum(map(sum, rect_start))*AREA
            elif name == 'mask':
                initial = sum(map(sum, mask_start))*AREA
            elif name == 'compressed':
                initial = sum(map(sum, coarse_start))*AREA
            else:
                initial = sum(map(sum, start))*AREA
            check(math.isfinite(loss) and math.isfinite(water) and loss >= 0 and water >= 0,
                  prefix+'_'+name+' finite nonnegative volumes')
            check(abs(water+loss-initial)/initial < 3e-5,
                  prefix+'_'+name+f' closed volume balance ({abs(water+loss-initial)/initial:.3g})')
            if name in ('constant', 'grid', 'evap', 'mask', 'rectangle', 'compressed'):
                expected = cells*AREA*1e-6*elapsed
                check(abs(loss-expected)/max(expected, 1) < 3e-5,
                      prefix+'_'+name+' analytic loss')
            if name == 'orient':
                expected = (N*N//2)*AREA*.0005*elapsed
                check(abs(loss-expected)/expected < 3e-5,
                      prefix+' north/south raster orientation')
                numeric = []
                for line in (out/'r.stage').read_text().splitlines():
                    fields = line.split()
                    try:
                        numeric.append([float(value) for value in fields])
                    except ValueError:
                        pass
                bottom, top = numeric[-1][1:3]
                check(abs(bottom-.08) < 6e-5 and abs(top-.07) < 6e-5,
                      prefix+' north/south losses reach the intended cells')
            if name == 'compressed':
                vtk = next(out.glob('r-*.vtk'))
                points = next(line for line in vtk.read_text().splitlines()
                              if line.startswith('POINTS '))
                elements = int(points.split()[1])//4
                if epsilon > 0:
                    check(0 < elements < N*N, prefix+' losses run on a compressed mesh')
                else:
                    check(elements == N*N, prefix+' zero threshold retains the finest mesh')
            if name == 'cap':
                expected = cells*AREA*.046
                check(abs(loss-expected)/expected < 3e-5,
                      prefix+' Green-Ampt soil cap is 46 mm')
            if name == 'dry':
                check(water/(cells*AREA) <= .00101, prefix+' dry case stops at threshold or zero')

        for a, b in [('constant', 'grid'), ('ga', 'ga_grid')]:
            _, ar = last_loss(results[a])
            _, br = last_loss(results[b])
            check(ar == br, prefix+f' {a} == {b}')

        old = run('off', epsilon, reference, prefix+'_off83')
        new_files = {p.name: p.read_bytes() for p in results['off'].glob('r.*')}
        old_files = {p.name: p.read_bytes() for p in old.glob('r.*')}
        exact_names = new_files.keys() - {'r.max'}
        exact = new_files.keys() == old_files.keys() and all(
            new_files[name] == old_files[name] for name in exact_names)
        def raster_values(path):
            return [float(value) for line in path.read_text().splitlines()[6:]
                    for value in line.split()]
        new_max = raster_values(results['off']/'r.max')
        old_max = raster_values(old/'r.max')
        max_error = max(abs(a-b) for a, b in zip(new_max, old_max))
        check(exact and len(new_max) == len(old_max) and max_error < 3e-9,
              prefix+f' feature off matches 8.3 numerically (max depth error {max_error:.3g} m)')

        repeat = run('cap', epsilon, binary, prefix+'_repeat')
        cap_files = {p.name: p.read_bytes() for p in results['cap'].glob('r.*')
                     if p.name != 'r.loss'}
        repeat_files = {p.name: p.read_bytes() for p in repeat.glob('r.*')
                        if p.name != 'r.loss'}
        check(cap_files == repeat_files,
              prefix+' repeat physical outputs are byte-identical')
        cap_rows, _ = last_loss(results['cap'])
        repeat_rows, _ = last_loss(repeat)
        diagnostic_error = max(
            abs(float(a[key])-float(b[key]))
            for a, b in zip(cap_rows, repeat_rows)
            for key in ('loss_volume', 'water_volume'))
        check(len(cap_rows) == len(repeat_rows)
              and all(a['time'] == b['time']
                      and a['integrated_time'] == b['integrated_time']
                      for a, b in zip(cap_rows, repeat_rows))
              and diagnostic_error <= 1e-5,
              prefix+f' repeat diagnostics agree ({diagnostic_error:.3g} m3 maximum difference)')

    guard = work/'guard'
    guard.mkdir(exist_ok=reuse)
    guard_par = work/'guard.par'
    guard_par.write_text(
        'DEMfile dem.asc\nstartfile start.asc\nstagefile stage.txt\nresroot r\ndirroot guard\nacc_nugrid\n'
        'cuda\nL 4\nepsilon 0\ninitial_tstep 60\nsim_time 60\n'
        'massint 60\nsaveint 60\nfpfric .06\nelevoff\ninfiltration .000001\n')
    result = subprocess.run([str(binary), str(guard_par)], cwd=work,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    check(result.returncode != 0 and 'L >= 5' in result.stdout,
          'small-tree loss run is rejected')

    no_velocity = work/'no_velocity'
    no_velocity.mkdir(exist_ok=reuse)
    no_velocity_par = work/'no_velocity.par'
    no_velocity_par.write_text(
        'DEMfile dem.asc\nstartfile start.asc\nstagefile stage.txt\nresroot r\n'
        'dirroot no_velocity\nacc_nugrid\ncuda\nL 5\nepsilon 0\ninitial_tstep 60\n'
        'sim_time 60\nmassint 60\nsaveint 60\nfpfric .06\nelevoff\n'
        'infiltration .000001\n')
    with (no_velocity/'run.log').open('w') as log:
        result = subprocess.run([str(binary), str(no_velocity_par)], cwd=work,
                                stdout=log, stderr=subprocess.STDOUT, timeout=180)
    check(result.returncode == 0 and (no_velocity/'r.loss').exists(),
          'stage output without velocity closes files safely')

    print(f'{len(failures)} failures', flush=True)
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
