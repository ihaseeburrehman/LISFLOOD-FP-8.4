#!/usr/bin/env python3
"""The Monai Valley tsunami benchmark of the adaptive solvers on Metal,
compared with CUDA 8.4 at the three gauges.

    python3 tests_8.4/monai_cuda.py BIN MONAI_DIR WORK

MONAI_DIR is testing/UoS/monai of the LISFLOOD-FP distribution. The DEM must
be named monai.dem: cuda/adaptive switches on its non-reflective west inflow
by that file name, and Metal does the same. Saved CUDA gauges:
tests_8.4/fixtures/monai_cuda.
"""
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

binary, source, work = (Path(a).resolve() for a in sys.argv[1:4])
fixtures = Path(__file__).resolve().parent / "fixtures" / "monai_cuda"
work.mkdir(parents=True, exist_ok=True)
for name in ("monai.bci", "monai.bdy", "monai.dem", "monai.par", "monai.stage", "monai.start"):
    shutil.copy(source / name, work / name)


def series(path):
    lines = path.read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("Time;")) + 1
    return np.array([[float(v) for v in line.split()] for line in lines[first:] if line.strip()])


failures = 0
for solver in ("mwdg2", "hwfv1"):
    par = work / f"{solver}.par"
    text = (work / "monai.par").read_text().replace("mwdg2", solver, 1)
    par.write_text(text.replace("dirroot       results", f"dirroot       res_{solver}"))
    with (work / f"{solver}.log").open("w") as log:
        rc = subprocess.call([str(binary), "-v", par.name], cwd=work, stdout=log, stderr=subprocess.STDOUT)
    log = (work / f"{solver}.log").read_text()
    if rc or "Monai (non-reflective wave)" not in log:
        print(f"FAIL {solver}: rc={rc} or the Monai inflow was not selected")
        failures += 1
        continue
    metal, cuda = series(work / f"res_{solver}" / "res.stage"), series(fixtures / f"{solver}.stage")
    for gauge in (1, 2, 3):
        diff = metal[:, gauge] - np.interp(metal[:, 0], cuda[:, 0], cuda[:, gauge])
        rms, worst = float(np.sqrt(np.mean(diff ** 2))), float(np.abs(diff).max())
        peak = abs((metal[:, gauge].max() - metal[0, gauge]) - (cuda[:, gauge].max() - cuda[0, gauge]))
        ok = rms < 1e-3 and worst < 5e-3 and peak < 2e-3
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {solver} gauge {gauge}: rms {rms*1000:.2f} mm, "
              f"max {worst*1000:.2f} mm, peak-rise difference {peak*1000:.2f} mm against CUDA")
print("ALL PASSED" if not failures else f"{failures} FAILED")
sys.exit(1 if failures else 0)
