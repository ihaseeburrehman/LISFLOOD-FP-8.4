#!/opt/homebrew/Caskroom/miniconda/base/bin/python3.11
import sys
from pathlib import Path
import numpy as np

CUDA_DIR = Path("cuda_reference_12h")
METAL_DIR = Path("metal_results")

print("| Step | Time | Valid Cells | Max Abs Diff (m) | RMSE (m) | Within 1mm (%) | Status |")
print("|:---|---:|---:|---:|---:|---:|:---|")

for step in range(13):
    name = f"1hr-{step:04d}.wd"
    fc = CUDA_DIR / name
    fm = METAL_DIR / name
    if not (fc.exists() and fm.exists()):
        print(f"| {step} | {step} h | - | - | - | - | Pending |")
        continue

    max_err = 0.0
    sq_err = 0.0
    count = 0
    within_1mm = 0

    with open(fc) as fa, open(fm) as fb:
        for _ in range(6):
            fa.readline(); fb.readline()
        for la, lb in zip(fa, fb):
            va = np.fromstring(la, sep=' ')
            vb = np.fromstring(lb, sep=' ')
            mask = (va > -9000) & (vb > -9000)
            if mask.any():
                err = np.abs(va[mask] - vb[mask])
                max_err = max(max_err, float(err.max()))
                sq_err += float(np.sum(err**2))
                within_1mm += int(np.sum(err <= 0.001))
                count += int(mask.sum())

    rmse = np.sqrt(sq_err / count) if count else 0.0
    pct_1mm = 100.0 * within_1mm / count if count else 0.0
    status = "EXACT" if max_err == 0.0 else ("PASS (<1cm)" if max_err < 0.01 else "CHECK")
    print(f"| {step} | {step} h | {count} | {max_err:.6e} | {rmse:.6e} | {pct_1mm:.2f}% | {status} |")
