#!/usr/bin/env python3
"""Compare Metal 2GPU results against CUDA reference and 1GPU Metal (12h Alzette)."""
import sys
from pathlib import Path
import numpy as np

CUDA_DIR  = Path(__file__).parent / "cuda_reference_12h"
METAL1DIR = Path(__file__).parent / "metal_results"
METAL2DIR = Path(__file__).parent / "metal_2gpu_results"

def compare(dir_a, dir_b, label):
    print(f"\n{label}")
    print("| Step | Time | Valid Cells | Max Abs Diff (m) | RMSE (m) | Within 1mm (%) | Status |")
    print("|:---|---:|---:|---:|---:|---:|:---|")
    for step in range(13):
        name = f"1hr-{step:04d}.wd"
        fa, fb = dir_a / name, dir_b / name
        if not (fa.exists() and fb.exists()):
            print(f"| {step} | {step} h | - | - | - | - | Pending |")
            continue
        max_err = sq_err = count = within_1mm = 0
        with open(fa) as a, open(fb) as b:
            for _ in range(6):
                a.readline(); b.readline()
            for la, lb in zip(a, b):
                va = np.fromstring(la, sep=' ')
                vb = np.fromstring(lb, sep=' ')
                mask = (va > -9000) & (vb > -9000)
                if mask.any():
                    err = np.abs(va[mask] - vb[mask])
                    max_err = max(max_err, float(err.max()))
                    sq_err += float(np.sum(err**2))
                    within_1mm += int(np.sum(err <= 0.001))
                    count += int(mask.sum())
        rmse = (sq_err / count) ** 0.5 if count else 0.0
        pct = 100.0 * within_1mm / count if count else 0.0
        status = "EXACT" if max_err == 0 else ("PASS (<1cm)" if max_err < 0.01 else "CHECK")
        print(f"| {step} | {step} h | {count:,} | {max_err:.4e} | {rmse:.4e} | {pct:.2f}% | {status} |")

compare(CUDA_DIR,  METAL2DIR, "Metal 2-GPU vs CUDA reference")
compare(METAL1DIR, METAL2DIR, "Metal 2-GPU vs Metal 1-GPU")
