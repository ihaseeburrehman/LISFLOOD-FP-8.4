#!/usr/bin/env python3
"""Compare single-rank and multi-rank CUDA FV2 Alzette outputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from compare_cuda_fv1_mpi import compare_rasters, compare_mass

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4, 5):
        raise SystemExit(
            "usage: compare_cuda_fv2_mpi.py SINGLE_DIR MULTI_DIR "
            "[raster_tol_m [mass_relative_tol]]")
    compare_rasters(Path(sys.argv[1]), Path(sys.argv[2]),
                    float(sys.argv[3]) if len(sys.argv) >= 4 else 0.03)
    mass_tolerance = float(sys.argv[4]) if len(sys.argv) == 5 else 1e-4
    compare_mass(Path(sys.argv[1]), Path(sys.argv[2]), mass_tolerance)
    print("CUDA FV2 single/multi-GPU comparison PASSED")
