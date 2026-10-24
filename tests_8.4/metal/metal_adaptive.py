#!/usr/bin/env python3
"""Small Apple-Metal checks for all three non-uniform/adaptive dense paths.

Usage: python3 tests_8.4/metal/metal_adaptive.py build/lisflood [WORK]
"""
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def raster(path, rows):
    path.write_text(
        f"ncols {len(rows[0])}\nnrows {len(rows)}\nxllcorner 0\n"
        "yllcorner 0\ncellsize 100\nNODATA_value -9999\n"
        + "\n".join(" ".join(map(str, row)) for row in rows) + "\n"
    )


def values(path):
    return [float(x) for x in " ".join(path.read_text().splitlines()[6:]).split()]


def main():
    binary = Path(sys.argv[1]).resolve()
    root = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(
        tempfile.mkdtemp(prefix="lisflood-metal-adaptive-")
    )
    root.mkdir(parents=True, exist_ok=True)
    n = 16
    raster(root / "dem.asc", [[0.0] * n for _ in range(n)])
    raster(root / "start.asc", [
        [0.15 if 6 <= x < 10 and 6 <= y < 10 else 0.10 for x in range(n)]
        for y in range(n)
    ])
    base = (
        "DEMfile dem.asc\nstartfile start.asc\nmetal\nmax_ref_lvl 4\n"
        "epsilon .001\nwall_height 25\ninitial_tstep 2\nsim_time 120\n"
        "saveint 60\nmassint 30\nfpfric .06\nraster_out\nqoutput\n"
        "output_precision 9\ninfiltration .000001\n"
    )
    failed = []
    for solver in ("hwfv1", "mwdg2", "acc_nugrid"):
        out = root / solver
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        par = root / f"{solver}.par"
        par.write_text(base + f"{solver}\ndirroot {solver}\nresroot r\n")
        run = subprocess.run([str(binary), str(par)], cwd=root,
                             text=True, capture_output=True, timeout=60)
        grids = sorted(out.glob("*.wd"))
        final = values(grids[-1]) if grids else []
        expected_mean = 0.103125 - 0.000001 * 120
        mean = sum(final) / len(final) if final else math.nan
        ok = (run.returncode == 0 and len(grids) == 3 and final and
              all(math.isfinite(v) and v >= -1e-7 for v in final) and
              abs(mean - expected_mean) < 2e-6 and
              "dense finest-level mesh" in run.stderr)
        print(("PASS " if ok else "FAIL ") +
              f"{solver}: rc={run.returncode}, grids={len(grids)}, "
              f"range={min(final) if final else 'n/a'}..{max(final) if final else 'n/a'}, "
              f"mean error={mean - expected_mean:.3g}")
        if not ok:
            failed.append(solver)
            print(run.stdout)
            print(run.stderr)
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())
