# CUDA adaptive losses: validation on 9 October 2026

NVIDIA A100, CUDA 12.6, GCC 8.5, single-precision flow with double-precision
Green-Ampt state. Both solvers were checked with epsilon 0 and 0.001.

- `adaptive.py`: **218 checks passed, 0 failures**.
- `adaptive_sources.py`: **5 checks passed, 0 failures**.
- Existing suite: T01, T02, T03, T030 and T031, all four uniform-grid solvers:
  **all 20 numerical output sets match CUDA 8.3**. Thirteen runs match every
  file; seven differ only in `.mass` (floating-point atomic reduction order).
- The regenerated source patch passes a dry run against the 8.3 source.

The adaptive suite checks constant and grid infiltration, evaporation, drying,
Green-Ampt numeric/grid equality and a 46 mm soil-storage cap. It also checks
north/south raster orientation, rectangular padding and NoData holes (HWFV1),
changing meshes, retained soil storage, repeatable output grids, and equality
with 8.3 without losses on both a uniform basin and moving water.

Uniform analytic depths and capped infiltration agree within 0.03 mm (the
assertion tolerance). Grid and numeric parameters produce identical depth
outputs. Padding/NoData contribute no loss; after depth carry-over was added,
the maximum relative storage-plus-loss error is 1.18e-7 for the adaptive
rectangular basin and 2.05e-5 for the adaptive NoData-hole basin.

The adaptive flow solver itself is not exactly conservative on moving meshes.
For the moving-water cap test, relative volume drift was:

| solver | epsilon | with losses | loss-free reference |
|---|---:|---:|---:|
| HWFV1 | 0 | 0.268 % | 0.365 % |
| HWFV1 | 0.001 | 0.193 % | 0.124 % |
| MWDG2 | 0 | 0.165 % | 0.0987 % |
| MWDG2 | 0.001 | 1.40 % | 1.06 % |

Those checks use 1.5 times the measured loss-free drift plus 0.03 % of initial
storage as the tolerance, to allow changed flow dynamics. They are not a claim
of exact conservation on a general adaptive mesh. The zero-loss point-source
check agrees with 8.3's final water volume within 4.34e-6 of initial storage;
its pre-existing adaptive source-balance errors (up to 0.856 %) remain.

Reproduce on a GPU node, using fresh output directories:
```
python3 tests_8.4/metal/adaptive.py BIN84 BIN83 WORK
python3 tests_8.4/metal/adaptive_sources.py BIN84 BIN83 SOURCE_WORK
```

HPC jobs: adaptive 36632956, source/mesh guard 36633384, uniform regression
36632958. Raw logs and the regression comparison are retained locally under
`build/validation_adaptive84/`; machine-specific locations are in the local
handover. Larger real-event adaptive loss tests and Metal adaptive solvers
are outside this validation.

## Uniform-grid regression fingerprints

| case | solver | run | result | files that differ |
|---|---|---|---|---|
| T01 | acc | A84 vs A83 | bit-identical |  |
| T01 | dg2 | A84 vs A83 | bit-identical |  |
| T01 | fv1 | A84 vs A83 | bit-identical |  |
| T01 | fv2 | A84 vs A83 | bit-identical |  |
| T02 | acc | A84 vs A83 | bit-identical |  |
| T02 | dg2 | A84 vs A83 | same except .mass |  |
| T02 | fv1 | A84 vs A83 | bit-identical |  |
| T02 | fv2 | A84 vs A83 | same except .mass |  |
| T030 | acc | A84 vs A83 | bit-identical |  |
| T030 | dg2 | A84 vs A83 | bit-identical |  |
| T030 | fv1 | A84 vs A83 | bit-identical |  |
| T030 | fv2 | A84 vs A83 | bit-identical |  |
| T031 | acc | A84 vs A83 | bit-identical |  |
| T031 | dg2 | A84 vs A83 | same except .mass |  |
| T031 | fv1 | A84 vs A83 | same except .mass |  |
| T031 | fv2 | A84 vs A83 | same except .mass |  |
| T03 | acc | A84 vs A83 | bit-identical |  |
| T03 | dg2 | A84 vs A83 | same except .mass |  |
| T03 | fv1 | A84 vs A83 | bit-identical |  |
| T03 | fv2 | A84 vs A83 | same except .mass |  |
