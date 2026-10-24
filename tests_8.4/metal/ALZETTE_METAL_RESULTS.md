# Alzette 100 m Metal validation

The reproducible matrix is `tests_8.4/metal/alzette_all_solvers.py`. It uses the
real 244 x 436 Alzette 100 m DEM, start water, Manning map, WRF rainfall and
`Border.bci`. Each run is six hours (`21600 s`) with one-hour mass and depth
outputs. The matrix covers all seven GPU keywords and four loss settings:
off, uniform infiltration (2 mm/h), evaporation (3 mm/day), and Green-Ampt
(`Ks=10 mm/h`, `psi=110 mm`, `porosity=.43`, `theta_init=.30`).

All 28 Metal runs completed with seven hourly depth rasters, finite values and
an active positive loss for every loss-enabled case.

| solver | off volume (m3) | uniform removed (m3) | evaporation removed (m3) | Green-Ampt removed (m3) | maximum depth (m) |
|---|---:|---:|---:|---:|---:|
| FV1 | 1,613,156 | 62,116 | 4,010 | 531,476 | 1.650 |
| FV2 | 1,609,564 | 61,844 | 3,999 | 529,027 | 1.651 |
| DG2 | 1,793,598 | 101,705 | 8,086 | 719,308 | 1.328 |
| ACC | 1,730,047 | 81,599 | 5,707 | 653,607 | 1.513 |
| ACC_NUGRID | 1,730,047 | 81,599 | 5,707 | 653,607 | 1.513 |
| HWFV1 | 1,613,156 | 62,116 | 4,010 | 531,476 | 1.650 |
| MWDG2 | 1,794,687 | 100,809 | 6,881 | 720,315 | 1.325 |

The dense compatibility aliases are deterministic: HWFV1 is byte-identical
to the Metal FV1 raster for all four settings, and ACC_NUGRID is byte-identical
to Metal ACC. MWDG2 is byte-identical to Metal DG2 when the DG2 reference is
run with MWDG2's CFL of 0.30. With the ordinary DG2 CFL of 0.33, 99.5--99.8%
of cells remain within 1 mm; the largest local difference is 0.135 m in the
Green-Ampt run. That difference is the expected accumulated time-step change,
not a loss-kernel discrepancy.

The only negative physical depths in the Metal matrix are three DG2 cells in
the loss-free run (minimum −0.000693 m) and two MWDG2 cells (minimum
−0.000740 m). Loss-enabled runs have no negative cells. These are small
second-order reconstruction undershoots; the volume calculation clips them at
zero and reports them instead of hiding them.

The local CPU ACC check agrees with Metal ACC to all cells within 1 mm for
off, uniform infiltration and evaporation; the volume differences are below
0.003%. The CUDA Alzette reference matrix was also started on an A100. FV1,
FV2, DG2, ACC and ACC_NUGRID completed their raster outputs; CUDA ACC's
evaporation case became unstable (final maximum 6.82 m and more water than
the loss-free case), while Metal ACC remained bounded. The CUDA adaptive
HWFV1 path completed but does not consume `dynamicrainfile` in the upstream
adaptive implementation and produced large padded-grid undershoots. Those
paths are therefore recorded as implementation references, not treated as
like-for-like rainfall comparisons. A direct CUDA adaptive comparison would
require first adding the missing dynamic-rain and output-mask behavior to that
upstream path.

Run the Metal matrix with:

```text
python3 tests_8.4/metal/alzette_all_solvers.py build/macos/lisflood \
  /path/to/metal_testing/cases/alzette_100m build/alzette84_all metal
```

The CSV summary is written beside the run outputs as `summary_metal.csv`.
