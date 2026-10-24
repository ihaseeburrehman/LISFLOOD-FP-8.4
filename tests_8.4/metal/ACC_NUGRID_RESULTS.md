# CUDA `acc_nugrid` loss results

Final validation used an NVIDIA A100, CUDA 12.6, a Release build targeting
CUDA architecture 80, and the frozen 8.3 binary as the loss-disabled
reference.

- Job 36653374: **106 checks passed, 0 failures** in a clean work directory.
- Job 36653375: all 20 uniform-grid CUDA cases matched 8.3: 14 output sets
  were byte-identical and six differed only in `.mass` atomic-sum last digits.

The standalone suite covers constant and raster infiltration, evaporation,
Green-Ampt numeric and raster parameters, the 46 mm finite soil-storage cap,
north/south raster orientation, NoData holes, rectangular inputs, drying, a
full finest-level mesh and a compressed mixed-size mesh. Closed-basin storage
plus loss differed from initial storage by at most `2.27e-16` relative in
these analytic cases.

Constant and raster forms of the same parameters produced identical results.
Repeated physical outputs were byte-identical. One repeated compressed run's
diagnostic `.loss` water-volume sum differed by `3.81e-06 m3` out of about
`3.28e10 m3` (`1.2e-16` relative), because block totals reach the double
atomic in a nondeterministic order; no depth, stage, velocity, maximum or mass
output differed.

With losses disabled, the 8.4 `acc_nugrid` output matched 8.3 except for the
existing decimal rendering of `r.max`: its largest numeric difference was
`2e-09 m`. All other physical output files were byte-identical.

Early diagnostic jobs used a build directory configured without the required
single-precision `_CONFIG=config.float.cmake`; its first configuration also
retained CMake's default CUDA architecture 61. Those double-precision results
cannot be compared with the frozen single-precision 8.3 binary. The final
jobs above use a clean build configured with `config.float.cmake`, Release
mode and `CMAKE_CUDA_ARCHITECTURES=80`, matching the frozen 8.3 build.

The implementation is validated for the closed analytic cases in this suite.
A larger real-event `acc_nugrid` loss benchmark is still desirable before
treating it as field validated.

## Loss-disabled uniform CUDA regression

| case | solver | 8.4 versus 8.3 |
|---|---|---|
| T01 | ACC, DG2, FV1, FV2 | bit-identical |
| T02 | ACC, FV1, FV2 | bit-identical |
| T02 | DG2 | same except `.mass` |
| T030 | ACC, DG2, FV1, FV2 | bit-identical |
| T031 | ACC | bit-identical |
| T031 | DG2, FV1, FV2 | same except `.mass` |
| T03 | ACC, FV1 | bit-identical |
| T03 | DG2, FV2 | same except `.mass` |

Every 8.4 run took the same number of time steps as its 8.3 reference. The
`.mass` differences are the already documented nondeterministic last digits
from floating-point atomic sums; depth and other physical outputs match.
