# Metal multi-rank cases (Alzette 10 m)

Run from a folder that holds the Alzette inputs in `inputs/` (DEM, Manning,
`rain_wrf1p3_24h_from_14jul18.nc`, `6hr-0004.wd`, `Border.bci`):

```bash
mpiexec-mpich-mp -n 2 lisflood metal_2gpu_12h.par
```

`*_1gpu_*` and `*_2gpu_*` pairs give the same case on one and two ranks;
`compare_*.py` compare the rasters.
