# FloodCastBench set-up, rerun of the generation code and scoring (Sect. 4.1)

Scripts as run on the ECMWF HPC in `lf84/floodcastbench` (paths inside are those of that machine).

| script | purpose |
|---|---|
| `remotezip2.py` | fetch single files (inputs, reference frames) from the FloodCastBench Zenodo archive |
| `make_case_exact.py` | build a LISFLOOD-FP case (DEM, Manning, rainfall, start, par) for one event, with the crops, 480 m point-sampling and start state of the dataset's generation code (`Data_Generation_Code`, commit c27d108) |
| `run_case.sbatch`, `build_exact.sbatch` | run one case with ACC or FV1 on one A100 (single precision) |
| `fcb_rerun.py`, `rerun.sbatch` | re-run the published generation code on the published inputs, with (`CLIP 1`) and without (`CLIP 0`) its restriction of negative fluxes |
| `fcb_scores.py`, `scores.sbatch` | hourly NSE, r, CSI (1 and 5 cm), RMSE and mean depth of a run against the reference and against the unrestricted rerun; writes `data/floodcastbench/<event>_<solver>.csv` |
| `compare_exact.py` | summary table over the events (Table 5) |
