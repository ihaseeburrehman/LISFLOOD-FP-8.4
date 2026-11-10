# LISFLOOD-FP 8.4 paper

**LISFLOOD-FP 8.4: optimised CUDA solvers and portable GPU flood simulation with hydrological losses, validated
against observations and three other GPU models** (GMD). All numbers come from the final 8.4 build (HPC `lf84/paper84/src`, builds `build` = single and
`build_double` = double precision); results without the new keywords are identical to the release.

Structure: developments (losses incl. Horton, outflow-only FREE, Metal solvers, several GPUs) -> verification
(analytical, 8.3, Metal, GPU counts) -> validation (FloodCastBench hydraulics; Alzette July 2021 against observed
discharge and three other GPU models: observations and scoring -> rules of the comparison -> no infiltration ->
Green-Ampt -> Horton, and why SERGHEI-SWE differs -> run time) -> performance (8.3 vs 8.4, Metal vs A100, scaling, roofline)
-> limitations.

Rebuild: `./make_all.sh` (Python 3 with numpy and matplotlib, and LaTeX).

```
paper/                      LISFLOOD_8.4.tex, references.bib, Copernicus class files, PDF
figures/                    f1_losses, f2_floodcastbench, f3_alzette, f4_scaling, f5_roofline (pdf + png)
analysis/
  paper_figures.py          all five figures, one style (colour key in its docstring, Okabe-Ito)
  horton_series.py          Horton test with 5 min outputs (data/verification/horton)
  compare_runs.py           Metal against the reference (Table 4)
  othermodels/              HiPIMS/SERGHEI-SWE/TRITON comparison on the HPC: input conversion, runs, gauge
                            extraction, scoring (alzette_models.py) and checks
  floodcastbench/           FloodCastBench set-up, rerun of the generation code, scoring (README there)
  performance/              HPC jobs: Alzette runs (p_alz*), scaling (p_scale*), 8.3 vs 8.4 (p_t83v84),
                            roofline (roofline84.sbatch, roof_parse.py), build
data/
  verification/             loss tests (losses_cuda, losses_metal, horton, checks), metal_vs_reference.csv,
                            baseline_cpu_vs_cuda.csv
  floodcastbench/           hourly scores of the four events (Sect. 4.1, Table 5, Fig. 2); README there
  alzette/                  observations/, scores_models.csv (Table 6), hydrographs_models.csv (Fig. 3),
                            water_balance_models.csv, scores_sensitivity.csv (Ks x0.5, x2, ERA5-Land start),
                            runs/<model>_<infiltration>_<n>gpu (LISFLOOD-FP .mass/par/log, SERGHEI-SWE inputs and
                            series, TRITON logs), horton_inputs/ (Horton grids recipe)
  performance/              run_times_models.csv (Table 7), timing_83_vs_84.csv, metal_vs_a100.csv (Table 8), scaling.csv (Table 9, Fig. 4),
                            roofline84_metrics.csv (Table 10, Fig. 5)
```

Where the numbers come from:

| in the paper | source |
|---|---|
| Table 3 (loss tests), Sect. 3.1 | `data/verification/checks/cuda_a100.txt` (final build), `metal_m1max.txt`; Horton row `data/verification/horton` |
| Sect. 3.3, 3.6 (8.3 identity, GPU counts) | HPC `lf84/paper84/logs` (p_t83v84, p_scale*, p_alz*, om_lf1g) |
| Table 4 (Metal) | `data/verification/metal_vs_reference.csv`; per row `compare_runs.py` on the raw runs |
| Table 5, Fig. 2 (FloodCastBench) | `data/floodcastbench`; run times: HPC `lf84/paper84/logs/p_fcb-33114267.out` |
| Table 6, Fig. 3, Sect. 4.2 | `data/alzette` (from HPC `othermodels/cmp/paper_tables`, `analysis/othermodels/alzette_models.py`) |
| Table 7 | `data/performance/run_times_models.csv` (job ids inside) |
| Table 8 (Metal vs A100) | `data/performance/metal_vs_a100.csv` (tests/run_suite.sh on both machines) |
| Sect. 5 first paragraph | `data/performance/timing_83_vs_84.csv` |
| Table 9, Fig. 4 | `data/performance/scaling.csv` |
| Table 10, Fig. 5 | `data/performance/roofline84_metrics.csv` (Nsight Compute 2024.3 reports in HPC `lf84/paper84/roofline`) |

Notes. LISFLOOD-FP `.mass`: the last column is the cumulative rain minus losses. Observations are the native AGE
records (UTC+1) shifted to UTC. SERGHEI-SWE: free outflow at the outlet as critical flow (its type 9 boundary does not
drain through faces to cells without data), multi-GPU runs need the CUDA 11 runtime on the library path for the
HPC-X UCX CUDA transport; superseded runs are listed in HPC `othermodels/runs/_superseded/README.md`.

The raw runs of Table 4 (depth grids, 5.8 GB, 185 MB compressed) are in
`../LISFLOOD-Metal-Acceleration/archive/8.4/raw_metal_comparisons.tar.gz` and in ECMWF `/perm/lux0804/lisflood_archive/`.
