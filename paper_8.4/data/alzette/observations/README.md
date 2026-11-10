# Observed discharge (not included)

The hourly discharge at the four gauges of the upper Alzette (Livange, Hesperange,
Pfaffenthal and Steinsel) was provided by the Administration
de la gestion de l'eau (AGE) of Luxembourg for this study. It is not redistributed here;
request it from AGE.

The scripts expect two files in this folder:

- `obs_hourly.csv`, columns `gauge,clock,h,Q`: hour `h` of the simulation (0 is 13 July 2021 18:00 UTC) and
  discharge `Q` in m3 s-1; `clock` is `utc` (records read as UTC) or `utc1` (records in
  UTC+1 shifted to UTC, the one used in the paper). Read by
  `analysis/othermodels/alzette_models.py`.
- `obs_utc.csv`, columns `gauge,h,Q`: the `utc1` series, written by that script and read
  by `analysis/paper_figures.py` for Fig. 3.

The scores computed against the observations are in `../scores_models.csv` and
`../scores_sensitivity.csv`; the simulated hydrographs are in `../hydrographs_models.csv`.
