# Halandri benchmark

Single storm of 22 Feb 2013 on the 5.1 km2 Halandri catchment (Athens), 10 m grid,
with measured outlet discharge and the FLOW-R2D results of Bellos and Tsakiris (2016).
Data as distributed by V. Bellos are in `data/` (see `data/readme.txt`).

    python3 make_case.py                      # writes case/
    python3 run.py BUILD NAME fv1 cuda infiltration=greenampt ...

`run.py` writes `runs/NAME/` and scores the .mass Qout (plus 0.16 m3/s base flow)
against the measurements: NSE, peak ratio, peak timing, volume ratio.
Use FV1: CUDA ACC does not count the water drained into NoData cells in Qout.
