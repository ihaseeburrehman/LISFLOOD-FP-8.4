Hourly scores of the FloodCastBench events, set up as the dataset's generation code
(HydroPML/FloodCastBench, Data_Generation_Code, commit c27d108). Made on ECMWF in
/ec/res4/scratch/lux0804/lf84/floodcastbench with tools/fcb_scores.py.

<event>_acc, _fv1     LISFLOOD-FP 8.4 (release 3766138, single precision, one A100)
<event>_gen           the generation code re-run on the published inputs (tools/fcb_rerun.py, CLIP 1)
<event>_gen_noclip    the same without its flux clipping (CLIP 0)

Columns: hour; nse, r, csi1cm, csi5cm, rmse (m) against the FloodCastBench reference;
mean_model_m, mean_ref_m = mean depth over the compared grid; nse_noclip, csi5cm_noclip
against the generation code without flux clipping. Mozambique is compared on the
480 m reference grid (30 m fields point-sampled with inter(), as in the dataset).
