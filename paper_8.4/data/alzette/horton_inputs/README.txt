Horton grids equivalent to the Green-Ampt soil of the Alzette case.

    horton_from_ga.py GA_DIR OUT_DIR [T_HOURS=24]

Same soil, same cells: fc = Ks (the long-time rate of both models), and f0, k
are fitted per cell so that Horton's ponded cumulative infiltration
    F_H(t) = fc t + (f0 - fc)/k (1 - exp(-k t))
matches Green-Ampt's,  Ks t = F - psi dtheta ln(1 + F/(psi dtheta)),
in least squares over 0..T. Both models in 8.4 follow cumulative F (not wall
time), so matching the ponded curves makes them the same soil in two forms.

In time tau = psi dtheta / Ks and depth psi dtheta the fit depends only on
X = T / tau, so it is done once on a table of X and interpolated per cell.
Ks = 0 cells get f0 = fc = 0; cells with dtheta <= 0 (saturated) f0 = fc = Ks.
Writes horton_f0_mmh.asc, horton_fc_mmh.asc (mm/h), horton_k_per_h.asc (1/h).

GA source: /ec/res4/scratch/lux0804/lf84_cases/inputs_v3b
T = 24 h
cells: 10643280 valid, 8260239 pervious
pervious means: Ks 0.731 mm/h, psi*dtheta 60.4 mm, tau 103.9 h (median)
Horton pervious: f0 mean 5.51 mm/h (median 5.32), fc mean 0.731 mm/h, k median 0.124 /h
