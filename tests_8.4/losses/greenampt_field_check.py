#!/usr/bin/env python3
"""Check the classical Green-Ampt infiltration equation against published data.

The first observation is used only to calibrate the suction parameter. The
remaining six cumulative infiltration observations are held out. This checks
the infiltration equation, not LISFLOOD routing or a rainfall-runoff model.
Data: Zheng et al. (2025), Scientific Reports 15, 1738, Table 1, TA1.
"""
import math


TIMES_MIN = (10, 22, 30, 40, 51, 60, 68)
OBSERVED_CM = (1.76, 2.79, 3.28, 3.75, 4.23, 4.66, 4.94)
KS_CM_MIN = 0.0136
DELTA_THETA = 0.46 - 0.03


def capacity(f_cm, a_cm):
    """Left side of the ponded Green-Ampt equation, in cm."""
    return f_cm - a_cm * math.log1p(f_cm / a_cm)


def solve_f(target_cm, a_cm):
    lo, hi = 0.0, max(1.0, target_cm + a_cm)
    while capacity(hi, a_cm) < target_cm:
        hi *= 2.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if capacity(mid, a_cm) < target_cm:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def main():
    # Fit A = psi * (theta_sat - theta_init) to the first observation only.
    lo, hi = 1e-9, 1e6
    target = KS_CM_MIN * TIMES_MIN[0]
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if capacity(OBSERVED_CM[0], mid) > target:
            lo = mid
        else:
            hi = mid
    a_cm = (lo + hi) / 2.0
    psi_cm = a_cm / DELTA_THETA

    predicted = [solve_f(KS_CM_MIN * t, a_cm) for t in TIMES_MIN]
    errors = [p - o for p, o in zip(predicted, OBSERVED_CM)]
    holdout_errors = errors[1:]
    holdout_obs = OBSERVED_CM[1:]
    holdout_pred = predicted[1:]
    mape = 100.0 * sum(abs(e / o) for e, o in zip(holdout_errors, holdout_obs)) / len(holdout_obs)
    max_relative = 100.0 * max(abs(e / o) for e, o in zip(holdout_errors, holdout_obs))
    rmse = math.sqrt(sum(e * e for e in holdout_errors) / len(holdout_errors))
    mean_obs = sum(holdout_obs) / len(holdout_obs)
    ss_res = sum(e * e for e in holdout_errors)
    ss_tot = sum((o - mean_obs) ** 2 for o in holdout_obs)
    r2 = 1.0 - ss_res / ss_tot

    print("time_min, observed_F_cm, predicted_F_cm, error_cm, role")
    for i, (t, obs, pred, err) in enumerate(zip(TIMES_MIN, OBSERVED_CM, predicted, errors)):
        role = "calibration" if i == 0 else "holdout"
        print(f"{t}, {obs:.4f}, {pred:.4f}, {err:+.4f}, {role}")
    print(f"calibrated_A_cm={a_cm:.6f}")
    print(f"equivalent_suction_psi_cm={psi_cm:.6f}")
    print(f"holdout_MAPE_percent={mape:.4f}")
    print(f"holdout_max_relative_error_percent={max_relative:.4f}")
    print(f"holdout_RMSE_cm={rmse:.6f}")
    print(f"holdout_R2={r2:.6f}")
    if mape > 2.0 or max_relative > 5.0:
        raise SystemExit("FAIL: holdout curve exceeds declared 2% MAPE / 5% max error gates")
    print("PASS (equation-level curve check only)")


if __name__ == "__main__":
    main()
