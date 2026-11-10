#!/usr/bin/env python3
"""All figures of the LISFLOOD-FP 8.4 paper, in one style, from ../data.

    python3 paper_figures.py          writes ../figures/f1_losses.pdf ... f5_roofline.pdf

Colours mean the same thing in every figure (Okabe-Ito palette, safe for colour-vision deficiency):
    black          observation, analytical solution or reference simulation
    blue           LISFLOOD-FP 8.4 (FV1; in Fig. 1 all four solvers, told apart by marker)
    orange         LISFLOOD-FP 8.4 ACC (Fig. 2)
    bluish green   HiPIMS
    vermilion      TRITON
    reddish purple SERGHEI-SWE
    light blue     one GPU, dark (LISFLOOD-FP) blue four GPUs (Figs. 4, 5)
Fill: Fig. 1 filled = CUDA, open = Metal; Fig. 5 filled = 5 m, open = 10 m.
Dashed: Fig. 2 the reference scheme without its flux restriction; Fig. 3 Horton.
Legends sit below the x-axis labels.
"""
import csv
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "..", "figures")

BLACK, BLUE, VERM, GREY, GREEN, ORANGE, SKY, PURPLE = (
    "#000000", "#0072B2", "#D55E00", "#8C8C8C", "#009E73", "#E69F00", "#56B4E9", "#CC79A7")
MARK = {"fv1": "o", "fv2": "s", "dg2": "^", "acc": "D"}
OFFSET = {"fv1": -1.5, "fv2": -0.5, "dg2": 0.5, "acc": 1.5}   # Fig. 1: solvers side by side
SOLVER_NAME = {"fv1": "FV1", "fv2": "FV2", "dg2": "DG2", "acc": "ACC"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
})
WIDTH = 7.0                      # inches, full text width


def panel_labels(axes, x=-0.02, y=1.04):
    """(a), (b), ... above the top-left corner; a panel without a title gets the label closer to its axes."""
    for ax, l in zip(np.ravel(axes), "abcdefghij"):
        ax.text(x, y if ax.get_title() else 1.0, f"({l})", transform=ax.transAxes, ha="right", va="bottom",
                fontweight="bold", fontsize=8)


def legend_below(fig, handles, ncol, y=0.0):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, y), handlelength=2.0, handleheight=1.0,
               handletextpad=0.6, columnspacing=1.8)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- Fig. 1
KS, PSI, DTH, H0_GA = 10 / 3.6e6, 0.110, 0.23, 0.1          # Green-Ampt pond test
F0, FC, KH, H0_H = 60.0, 10.0, 2.0, 0.2                      # Horton test (mm/h, 1/h, m)


def greenampt_continuous(times_s, dt=0.25):
    """Ponded depth of the continuous Green-Ampt solution at the given times (s), RK4 from t = 1 s."""
    f = lambda F: 0.0 if F >= H0_GA else KS * (1 + (PSI + H0_GA - F) * DTH / F)
    t, F, out = 1.0, math.sqrt(2 * KS * (PSI + H0_GA) * DTH), {}
    for target in sorted(times_s):
        while t < target - 1e-9:
            h = min(dt, target - t)
            k1 = f(F); k2 = f(F + h / 2 * k1); k3 = f(F + h / 2 * k2); k4 = f(F + h * k3)
            F = min(F + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), H0_GA)
            t += h
        out[target] = H0_GA - F
    return out


def output_times(backend, solver, hours):
    """A grid is written at the end of the step that reaches each hour: take its time from the .mass log."""
    p = os.path.join(DATA, "verification", f"losses_{backend}", f"t012_ga_pond_cuda_{solver}_84", "r.mass")
    t = np.loadtxt(p, skiprows=1)[:, 0] if os.path.exists(p) else np.array([])
    return [float(t[t >= 3600 * k - 1e-6].min()) if (t >= 3600 * k - 1e-6).any() else 3600.0 * k for k in hours]


def horton_exact(t_s):
    t = np.asarray(t_s) / 3600.0
    return H0_H - (FC * t + (F0 - FC) / KH * (1 - np.exp(-KH * t))) / 1000.0


def pond_depth(backend, solver, k):
    p = glob.glob(os.path.join(DATA, "verification", f"losses_{backend}", f"t012_ga_pond_cuda_{solver}_84", f"*-{k:04d}.wd"))
    return np.loadtxt(p[0], skiprows=6).mean()


def fig_losses():
    hours = list(range(0, 25))
    curve = np.linspace(1, 8 * 3600, 400)
    ga_curve = greenampt_continuous(curve)
    ax_t = np.r_[0, curve / 3600]
    fig, ax = plt.subplots(2, 2, figsize=(WIDTH, 4.4))
    ax[0, 0].plot(ax_t, [100 * H0_GA] + [100 * ga_curve[t] for t in curve], "-", color=BLACK, lw=1.0)
    for backend, filled in (("metal", False), ("cuda", True)):
        for s in MARK:
            ts = output_times(backend, s, hours[1:])
            ga = greenampt_continuous(ts)
            h = [pond_depth(backend, s, k) for k in hours[1:]]
            kw = dict(marker=MARK[s], color=BLUE, mfc=BLUE if filled else "white", mew=0.7, ls="none",
                      ms=2.6 if filled else 4.2)
            th = [t / 3600 + 0.13 * OFFSET[s] for t in ts]
            ax[0, 0].plot(th, [100 * x for x in h], **kw)
            ax[0, 1].plot(th, [1000 * (x - ga[t]) for x, t in zip(h, ts)], **kw)
    ax[0, 0].set(xlabel="Time (h)", ylabel="Ponded depth (cm)", xlim=(0, 8), ylim=(0, 10.5),
                 title="Green–Ampt")
    ax[0, 1].set(xlabel="Time (h)", ylabel="Simulated minus continuous (mm)", xlim=(0, 8),
                 title="Green–Ampt")
    ax[0, 1].axhline(0, color=BLACK, lw=0.6)

    rows = []
    for f in ("horton_cuda.csv", "horton_metal.csv"):
        p = os.path.join(DATA, "verification", "horton", f)
        if os.path.exists(p):
            rows += list(csv.DictReader(open(p)))
    ts = np.linspace(0, 7200, 300)
    ax[1, 0].plot(ts / 3600, 100 * horton_exact(ts), "-", color=BLACK, lw=1.0)
    for backend, filled in (("metal", False), ("cuda", True)):
        for s in MARK:
            r = [x for x in rows if x["backend"] == backend and x["solver"] == s and float(x["t_s"]) > 0]
            if not r:
                continue
            t = np.array([float(x["t_s"]) for x in r]); h = np.array([float(x["mean_depth_m"]) for x in r])
            kw = dict(marker=MARK[s], color=BLUE, mfc=BLUE if filled else "white", mew=0.7, ls="none",
                      ms=2.6 if filled else 4.2)
            ax[1, 0].plot(t / 3600, 100 * h, **kw)
            ax[1, 1].plot(t / 3600 + 0.012 * OFFSET[s], 1000 * (h - horton_exact(t)), **kw)
    ax[1, 0].set(xlabel="Time (h)", ylabel="Ponded depth (cm)", xlim=(0, 2), xticks=[0, 0.5, 1, 1.5, 2], title="Horton")
    ax[1, 1].set(xlabel="Time (h)", ylabel="Simulated minus exact (mm)", xlim=(0, 2), xticks=[0, 0.5, 1, 1.5, 2], title="Horton")
    ax[1, 1].axhline(0, color=BLACK, lw=0.6)
    panel_labels(ax)
    # Legend in six columns of two rows (filled column by column): the analytical
    # solution, then CUDA (filled) above Metal (open), then the four solvers.
    blank = lambda: Line2D([], [], ls="none", marker=None, label=" ")
    handles = [Line2D([], [], color=BLACK, lw=1.0, label="Analytical solution"), blank(),
               Line2D([], [], marker="o", color=GREY, mfc=GREY, ls="none", label="CUDA (filled)"),
               Line2D([], [], marker="o", color=GREY, mfc="white", mew=0.7, ms=4.2, ls="none",
                      label="Metal (open)")]
    for s in MARK:
        handles += [Line2D([], [], marker=MARK[s], color=GREY, ls="none", label=SOLVER_NAME[s]), blank()]
    fig.tight_layout(rect=(0, 0.10, 1, 1), h_pad=1.2)
    legend_below(fig, handles, ncol=6)
    save(fig, "f1_losses")


# --------------------------------------------------------------------------- Fig. 2
EVENTS = (("uk", "UK 2015, 30 m"), ("australia", "Australia 2022, 30 m"),
          ("mozambique", "Mozambique 2019, 30 m"), ("pakistan", "Pakistan 2022, 480 m"))


def series(name):
    p = os.path.join(DATA, "floodcastbench", name + ".csv")
    a = np.genfromtxt(p, delimiter=",", names=True)
    return a if a.size > 1 else None


def fig_floodcastbench():
    fig, ax = plt.subplots(2, 4, figsize=(WIDTH, 3.9))
    for j, (ev, title) in enumerate(EVENTS):
        ref = None
        for solver, col in (("acc", ORANGE), ("fv1", BLUE)):
            a = series(f"{ev}_{solver}")
            d = a["hour"] / 24
            ax[0, j].plot(d, a["mean_model_m"], "-", color=col)
            ax[1, j].plot(d, a["nse"], "-", color=col)
            ax[1, j].plot(d, a["nse_noclip"], "--", color=col, lw=0.9)
            ref = a
        g = series(f"{ev}_gen_noclip")
        ax[0, j].plot(ref["hour"] / 24, ref["mean_ref_m"], "-", color=BLACK, lw=1.0, zorder=0)
        ax[0, j].plot(g["hour"] / 24, g["mean_model_m"], "--", color=BLACK, lw=0.9, zorder=0)
        ax[0, j].set_title(title)
        ax[0, j].set_ylim(bottom=0)
        ax[1, j].set_xlabel("Time (days)")
        ax[1, j].set_ylim(-0.2, 1.02)
    ax[0, 0].set_ylabel("Mean water depth (m)")
    ax[1, 0].set_ylabel("NSE of water depth")
    panel_labels(ax)
    handles = [Line2D([], [], color=BLACK, label="FloodCastBench reference"),
               Line2D([], [], color=BLACK, ls="--", lw=0.9, label="Reference scheme without flux restriction"),
               Line2D([], [], color=ORANGE, label="LISFLOOD-FP ACC"),
               Line2D([], [], color=BLUE, label="LISFLOOD-FP FV1"),
               Line2D([], [], color=GREY, label="NSE against the reference"),
               Line2D([], [], color=GREY, ls="--", lw=0.9, label="NSE against the scheme without restriction")]
    fig.tight_layout(rect=(0, 0.11, 1, 1), w_pad=0.8, h_pad=1.0)
    legend_below(fig, handles, ncol=3)
    save(fig, "f2_floodcastbench")


# --------------------------------------------------------------------------- Fig. 3
GAUGES = ("Livange", "Hesperange", "Pfaffenthal", "Steinsel")      # upstream to downstream
AZ = os.path.join(DATA, "alzette")
MODEL_COL = {"LISFLOOD-FP": BLUE, "HiPIMS": GREEN, "TRITON": VERM, "SERGHEI-SWE": PURPLE}
LS = {"none": "-", "Green-Ampt": "-", "Horton": "--"}


def hydrographs():
    out = {}
    for r in csv.DictReader(open(os.path.join(AZ, "hydrographs_models.csv"))):
        out.setdefault((r["model"], r["infiltration"], r["gauge"]), []).append((int(r["h"]), float(r["Q"])))
    obs = {}
    for r in csv.DictReader(open(os.path.join(AZ, "observations", "obs_utc.csv"))):
        obs.setdefault(r["gauge"], []).append((int(r["h"]), float(r["Q"])))
    return out, obs


def lf_infiltration(run):
    """Hourly cumulative infiltration (10^6 m3) and its rate (m3 s-1) from a LISFLOOD-FP .mass file. The last column is
    the cumulative rain minus losses; the rain is that column of the run without infiltration (same rainfall)."""
    load = lambda r: np.loadtxt(os.path.join(AZ, "runs", r, "1hr.mass"), skiprows=1)
    a, n = load(run), load("lisflood_none_1gpu")
    h = np.arange(0, 97) * 3600.0
    F = np.interp(h, n[:, 0], n[:, -1]) - np.interp(h, a[:, 0], a[:, -1])
    return h / 3600, F / 1e6, np.r_[np.nan, np.diff(F) / 3600]


def hourly_rate(t_h, cum):
    """Hourly mean rate (m3 s-1) from a cumulative volume sampled at times t_h (h)."""
    h = np.arange(0, 97)
    c = np.interp(h, t_h, cum)
    return h, np.r_[np.nan, np.diff(c) / 3600]


def serghei_series(run):
    # rows after the first (t = 0) add the boundary columns: time, volume, 6 boundary terms, rain and infiltration flux and total
    lines = open(os.path.join(AZ, "runs", run, "output", "domainTimeSeries.out")).readlines()[1:]
    a = np.array([[float(x) for x in l.split()] for l in lines if len(l.split()) >= 12])
    return a[:, 0] / 3600, a[:, 11] / 1e6, a[:, 10], a[:, 8]       # time (h), cumulative inf, inf rate, rain rate


def fig_alzette():
    q, obs = hydrographs()
    fig = plt.figure(figsize=(WIDTH, 6.6))
    gs = fig.add_gridspec(3, 4, height_ratios=(1, 1, 0.95))
    rows = ((("LISFLOOD-FP", "none"), ("HiPIMS", "none"), ("TRITON", "none"), ("SERGHEI-SWE", "none")),
            (("LISFLOOD-FP", "Green-Ampt"), ("HiPIMS", "Green-Ampt"), ("LISFLOOD-FP", "Horton"), ("SERGHEI-SWE", "Horton")))
    axes = []
    for i, runs in enumerate(rows):
        for j, g in enumerate(GAUGES):
            ax = fig.add_subplot(gs[i, j]); axes.append(ax)
            o = sorted(obs[g])
            ax.plot([h for h, _ in o if h <= 96], [v for h, v in o if h <= 96], "o", color=BLACK, ms=1.8, zorder=5)
            for m, inf in runs:
                s = sorted(q.get((m, inf, g), []))
                if s:   # TRITON is drawn wide underneath: it coincides with SERGHEI-SWE
                    wide = m == "TRITON"
                    ax.plot([h for h, _ in s], [v for _, v in s], LS[inf], color=MODEL_COL[m], lw=2.4 if wide else 1.0,
                            zorder=1 if wide else 2)
            ax.set(xlim=(0, 96), xticks=range(0, 97, 24), ylim=(0, 280))
            if i == 0:
                ax.set_title(g)
            if j == 0:
                ax.set_ylabel(("Without infiltration\n", "With infiltration\n")[i] + "Discharge (m$^3$ s$^{-1}$)")
            if i == 1:
                ax.set_xlabel("Time (h)")
    a1 = fig.add_subplot(gs[2, 0:2]); a2 = fig.add_subplot(gs[2, 2:4]); axes += [a1, a2]
    for run, ls in (("lisflood_greenampt_4gpu", "-"), ("lisflood_horton_4gpu", "--")):
        h, F, f = lf_infiltration(run)
        a1.plot(h, F, ls, color=BLUE); a2.plot(h, f, ls, color=BLUE)
    t, F, _, _ = serghei_series("serghei_horton_1gpu")
    a1.plot(t, F, "--", color=PURPLE)
    a2.plot(*hourly_rate(np.r_[0, t], np.r_[0, F * 1e6]), "--", color=PURPLE)
    n = np.loadtxt(os.path.join(AZ, "runs", "lisflood_none_1gpu", "1hr.mass"), skiprows=1)
    a2.plot(*hourly_rate(n[:, 0] / 3600, n[:, -1]), "-", color=BLACK, lw=0.6, zorder=0)   # hourly rainfall
    a1.set(xlim=(0, 96), xticks=range(0, 97, 24), ylim=(0, 17), xlabel="Time (h)",
           yticks=range(0, 16, 5), ylabel="Cumulative infiltrated volume (10$^6$ m$^3$)")
    a2.set(xlim=(0, 96), xticks=range(0, 97, 24), ylim=(0, None), xlabel="Time (h)",
           ylabel="Basin-mean rate (m$^3$ s$^{-1}$)")
    panel_labels(axes)
    handles = [Line2D([], [], marker="o", color=BLACK, ls="none", ms=2.5, label="Observed"),
               Line2D([], [], color=BLUE, label="LISFLOOD-FP 8.4"),
               Line2D([], [], color=GREEN, label="HiPIMS"),
               Line2D([], [], color=VERM, lw=2.4, label="TRITON"),
               Line2D([], [], color=PURPLE, label="SERGHEI-SWE"),
               Line2D([], [], color=GREY, label="No infiltration (a–d), Green–Ampt (e–j)"),
               Line2D([], [], color=GREY, ls="--", label="Horton"),
               Line2D([], [], color=BLACK, lw=0.6, label="Rainfall rate over the basin (j)")]
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6, h_pad=0.9)
    legend_below(fig, handles, ncol=4)
    save(fig, "f3_alzette")


# --------------------------------------------------------------------------- Fig. 4
PERF = os.path.join(DATA, "performance")


def fig_scaling():
    rows = list(csv.DictReader(open(os.path.join(PERF, "scaling.csv"))))
    lab = [r["label"].replace("\\n", "\n") for r in rows]
    one = np.array([float(r["min_1gpu"]) for r in rows]); four = np.array([float(r["min_4gpu"]) for r in rows])
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.8), gridspec_kw={"width_ratios": (1.35, 1)})
    x = np.arange(len(rows)); w = 0.36
    a.bar(x - w / 2, one, w, color=SKY, edgecolor="none")
    a.bar(x + w / 2, four, w, color=BLUE, edgecolor="none")
    for xi, v in list(zip(x - w / 2, one)) + list(zip(x + w / 2, four)):
        a.text(xi, v * 1.04, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
    a.set_yscale("log"); a.set_ylim(1, 200)
    a.set_xticks(x, lab)
    a.set_ylabel("Wall-clock time (min)")
    sp = np.array([float(r["speedup"]) for r in rows])
    b.plot(x, sp, "o-", color=BLUE)
    eff = [r["efficiency_pct"] for r in rows]
    for xi, v, e in zip(x, sp, eff):
        b.text(xi, v + 0.12, f"{v:.2f}×\n({e} %)", ha="center", va="bottom", fontsize=6.5)
    b.axhline(4, color=BLACK, lw=0.6, ls=":")
    b.text(-0.3, 4.05, "ideal (4×)", ha="left", va="bottom", fontsize=6.5)
    b.set_xticks(x, lab)
    b.set(ylim=(1, 4.4), ylabel="Speed-up on four GPUs")
    b.set_xlim(-0.4, len(x) - 0.6)
    panel_labels((a, b))
    handles = [Patch(color=SKY, label="1 GPU"), Patch(color=BLUE, label="4 GPUs"),
               Line2D([], [], color=BLUE, marker="o", label="Speed-up (parallel efficiency)")]
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.0)
    legend_below(fig, handles, ncol=3)
    save(fig, "f4_scaling")


# --------------------------------------------------------------------------- Fig. 5
PEAK_FP64, PEAK_BW = 9700.0, 1555.0                       # A100 SXM4 40 GB
KERNELS = ("friction", "flux_x", "flux_y")
KNAME = {"friction": "Friction", "flux_x": "Flux, $x$ faces", "flux_y": "Flux, $y$ faces"}
KMARK = {"friction": "o", "flux_x": "s", "flux_y": "^"}
KSHADE = {"friction": "#BFBFBF", "flux_x": "#7F7F7F", "flux_y": "#3F3F3F"}


def fig_roofline():
    m = {}
    for r in csv.DictReader(open(os.path.join(PERF, "roofline84_metrics.csv"))):
        m[(r["case"], int(r["gpu"]), r["kernel"])] = r
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.0), gridspec_kw={"width_ratios": (1, 1.15)})
    ai = np.logspace(-1, 2, 200)
    a.plot(ai, np.minimum(PEAK_FP64, PEAK_BW * ai), "-", color=BLACK, lw=1.0)
    ridge = PEAK_FP64 / PEAK_BW
    a.axvline(ridge, color=BLACK, lw=0.6, ls=":")
    a.text(ridge * 1.08, 120, f"ridge point\n{ridge:.2f} FLOP byte$^{{-1}}$", fontsize=6.5, va="bottom")
    a.text(0.16, PEAK_BW * 0.16 * 1.35, "1555 GB s$^{-1}$", rotation=37, fontsize=6.5, va="bottom")
    a.text(20, PEAK_FP64 * 1.15, "9.7 TFLOP s$^{-1}$ FP64", fontsize=6.5, ha="center", va="bottom")
    for (case, gpu, kern), r in m.items():
        ng = 4 if "4gpu" in case else 1
        col = SKY if ng == 1 else BLUE
        fill = col if case.startswith("5m") else "white"
        a.plot(float(r["ai_dram"]), float(r["gflops"]), KMARK[kern], color=col, mfc=fill, mew=0.8,
               ms=4.5 if ng == 1 else 3.2, zorder=4 if ng == 1 else 3)
    a.set_xscale("log"); a.set_yscale("log")
    a.set(xlim=(0.1, 30), ylim=(50, 2e4), xlabel="Arithmetic intensity (FLOP byte$^{-1}$)",
          ylabel="FP64 performance (GFLOP s$^{-1}$)")
    # (b) kernel time of one step on each GPU: one GPU, then the four ranks
    groups = [("10m_1gpu", "10 m\n1 GPU"), ("10m_4gpu", "10 m\n4 GPUs"), ("5m_1gpu", "5 m\n1 GPU"), ("5m_4gpu", "5 m\n4 GPUs")]
    x, ticks, labels = 0.0, [], []
    for case, lab in groups:
        ranks = sorted({g for (c, g, _) in m if c == case})
        xs = x + np.arange(len(ranks)) * 0.8
        bottom = np.zeros(len(ranks))
        for kern in KERNELS:
            v = np.array([float(m[(case, g, kern)]["t_us"]) / 1000 for g in ranks])
            b.bar(xs, v, 0.7, bottom=bottom, color=KSHADE[kern], edgecolor=SKY if len(ranks) == 1 else BLUE, lw=0.8)
            bottom += v
        ticks.append(xs.mean()); labels.append(lab)
        x = xs[-1] + 1.4
    b.set_xticks(ticks, labels)
    b.set_ylabel("Kernel time per step (ms)")
    panel_labels((a, b))
    handles = [Line2D([], [], marker=KMARK[k], color=GREY, ls="none", label=KNAME[k] + " (a)") for k in KERNELS]
    handles += [Line2D([], [], marker="o", color=GREY, mfc="white", mew=0.8, ls="none", label="10 m (open), 5 m (filled)"),
                Patch(color=SKY, label="1 GPU"), Patch(color=BLUE, label="4 GPUs, one symbol or bar per rank")]
    handles += [Patch(facecolor=KSHADE[k], edgecolor="none", label=KNAME[k] + " (b)") for k in KERNELS]
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=2.0)
    legend_below(fig, handles, ncol=3)
    save(fig, "f5_roofline")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_losses()
    fig_floodcastbench()
    fig_alzette()
    fig_scaling()
    fig_roofline()
