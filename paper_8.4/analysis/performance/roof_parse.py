"""Roofline metrics from the Nsight Compute raw CSVs written by roofline84.sbatch.
    roof_parse.py REPORT_CSV [REPORT_CSV ...]   -> table on stdout, roofline84_metrics.csv next to the first CSV
Per launch: FP64 FLOP = dadd + dmul + 2 dfma (thread instructions); arithmetic intensity against DRAM bytes.
L2 and L1 load is given as throughput in % of peak (lts__throughput, l1tex__throughput): Nsight's L2/L1 byte
counters (lts2xbar, lsu_writeback) count reads only, so they give no consistent cache-level intensity.
Averages are over the 16 sampled launches of each kernel on each GPU (process)."""
import csv, io, os, sys
from collections import defaultdict
import numpy as np

PEAK_FP64, PEAK_DRAM = 9.7e12, 1555e9          # A100-SXM4-40GB
SCALE = {"": 1, "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
         "byte/second": 1, "Kbyte/second": 1e3, "Mbyte/second": 1e6, "Gbyte/second": 1e9, "Tbyte/second": 1e12,
         "ns": 1e-9, "nsecond": 1e-9, "us": 1e-6, "usecond": 1e-6, "ms": 1e-3, "msecond": 1e-3,
         "cycle/second": 1, "cycle/nsecond": 1e9, "cycle/usecond": 1e6, "inst/cycle": 1, "inst": 1,
         "Gbyte/s": 1e9, "Mbyte/s": 1e6, "Tbyte/s": 1e12, "Ghz": 1e9, "Mhz": 1e6, "%": 1}
# the derived per-second byte metrics carry the unit of the amount (e.g. "Gbyte"); they are rates
SHORT = {"apply_friction_batched": "friction", "update_flow_variables_xm": "flux_x", "update_flow_variables_ym": "flux_y"}

def read(p):
    lines = [l for l in open(p) if l.startswith('"')]
    r = csv.reader(io.StringIO("".join(lines)))
    head, units = next(r), next(r)
    for row in r:
        d = dict(zip(head, row))
        def v(name):
            u = units[head.index(name)]
            return float(d[name].replace(",", "")) * SCALE[u]
        yield d, v

def has(d, *names):
    return all(n in d and d[n] not in ("", "n/a") for n in names)

rows = []
for p in sys.argv[1:]:
    case = os.path.basename(p).split("_pid_")[0]
    for d, v in read(p):
        kn = d["Kernel Name"].split("(")[0].strip(); k = SHORT.get(kn, kn)
        t = v("gpu__time_duration.sum")
        cps = v("smsp__cycles_elapsed.avg.per_second")
        ipc = sum(f * v(f"smsp__sass_thread_inst_executed_op_{o}_pred_on.sum.per_cycle_elapsed")
                  for o, f in (("dadd", 1), ("dmul", 1), ("dfma", 2)))
        flop = ipc * cps * t
        b = {"dram": v("dram__bytes.sum.per_second") * t}
        if has(d, "derived__lts__lts2xbar_bytes.sum.per_second"): b["l2"] = v("derived__lts__lts2xbar_bytes.sum.per_second") * t
        if has(d, "derived__l1tex__lsu_writeback_bytes_mem_lg.sum.per_second"): b["l1"] = v("derived__l1tex__lsu_writeback_bytes_mem_lg.sum.per_second") * t
        rows.append(dict(case=case, pid=d.get("Process ID", ""), kernel=k, t_us=t * 1e6, gflops=flop / t / 1e9,
                         dram_gbs=b["dram"] / t / 1e9, ai_dram=flop / b["dram"],
                         ai_l2=flop / b["l2"] if "l2" in b else np.nan, ai_l1=flop / b["l1"] if "l1" in b else np.nan,
                         l2_gbs=b.get("l2", np.nan) / t / 1e9,
                         l2_pct=v("lts__throughput.avg.pct_of_peak_sustained_elapsed"),
                         l1_pct=v("l1tex__throughput.avg.pct_of_peak_sustained_active")))

out = os.path.join(os.path.dirname(sys.argv[1]), "roofline84_metrics.csv")
g = defaultdict(list)
for r in rows: g[(r["case"], r["pid"], r["kernel"])].append(r)
keys = ["t_us", "gflops", "dram_gbs", "ai_dram", "l2_pct", "l1_pct"]
with open(out, "w") as f:
    w = csv.writer(f); w.writerow(["case", "gpu", "kernel", "n"] + keys + ["pct_dram_peak", "pct_fp64_peak"])
    print(f"{'case':10s} {'gpu':>3s} {'kernel':9s} {'n':>3s} {'t_us':>7s} {'GFLOP/s':>8s} {'%FP64':>6s} {'DRAM GB/s':>9s} "
          f"{'%DRAM':>6s} {'AI':>6s} {'%L2':>5s} {'%L1':>5s}")
    for case in sorted({k[0] for k in g}):
        pids = sorted({k[1] for k in g if k[0] == case})
        for i, pid in enumerate(pids):
            for kern in ("friction", "flux_x", "flux_y"):
                L = g.get((case, pid, kern))
                if not L: continue
                m = {k: float(np.mean([r[k] for r in L])) for k in keys}
                pbw, pfp = 100 * m["dram_gbs"] * 1e9 / PEAK_DRAM, 100 * m["gflops"] * 1e9 / PEAK_FP64
                w.writerow([case, i, kern, len(L)] + [round(m[k], 4) for k in keys] + [round(pbw, 2), round(pfp, 2)])
                print(f"{case:10s} {i:3d} {kern:9s} {len(L):3d} {m['t_us']:7.1f} {m['gflops']:8.1f} {pfp:6.1f} {m['dram_gbs']:9.1f} "
                      f"{pbw:6.1f} {m['ai_dram']:6.3f} {m['l2_pct']:5.1f} {m['l1_pct']:5.1f}")
print("written", out)
