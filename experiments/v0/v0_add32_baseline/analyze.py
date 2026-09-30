"""Parse the saved V0 logs; retain round 0 but exclude it from aggregates.

Usage: python3 analyze.py [path/to/add32-repeat-3rE4Wh]
Requires matplotlib for the figure. Does not run either solver.
"""
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "add32-repeat-3rE4Wh"
out = Path(__file__).parent

def number(text, label):
    match = re.search(re.escape(label) + r"\s*([0-9.eE+-]+)", text)
    if not match:
        raise ValueError(f"Missing field: {label}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite field: {label}")
    return value

rows = []
for r in range(6):
    folder = root / f"round-{r}"
    klu = (folder / "klu.txt").read_text()
    glu = (folder / "glu.txt").read_text()
    error = (folder / "error.txt").read_text()
    e = number(error, "relative_l2_error =")
    if "correctness = PASS" not in error or e > 1e-6 or number(error, "n =") != 4960:
        raise ValueError(f"Round {r} did not pass correctness")
    k = number(klu, "Total solve wall time:")
    g = number(glu, "Total solve wall time:")
    if min(k, g) <= 0:
        raise ValueError("Nonpositive elapsed time")
    rows.append(dict(round=r, warmup=(r == 0), klu_total_ms=k, glu_total_ms=g,
                     glu_legacy_event_ms=number(glu, "Total GPU time:"),
                     relative_l2_error=e, paired_speedup=k/g, correctness="PASS"))

with (out / "samples.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

formal = rows[1:]
k = [row["klu_total_ms"] for row in formal]
g = [row["glu_total_ms"] for row in formal]
errors = [row["relative_l2_error"] for row in formal]
km, gm = statistics.median(k), statistics.median(g)
summary = dict(source_run=root.name, matrix="add32_csr.mtx", n=4960,
               warmup_rounds=[0], measured_rounds=[1, 2, 3, 4, 5],
               process_model="fresh process each run; KLU then GLU; same b=ones",
               klu_median_ms=km, glu_median_ms=gm, speedup_ratio_of_medians=km/gm,
               glu_slowdown=gm/km, klu_range_ms=[min(k),max(k)],
               glu_range_ms=[min(g),max(g)], max_relative_l2_error=max(errors),
               gpu_util=None, achieved_occupancy=None, bandwidth_util=None,
               peak_device_memory=None, pure_lu_kernel_time_ms=None,
               triangular_solve_ms=None, gflops=None)
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
for ax, values, median, title, color, ymax in [
    (axes[0], k, km, "KLU: complete solve", "#15856B", 17),
    (axes[1], g, gm, "GLU V0: complete solve", "#2864C8", 90),
]:
    ax.plot(range(1,6), values, "o-", color=color, linewidth=1.8)
    ax.axhline(median, color=color, linestyle="--", alpha=.65,
               label=f"Median {median:.3f} ms")
    for i, value in enumerate(values, 1):
        ax.annotate(f"{value:.2f}", (i, value), xytext=(0,8),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set(title=title, xlabel="Measured round", ylabel="Wall-clock time (ms)",
           ylim=(0,ymax), xticks=range(1,6), xlim=(.6,5.4))
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="lower left", frameon=False)

ax = axes[2]
ax.plot(range(1,6), errors, "o-", color="#7445AD")
ax.axhline(1e-6, color="#B54738", linestyle="--", label="Threshold: 1e-6")
ax.set_yscale("log")
ax.set(title="Solution error vs KLU", xlabel="Measured round",
       ylabel="Relative L2 error (log scale)", ylim=(1e-16,1e-5),
       xticks=range(1,6), xlim=(.6,5.4))
ax.text(.05,.34, f"All 5 pass\nMaximum: {max(errors):.3e}", transform=ax.transAxes,
        color="#7445AD")
ax.legend(loc="upper right", frameon=False)
ax.grid(axis="y", alpha=.18)
fig.suptitle(f"V0 / add32 | KLU-to-GLU speedup: {km/gm:.3f}x | GLU is {gm/km:.2f}x slower",
             fontsize=14, fontweight="bold", y=.99)
fig.text(.5,.015,"Round 0 excluded by protocol. Fresh process each run. Time panels use different y-axis scales.\n"
         "Includes input preparation and solve; excludes residual checking and writing the solution.",
         ha="center", fontsize=9, color="#555555")
fig.tight_layout(rect=(0,.12,1,.93))
fig.savefig(out / "baseline.png", dpi=180)
fig.savefig(out / "baseline.svg")
plt.close(fig)
print(json.dumps(summary, indent=2))
