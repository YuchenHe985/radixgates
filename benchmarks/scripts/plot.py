#!/usr/bin/env python3
"""Plot the simulated-worker benchmark (original RadixGates vs the upgraded fork).

Reads  benchmarks/results/sim/<scenario>/<mode>/{requests.csv,summary.json}
Writes benchmarks/plots/sim_<scenario>_timeline.png and sim_summary.png

Colours are slots 1 (blue) and 2 (orange) of a validated categorical palette;
text uses neutral ink tokens, never the series colour. Identity is carried by the
legend and direct end labels, not by colour alone.
"""
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results" / "sim"
OUT = ROOT / "benchmarks" / "plots"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
COLORS = {"original": "#eb6834", "upgraded": "#2a78d6"}
LABELS = {"original": "Original RadixGates (as delivered)", "upgraded": "Upgraded (rendezvous + failover + breaker)"}
TITLES = {"crash": "Worker crash (SIGKILL, then restart)", "brownout": "Gray failure (worker +3 s time-to-first-token)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def load(scenario, mode):
    rows = []
    with open(RES / scenario / mode / "requests.csv") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["t_s"]), float(r["latency_ms"]), r["ok"] == "true"))
    summary = json.loads((RES / scenario / mode / "summary.json").read_text())
    return rows, summary


def fault_window(scenario):
    env = (RES / scenario / "ENVIRONMENT.txt").read_text()
    m = re.search(r"fault at (\d+)s, recover at (\d+)s", env)
    return (int(m.group(1)), int(m.group(2))) if m else None


def per_second(rows, horizon):
    succ, p95 = [], []
    for s in range(horizon):
        bucket = [(lat, ok) for t, lat, ok in rows if s <= t < s + 1]
        if not bucket:
            succ.append(np.nan)
            p95.append(np.nan)
            continue
        succ.append(100.0 * sum(ok for _, ok in bucket) / len(bucket))
        lats = [lat for lat, ok in bucket if ok]
        p95.append(np.percentile(lats, 95) if len(lats) >= 5 else np.nan)
    return np.array(succ), np.array(p95)


def timeline(scenario):
    data = {m: load(scenario, m) for m in ("original", "upgraded")}
    horizon = int(min(s["duration_s"] for _, s in data.values()))
    x = np.arange(horizon) + 0.5
    win = fault_window(scenario)
    gap_note = ""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    for mode, (rows, _) in data.items():
        succ, p95 = per_second(rows, horizon)
        if mode == "original" and np.isnan(p95).any():
            gap_note = "Gaps in a P95 line: fewer than 5 requests completed from that second (clients blocked behind the slow worker)."
        for ax, y in ((ax1, succ), (ax2, p95)):
            ax.plot(x, y, color=COLORS[mode], linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=LABELS[mode])
            valid = ~np.isnan(y)
            if valid.any():
                k = np.where(valid)[0][-1]
                ax.plot(x[k], y[k], "o", color=COLORS[mode], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    if win:
        for ax in (ax1, ax2):
            ax.axvspan(win[0], win[1], color=INK2, alpha=0.08, linewidth=0)
        ax1.text(win[0] + 0.25, 5, "fault active", color=INK2, fontsize=9, va="center")

    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Success rate per second (%)")
    ax2.set_ylabel("P95 latency of successful requests (ms)")
    ax2.set_xlabel("Seconds since load start")
    ax2.set_ylim(bottom=0)
    fig.suptitle(f"{TITLES[scenario]}: 16 closed-loop clients, 4 simulated workers", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=2, frameon=False, fontsize=9)
    fig.text(0.01, 0.005, "Simulated workers (internal/mock): shows gateway behaviour under failure, not GPU performance."
             + ("\n" + gap_note.strip() if gap_note else ""), color=INK2, fontsize=7.5, va="bottom")
    fig.tight_layout(rect=(0, 0.05 if gap_note else 0.025, 1, 0.925))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"sim_{scenario}_timeline.png", dpi=160)
    plt.close(fig)


def summary_fig():
    scenarios = [s for s in ("crash", "brownout") if (RES / s / "original" / "summary.json").exists()]
    fig, axes = plt.subplots(2, len(scenarios), figsize=(4.8 * len(scenarios) + 0.6, 6.2), squeeze=False)
    for j, sc in enumerate(scenarios):
        sums = {m: json.loads((RES / sc / m / "summary.json").read_text()) for m in ("original", "upgraded")}
        panels = [
            ("Requests that completed cleanly (%)", lambda s: 100 * s["success_rate"], (0, 108), "{:.1f}%"),
            ("P99 latency, successful requests (ms)", lambda s: s["latency_ms_ok_only"]["p99"], None, "{:,.0f}"),
        ]
        for i, (ylabel, fn, ylim, fmt) in enumerate(panels):
            ax = axes[i][j]
            vals = [fn(sums[m]) for m in ("original", "upgraded")]
            xs = np.arange(2)
            bars = ax.bar(xs, vals, width=0.34, color=[COLORS["original"], COLORS["upgraded"]], linewidth=0)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=10)
            ax.set_xticks(xs)
            ax.set_xticklabels(["Original", "Upgraded"])
            ax.grid(axis="x", visible=False)
            if ylim:
                ax.set_ylim(*ylim)
            else:
                ax.set_ylim(0, max(vals) * 1.18)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(TITLES[sc].split(" (")[0], loc="left")
    fig.text(0.01, 0.005, "Simulated workers, 30 s per run, fault from t=8 s to t=20 s. Raw data: benchmarks/results/sim/.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "sim_summary.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    for sc in ("crash", "brownout"):
        if (RES / sc / "upgraded" / "requests.csv").exists():
            timeline(sc)
    summary_fig()
    print("plots written to", OUT)
