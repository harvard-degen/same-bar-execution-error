#!/usr/bin/env python3
"""make_fig1_v3.py - regenerate Figure 1 (SVG) for W1 paper v3.0.

Dependency-free (no matplotlib in this environment). Recomputes every plotted
arm from the canonical experiment module and ASSERTS terminal-wealth equality
with w1_results_v4.json before writing the SVG. A hand-typed number in a figure
is a bug; this script makes that bug impossible.
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w1_experiment_v4 as exp

OUT = os.path.join(exp.PAPERS, "fig1_timeline_arms_v3.svg")

o, c, dates, src = exp.load_crypto("BTCUSDT")
n = len(o)
p = exp.build_positions(c, exp.M0, exp.ZW, exp.THR0)
stored = json.load(open(os.path.join(exp.PAPERS, "w1_results_v4.json")))["btc"]["arms"]

ARMS = ("A", "B", "C", "F", "G")
NAMES = {"A": "A clean next-open", "B": "B T+1 delay", "C": "C signal-close",
         "F": "F same-bar ERROR", "G": "G unshifted ERROR"}
COLORS = {"A": "#2e7d32", "B": "#7cb342", "C": "#1565c0",
          "F": "#c62828", "G": "#ef6c00"}

series = {}
for arm in ARMS:
    g = exp.arm_nets(o, c, p, arm)
    w, path = 1.0, [1.0]
    for r in g:
        w *= (1.0 + r)
        path.append(w)
    target = stored[arm]["x"]
    assert abs(w - target) <= 0.01 * max(1.0, abs(target)), \
        f"figure assert FAILED for arm {arm}: computed {w:.4f} vs stored {target}"
    series[arm] = path
print("assert OK: all plotted arms equal stored results within 1%")

W, H = 1000, 560
ML, MR, MT, MB = 78, 20, 44, 52
ymax = max(s[-1] for s in series.values()) * 1.3
lo, hi = 0.0, math.log10(ymax)

def X(i):
    return ML + (W - ML - MR) * i / n

def Y(v):
    lv = math.log10(max(v, 1.0))
    return H - MB - (H - MT - MB) * (lv - lo) / (hi - lo)

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'font-family="DejaVu Sans, sans-serif" font-size="13">',
    f'<rect width="{W}" height="{H}" fill="white"/>',
    f'<text x="{W/2}" y="26" text-anchor="middle" font-size="15" font-weight="bold">'
    f'Figure 1. BTCUSDT daily momentum (M=20, z&gt;0.5): identical positions, five ledgers</text>',
]
e = 0
while 10 ** e <= ymax:
    y = Y(10 ** e)
    parts.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{W-MR}" y2="{y:.1f}" stroke="#e6e6e6"/>')
    lab = "1" if e == 0 else f"10^{e}"
    parts.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{lab}</text>')
    e += 1
for i in range(0, n + 1, 365):
    x = X(min(i, n))
    parts.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{H-MB}" stroke="#f2f2f2"/>')
for arm in ("B", "C", "A", "G", "F"):
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(series[arm]) if i % 3 == 0)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[arm]}" stroke-width="1.6"/>')
ly = MT + 14
for arm in ARMS:
    lab = f"{NAMES[arm]} ({stored[arm]['x']:,.2f}x)"
    parts.append(f'<rect x="{ML+16}" y="{ly-8}" width="18" height="5" fill="{COLORS[arm]}"/>')
    parts.append(f'<text x="{ML+42}" y="{ly-1}">{lab}</text>')
    ly += 21
parts.append(f'<text x="{W/2}" y="{H-14}" text-anchor="middle" fill="#555">'
             f'bar index ({dates[0]} .. {dates[-1]}); terminal wealth, log scale; '
             f'10 bp per unit turnover; v4 canonical conventions</text>')
parts.append('</svg>')

with open(OUT, "w") as fh:
    fh.write("\n".join(parts))
print("wrote", OUT)
