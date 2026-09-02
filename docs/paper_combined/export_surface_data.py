"""Export the surface-roughness mechanism data (Fig, main) to .dat for pgfplots.

Reuses evaluation/build_mechanism_surface_diagnostic.py to load the .pt surrogate
models once and dump: median action->next-T response curves, per-state relative
roughness box statistics (+ fold vs v3), closed-loop action saturation, and live m_s.
Run: python docs/paper_combined/export_surface_data.py
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "tikz_data"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import build_mechanism_surface_diagnostic as M
from surrogate.direct_tsup_adapter import load_direct_tsup_adapter

ORDER = ["v3 hourly (1h)", "v3 matched (15min)", "v3.5 calibrated"]
adapters = {name: load_direct_tsup_adapter(**kw) for name, kw in M.SURROGATES}

# (A) median normalised response curves
curve_cols = [np.asarray(M.A0)]
for name in ORDER:
    med = np.median(np.array(M.per_state_curves(adapters[name])), axis=0)
    rng = med.max() - med.min()
    curve_cols.append(med / rng)
with (OUT / "surface_curves.dat").open("w", encoding="utf-8") as f:
    f.write("A0 v3 matched v35\n")
    for i in range(len(M.A0)):
        f.write(" ".join(f"{c[i]}" for c in curve_cols) + "\n")

# (B) per-state relative-roughness box statistics + fold vs v3
sh = {r["surrogate"]: float(r["rel_roughness"]) for r in csv.DictReader(open(ROOT / "reports/block2_mechanism_surface_sharpness.csv"))}
base = sh["v3 hourly (1h)"]
with (OUT / "surface_box.dat").open("w", encoding="utf-8") as f:
    f.write("idx lw q1 med q3 uw fold\n")
    for i, name in enumerate(ORDER):
        v = np.array(M.per_state_rel(adapters[name]), dtype=float)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        f.write(f"{i} {v.min()} {q1} {med} {q3} {v.max()} {sh[name]/base}\n")

# (C,D) closed-loop saturation and live m_s
sat = {name: M._saturation_pct(M.TRACE_DIRS[name]) for name in ORDER}
scatter = list(csv.DictReader(open(ROOT / "reports/block2_fidelity_utility_scatter.csv")))
msvals = [float(scatter[i]["m_s_mean"]) for i in range(3)]
with (OUT / "surface_bars.dat").open("w", encoding="utf-8") as f:
    f.write("idx sat ms\n")
    for i, name in enumerate(ORDER):
        f.write(f"{i} {sat[name]} {msvals[i]}\n")

print("wrote surface_curves.dat, surface_box.dat, surface_bars.dat")
print("folds:", {name: round(sh[name] / base, 2) for name in ORDER})
