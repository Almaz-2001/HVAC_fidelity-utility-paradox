"""Export the monotone temperature-response curves (rie09) to .dat for pgfplots.

Loads the canonical calibrated GB (v3.5) adapter and probes predicted next-step
zone temperature over the supply-temperature command range, for six representative
initial zone temperatures. Run: python docs/paper_combined/export_rie09_data.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "tikz_data"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from surrogate.direct_tsup_adapter import load_direct_tsup_adapter

adapter = load_direct_tsup_adapter(
    kind="v35_calibrated",
    summary_json=str(ROOT / "outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json"),
    device="cpu")
adapter.eval()
model = adapter.model

df = pd.read_csv(ROOT / "data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv")
rng = np.random.default_rng(42)
s = df.iloc[rng.choice(len(df), min(150, len(df)), replace=False)].reset_index(drop=True)
a0g = np.linspace(-1.0, 1.0, 11)
tsup = 18.0 + (a0g + 1.0) / 2.0 * (35.0 - 18.0)

curves = []
with torch.no_grad():
    for r in s.itertuples(index=False):
        tc = [float(model(torch.tensor([float(r.t_zone)]), torch.tensor([float(r.t_amb)]),
                          torch.tensor([float(r.hour)]), torch.tensor([float(r.day)]),
                          torch.tensor([float(av)]), torch.tensor([float(r.a1_raw)]))[1])
              for av in a0g]
        curves.append(tc)
curves = np.array(curves)
t0 = s["t_zone"].to_numpy()
order = np.argsort(t0)
pick = order[np.linspace(0, len(order) - 1, 6).astype(int)]

with (OUT / "rie09.dat").open("w", encoding="utf-8") as f:
    f.write("tsup " + " ".join(f"c{j}" for j in range(6)) + "\n")
    for i in range(len(tsup)):
        f.write(f"{tsup[i]} " + " ".join(f"{curves[p][i]}" for p in pick) + "\n")
with (OUT / "rie09_labels.dat").open("w", encoding="utf-8") as f:
    f.write("j t0\n")
    for j, p in enumerate(pick):
        f.write(f"{j} {t0[p]}\n")
print("wrote rie09.dat; initial T0:", [round(float(t0[p]), 1) for p in pick])
