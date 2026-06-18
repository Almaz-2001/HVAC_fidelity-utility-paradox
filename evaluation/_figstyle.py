"""Shared figure style + canonical-number loader for all paper figures.

Single source of truth so that (a) one colour = one meaning across every figure
(v3 = green, accurate single-model surrogate = red/orange, hybrid = blue), the
engineering threshold constants are defined once, and (b) no headline number is
hand-typed into a plotting script -- paper_numbers() loads them from the committed
artefacts (reports/, outputs/). Import this from any figure generator.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd

# --- unified palette (one colour = one meaning) ---
V3 = "#1b7837"          # coarse black-box v3, usable
ACCURATE = "#b2182b"    # calibrated v3.5 (accurate single-model), collapse
MATCHED = "#d6604d"     # matched-resolution v3 (accurate single-model), collapse
HYBRID = "#2166ac"      # role-separating hybrid, robust
EDGE = "#222222"
NEUTRAL = "#6f4e7c"

# --- engineering threshold / reference constants (pre-specified, not data) ---
COMFORT_LO, COMFORT_HI = 21.0, 24.0     # deg C occupied comfort band
MS_COLLAPSE = 1.0                        # m_s = 1 live-failure threshold
MS_USABLE = 0.1                          # tight usable band
VIOLATION_BAR = 5.0                      # 5 % comfort-violation reference
TAU_FACTOR = 1.25                        # transfer threshold tau = 1.25 x PI

TRACE_DIRS = {                           # peak-window closed-loop traces
    "v3": "bestest_air_article7_style_15min",
    "matched": "bestest_air_pure_v3_15min",
    "v35": "block2_bestest_air_15min_thermostatic_v35",
    "hybrid": "block2_thermostatic_hybrid_v3_v35_l010",
}


def apply():
    """Apply consistent rcParams (call once per figure script)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.grid": False, "savefig.dpi": 300, "figure.dpi": 120,
        "axes.edgecolor": "0.5", "axes.linewidth": 0.8,
    })


def _sat(root: Path, key: str) -> float:
    a = pd.read_csv(root / "outputs" / TRACE_DIRS[key] / "traces" / "peak_heat_window_thermostatic.csv")["a0"]
    return 100.0 * float((a.abs() > 0.9).mean())


def paper_numbers(root: Path) -> dict:
    """Load every headline figure number from committed artefacts (no hand entry)."""
    root = Path(root)
    cm = pd.read_csv(root / "reports/block1_corpus_matched_comparison.csv").set_index("variant")
    arch = pd.read_csv(root / "reports/hou_evins_architecture_justification_table.csv").set_index("variant")
    sh = pd.read_csv(root / "reports/block2_mechanism_surface_sharpness.csv").set_index("surrogate")
    sc = pd.read_csv(root / "reports/block2_fidelity_utility_scatter.csv")
    spd = pd.read_csv(root / "reports/speed_benchmark_table.csv").set_index("backend")

    def ms(prefix, col="m_s_mean"):
        return float(sc[sc.controller.str.startswith(prefix)].iloc[0][col])

    base_r = float(sh.loc["v3 hourly (1h)"]["rel_roughness"])
    return {
        "rmse": {
            "v3": float(cm.loc["v3_hourly"]["rmse_24h_c"]),
            "matched": float(cm.loc["v3_15min_matched"]["rmse_24h_c"]),
            "v35": float(arch.loc["v35_calibrated"]["block1_rollout_24h_rmse_c"]),
        },
        "rough_fold": {
            "matched": float(sh.loc["v3 matched (15min)"]["rel_roughness"]) / base_r,
            "v35": float(sh.loc["v3.5 calibrated"]["rel_roughness"]) / base_r,
        },
        "m_s": {
            "v3": ms("v3 ("), "matched": ms("matched"), "v35": ms("v3.5"),
            "hybrid_mean": ms("hybrid"), "hybrid_typ": ms("hybrid", "m_s_typ"),
        },
        "saturation": {k: _sat(root, k) for k in ("v3", "matched", "v35", "hybrid")},
        "speedup": float(spd.loc["hybrid_v3_v35_surrogate"]["speedup_vs_boptest_rte"]),
    }
