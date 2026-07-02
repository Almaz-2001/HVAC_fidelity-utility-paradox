"""Recreate the four main-paper Block-1/Block-2 diagnostic figures with BB/GB naming.

Self-contained, data-driven from committed artefacts; writes directly to the paper
figure dir with the exact filenames the manuscript references. Replaces the stale
v3/v3.5-labelled bespoke figures (rie_fig03/04/05, fig_block2_ms_decomposition).

  BB = black-box surrogate (v3)   GB = grey-box surrogate (v3.5)
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import _figstyle as fs

OUT = ROOT / "docs/paper_combined/figures"
BB = fs.V3            # black-box (green)
RAWGB = "#e08214"     # raw grey-box (orange)
CALGB = fs.ACCURATE   # calibrated grey-box (red)
TIME = "#21867a"      # r_time (teal)
SEV = "#9b2226"       # r_sev (dark red)

V3R = "outputs/surrogate_v3_rollout_prepared_15min/v3"
RAWR = "outputs/surrogate_v35_rollout_prepared_15min_episodeaware/raw_v35"
CALR = "outputs/surrogate_v35_rollout_prepared_15min_episodeaware/calibrated_v35"


def rd(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.18)


# ---------- rie_fig03: Stage A/B/C calibration diagnostics ----------
def fig_stage_abc():
    sb = rd(f"{V3R}/../../surrogate_v35_inverse_boptest_15min_episodeaware/stage_b_history_v35.csv") \
        if False else rd("outputs/surrogate_v35_inverse_boptest_15min_episodeaware/stage_b_history_v35.csv")
    raw_w, cal_w = rd(f"{RAWR}/window_errors.csv"), rd(f"{CALR}/window_errors.csv")
    raw_r, cal_r = rd(f"{RAWR}/all_full_rollouts.csv"), rd(f"{CALR}/all_full_rollouts.csv")

    def h_rmse(df, h):
        return float(df.loc[df["horizon_h"] == h, "temp_window_rmse_c"].dropna().mean())
    def pmae(df):
        return float(df["power_error_w"].abs().mean())

    fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.2))
    # (a) C_zon Stage B identification
    prior = 4.200
    ax[0].axhspan(prior * 0.9, prior * 1.1, color="#eef2f6")
    ax[0].axhline(prior, color="0.5", ls="--", lw=1.3, label="prior")
    ax[0].plot(sb["epoch"], sb["c_zon_j_per_k"] / 1e5, color=CALGB, lw=2.6)
    final = float(sb["c_zon_j_per_k"].iloc[-1]) / 1e5
    ax[0].scatter([sb["epoch"].iloc[-1]], [final], s=55, color=CALGB, ec="#222", zorder=4,
                  label=f"final {final:.3f}")
    ax[0].set_title("(a) Stage B: $C_{\\mathrm{zon}}$ identification", fontsize=10.5, weight="bold")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("$C_{\\mathrm{zon}}$ ($10^5$ J/K)")
    ax[0].legend(frameon=False, fontsize=8); _despine(ax[0])
    # (b) absolute error: raw -> calibrated
    one_r, one_c = h_rmse(raw_w, 1.0), h_rmse(cal_w, 1.0)
    d24_r, d24_c = h_rmse(raw_w, 24.0), h_rmse(cal_w, 24.0)
    x = np.arange(2)
    ax[1].bar(x - 0.19, [one_r, d24_r], 0.38, color=RAWGB, label="raw GB")
    ax[1].bar(x + 0.19, [one_c, d24_c], 0.38, color=CALGB, label="calibrated GB")
    for xi, (r, c) in zip(x, [(one_r, one_c), (d24_r, d24_c)]):
        ax[1].text(xi + 0.19, c + 0.03, f"$-${100*(1-c/r):.0f}%", ha="center", va="bottom",
                   fontsize=9, color=CALGB, weight="bold")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["1 h horizon\nRMSE$_T$", "24 h horizon\nRMSE$_T$"])
    ax[1].set_ylabel("rollout RMSE$_T$ (°C)")
    ax[1].set_ylim(0, max(one_r, d24_r) * 1.2)
    ax[1].set_title("(b) Rollout error: raw $\\to$ calibrated GB", fontsize=10.5, weight="bold")
    ax[1].legend(frameon=False, fontsize=8, loc="upper left"); _despine(ax[1])
    # (c) residual distribution
    er, ec = raw_r["temp_error_c"].dropna().to_numpy(), cal_r["temp_error_c"].dropna().to_numpy()
    ax[2].hist(er, bins=80, density=True, color=RAWGB, alpha=0.55, label=f"raw GB ($\\sigma$={er.std():.2f})")
    ax[2].hist(ec, bins=80, density=True, color=CALGB, alpha=0.55, label=f"calibrated GB ($\\sigma$={ec.std():.2f})")
    ax[2].set_xlim(-3, 3)
    ax[2].set_title("(c) Prediction-residual distribution", fontsize=10.5, weight="bold")
    ax[2].set_xlabel("prediction error (°C)"); ax[2].set_ylabel("density")
    ax[2].legend(frameon=False, fontsize=8); _despine(ax[2])
    fig.suptitle("Stage A/B/C calibration: bounded $C_{\\mathrm{zon}}$, lower absolute error, tighter residuals",
                 fontsize=12.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "rie_fig03_stage_abc_diagnostics")


# ---------- rie_fig04: predictive validity ----------
def fig_predictive_validity():
    specs = [("BB hourly", f"{V3R}", BB), ("raw GB", f"{RAWR}", RAWGB), ("calibrated GB", f"{CALR}", CALGB)]
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for label, base, color in specs:
        w = rd(f"{base}/window_errors.csv")
        xs = [1.0, 4.0, 8.0, 24.0]
        ys = [float(w.loc[w["horizon_h"] == h, "temp_window_rmse_c"].dropna().mean()) for h in xs]
        ax[0].plot(xs, ys, marker="o", lw=2.3, color=color, label=label)
    ax[0].set_xscale("log"); ax[0].set_xticks([1, 4, 8, 24])
    ax[0].get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax[0].set_title("(a) Multi-horizon rollout RMSE", fontsize=10.5, weight="bold")
    ax[0].set_xlabel("prediction horizon (h)"); ax[0].set_ylabel("RMSE$_T$ (°C)")
    ax[0].legend(frameon=False, fontsize=8); _despine(ax[0])
    for label, base, color in specs:
        r = rd(f"{base}/all_full_rollouts.csv")
        a = np.sort(np.abs(r["temp_error_c"].dropna().to_numpy()))
        cdf = np.arange(1, len(a) + 1) / len(a)
        ax[1].plot(a, cdf, lw=2.1, color=color, label=label)
    for xv in (0.5, 1.0, 1.5):
        ax[1].axvline(xv, color="0.7", ls="--", lw=0.8)
    ax[1].set_xlim(0, 5); ax[1].set_ylim(0, 1.01)
    ax[1].set_title("(b) Engineering tolerance CDF", fontsize=10.5, weight="bold")
    ax[1].set_xlabel("|prediction error| (°C)"); ax[1].set_ylabel("fraction below threshold")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right"); _despine(ax[1])
    fig.suptitle("Predictive validity across horizons and engineering error tolerances", fontsize=12.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "rie_fig04_predictive_validity")


# ---------- rie_fig05: matched-corpus attribution (waterfall) ----------
def fig_matched_waterfall():
    m = rd("reports/block1_corpus_matched_comparison.csv").set_index("variant")
    start = float(m.loc["v3_hourly"]["rmse_24h_c"])
    mid = float(m.loc["v3_15min_matched"]["rmse_24h_c"])
    end = float(m.loc["v35_calibrated"]["rmse_24h_c"])
    raw = float(m.loc["v35_raw"]["rmse_24h_c"])
    res, cal = start - mid, mid - end
    tot = res + cal
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    ax.bar([0], [start], 0.56, color=BB, ec="#222", lw=0.5)
    ax.bar([1], [-res], 0.56, bottom=[start], color=fs.MATCHED, ec="#222", lw=0.5)
    ax.bar([2], [mid], 0.56, color="#f0c8c0", ec="#222", lw=0.5)
    ax.bar([3], [-cal], 0.56, bottom=[mid], color=CALGB, ec="#222", lw=0.5)
    ax.bar([4], [end], 0.56, color=CALGB, ec="#222", lw=0.5)
    ax.text(1, start - res / 2, f"$-${res:.3f}\nresolution\n({100*res/tot:.0f}%)", ha="center", va="center",
            fontsize=8.5, color="white", weight="bold")
    ax.text(3, mid - cal / 2, f"$-${cal:.3f}\nStage A/B/C\n({100*cal/tot:.0f}%)", ha="center", va="center",
            fontsize=8.5, color="white", weight="bold")
    for xi, v in [(0, start), (2, mid), (4, end)]:
        ax.text(xi, v + 0.03, f"{v:.3f}", ha="center", fontsize=11, weight="bold")
    ax.axhline(raw, color=RAWGB, ls="--", lw=1.8)
    ax.text(4.1, raw + 0.02, f"raw GB (uncalibrated) {raw:.3f}", ha="right", va="bottom", fontsize=8.5,
            color=RAWGB, style="italic")
    ax.plot([0.28, 0.72], [start, start], color="0.6", ls=":", lw=1)
    ax.plot([1.28, 1.72], [mid, mid], color="0.6", ls=":", lw=1)
    ax.plot([2.28, 2.72], [mid, mid], color="0.6", ls=":", lw=1)
    ax.plot([3.28, 3.72], [end, end], color="0.6", ls=":", lw=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["BB (hourly)", "$-$ resolution", "matched-res BB\n(15 min)", "$-$ calibration", "calibrated GB"])
    ax.set_ylabel("24 h rollout RMSE$_T$ (°C)")
    ax.set_title("Matched-corpus decomposition: $\\Delta$RMSE $= \\Delta_{\\mathrm{resolution}} + \\Delta_{\\mathrm{physics\\text{-}calibration}}$",
                 fontsize=12, weight="bold")
    _despine(ax)
    fig.tight_layout()
    _save(fig, "rie_fig05_matched_corpus_attribution")


# ---------- fig_block2_ms_decomposition ----------
def fig_ms_decomposition():
    # canonical live-BOPTEST KPIs from the main-paper Table (m_s, comfort-violation %)
    bars = [("pure BB\n(peak)", 0.073, 1.49), ("direct GB\n(peak)", 1.046, 77.08),
            ("hybrid\n(peak)", 0.087, 4.69), ("pure BB\n(typical)", 0.095, 4.39),
            ("direct GB\n(typical)", 1.102, 82.37), ("hybrid\n(typical)", 0.041, 2.38)]
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    x = np.arange(len(bars))
    rtime = [v / 100.0 for _, _, v in bars]
    rsev = [max(ms - rt, 0.0) for (_, ms, _), rt in zip(bars, rtime)]
    ax.bar(x, rtime, 0.62, color=TIME, ec="#222", lw=0.4, label="$r_{\\mathrm{time}}$ (violation fraction)")
    ax.bar(x, rsev, 0.62, bottom=rtime, color=SEV, ec="#222", lw=0.4, label="$r_{\\mathrm{sev}}$ (worst rel. severity)")
    for xi, (_, ms, _) in zip(x, bars):
        ax.text(xi, ms + 0.015, f"{ms:.3f}", ha="center", fontsize=10, weight="bold")
    ax.axhline(0.10, color="0.45", ls="--", lw=1.2)
    ax.text(x[-1] + 0.1, 0.105, "$m_s=0.10$", ha="right", va="bottom", fontsize=9, color="0.4")
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("$m_s = r_{\\mathrm{time}} + r_{\\mathrm{sev}}$")
    ax.set_title("Maintenance-score decomposition on the live BOPTEST windows", fontsize=12.5, weight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper center"); _despine(ax)
    fig.tight_layout()
    _save(fig, "fig_block2_ms_decomposition")


def main():
    fs.apply()
    fig_stage_abc()
    fig_predictive_validity()
    fig_matched_waterfall()
    fig_ms_decomposition()


if __name__ == "__main__":
    main()
