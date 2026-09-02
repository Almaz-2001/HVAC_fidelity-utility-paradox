"""Seed-band figure for the HDRL lambda_temp_disagree sweep (N seeds).

The frozen sweep (reports/block2_hdrl_lambda_sweep_summary.csv) is single-seed,
which is the weakest evidence in the manuscript and the first thing an IEEE
Access reviewer will attack. This figure replaces the single-seed line with a
mean +/- sd band read straight from
``reports/block2_hdrl_lambda_sweep_seed_band.csv`` -- no hand-entered numbers --
and plots it against the two pre-specified engineering references (the m_s = 1
collapse line and the 5 % comfort-violation bar).

The frozen single-seed points are overlaid as hollow markers so the reader can
see how much of the reported trend survives seed variance.

Outputs
-------
    reports/figures/hdrl_seed_band/block2_hdrl_lambda_seed_band.{pdf,png}

Usage
-----
    python evaluation/build_hdrl_seed_band_figure.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import _figstyle as fs

SRC = ROOT / "reports" / "block2_hdrl_lambda_sweep_seed_band.csv"
FROZEN = ROOT / "reports" / "block2_hdrl_lambda_sweep_summary.csv"
OUT_DIR = ROOT / "reports" / "figures" / "hdrl_seed_band"

WINDOW_LABEL = {"peak": "Peak heat window", "typical": "Typical heat window"}
LAMBDA_OF = {"l000": 0.00, "l003": 0.03, "l005": 0.05, "l010": 0.10}


def load_frozen() -> pd.DataFrame:
    if not FROZEN.exists():
        return pd.DataFrame()
    df = pd.read_csv(FROZEN)
    df["window"] = df["scenario"].map(
        {"peak_heat_window": "peak", "typical_heat_window": "typical"})
    df["lambda_temp_disagree"] = df["variant"].map(LAMBDA_OF)
    return df.dropna(subset=["window", "lambda_temp_disagree"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the HDRL lambda sweep seed band.")
    parser.add_argument("--src", default=str(SRC))
    parser.add_argument("--no-frozen", action="store_true",
                        help="Omit the frozen single-seed overlay.")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(
            f"{src} not found.\nRun: python evaluation/build_hdrl_seed_band.py")

    band = pd.read_csv(src)
    frozen = pd.DataFrame() if args.no_frozen else load_frozen()

    fs.apply()
    # _figstyle turns on text.usetex to match the manuscript fonts, but the runtime
    # container ships no TeX. Fall back to mathtext there rather than dying: the
    # publication-quality PDF is regenerated on the machine that has LaTeX.
    if shutil.which("latex") is None:
        plt.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans",
                             "mathtext.fontset": "dejavusans"})
        print("[note] no 'latex' binary found: rendering with mathtext instead. "
              "Re-run where LaTeX is available for the final PDF.")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)

    n_seeds = int(band["n_seeds"].max()) if not band.empty else 0

    for ax, window in zip(axes, ("peak", "typical")):
        sub = band[band["window"] == window].sort_values("lambda_temp_disagree")
        if sub.empty:
            ax.set_visible(False)
            continue

        x = sub["lambda_temp_disagree"].to_numpy(dtype=float)
        y = sub["m_s_mean"].to_numpy(dtype=float)
        e = sub["m_s_std"].to_numpy(dtype=float)

        ax.fill_between(x, y - e, y + e, color=fs.HYBRID, alpha=0.18, linewidth=0)
        ax.errorbar(x, y, yerr=e, color=fs.HYBRID, marker="D", markersize=6.5,
                    markeredgecolor="white", markeredgewidth=0.8, linewidth=1.6,
                    capsize=3.5, zorder=3,
                    label=f"HDRL, mean $\\pm$ sd over {n_seeds} seeds")

        if not frozen.empty:
            fz = frozen[frozen["window"] == window].sort_values("lambda_temp_disagree")
            ax.plot(fz["lambda_temp_disagree"], fz["m_s"], linestyle=":", linewidth=1.2,
                    color="0.35", marker="o", markersize=5.5, markerfacecolor="white",
                    markeredgecolor="0.35", zorder=2,
                    label="frozen single-seed sweep")

        fs.threshold(ax, fs.MS_USABLE, f"usable band $m_s = {fs.MS_USABLE:g}$")
        ax.set_title(WINDOW_LABEL[window], fontsize=10)
        ax.set_xlabel(r"disagreement penalty $\lambda_T$")
        ax.set_xticks(sorted(LAMBDA_OF.values()))
        ax.margins(x=0.08)

    axes[0].set_ylabel(r"maintenance score $m_s$")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))

    # Deliberately descriptive: the title must not assert the trend before the
    # seed band has been read. build_hdrl_seed_band.py prints the verdict.
    fig.suptitle(r"HDRL sensitivity to the disagreement penalty $\lambda_T$, across seeds",
                 fontsize=11, y=1.0)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"block2_hdrl_lambda_seed_band.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print("wrote", path.relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
