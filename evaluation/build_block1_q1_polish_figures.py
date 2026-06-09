"""Build Q1-polished Block 1 diagnostic figures from existing artifacts.

The script intentionally avoids inventing confidence intervals. Horizon bands
are computed from the available window-level rollout errors, and residual
diagnostics are computed from stored rollout prediction CSVs.
"""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "article_real"

BLUE = "#2f5d8c"
TEAL = "#21867a"
ORANGE = "#b25f2c"
PURPLE = "#6f4e7c"
GREY = "#5c6470"
LIGHT_GREY = "#eef2f6"


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def style(ax: plt.Axes, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def bootstrap_ci(values: np.ndarray, seed: int = 42, n_boot: int = 1000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) > 6000:
        values = rng.choice(values, size=6000, replace=False)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def fig_czon_prior_band() -> None:
    df = read("outputs/surrogate_v35_inverse_boptest_15min_episodeaware/stage_b_history_v35.csv")
    prior = 4.200e5
    final = float(df["c_zon_j_per_k"].iloc[-1])
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.axhspan((prior * 0.9) / 1e5, (prior * 1.1) / 1e5, color=LIGHT_GREY, label="prior physical band (+/-10%)")
    ax.axhline(prior / 1e5, color=GREY, linestyle="--", linewidth=1.4, label="prior 4.200e5 J/K")
    ax.plot(df["epoch"], df["c_zon_j_per_k"] / 1e5, color=TEAL, linewidth=2.6, label="identified trajectory")
    ax.axhline(final / 1e5, color=ORANGE, linestyle=":", linewidth=2.0, label=f"final {final/1e5:.3f}e5 J/K")
    ax.scatter([df["epoch"].iloc[-1]], [final / 1e5], s=60, color=ORANGE, edgecolor="#222222", zorder=4)
    ax.annotate(
        "converges inside\nphysical prior band",
        xy=(df["epoch"].iloc[-1], final / 1e5),
        xytext=(df["epoch"].iloc[-1] * 0.58, final / 1e5 + 0.28),
        arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.2),
        fontsize=9,
        color="#333333",
    )
    style(ax, "Stage B inverse identification of $C_{zon}$ with physical prior band", "Stage B epoch", "$C_{zon}$ ($10^5$ J/K)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "block1_q1_polish_czon_prior_band")


def fig_horizon_ci() -> None:
    specs = [
        ("v3", "outputs/surrogate_v3_rollout_prepared_15min/v3/window_errors.csv", BLUE),
        ("raw v3.5", "outputs/surrogate_v35_rollout_prepared_15min_episodeaware/raw_v35/window_errors.csv", ORANGE),
        ("calibrated v3.5", "outputs/surrogate_v35_rollout_prepared_15min_episodeaware/calibrated_v35/window_errors.csv", TEAL),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    for label, rel, color in specs:
        df = read(rel)
        xs, means, lows, highs = [], [], [], []
        for h in [1.0, 4.0, 8.0, 24.0]:
            vals = df.loc[df["horizon_h"] == h, "temp_window_rmse_c"].dropna().to_numpy()
            mean, low, high = bootstrap_ci(vals)
            xs.append(h)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        ax.plot(xs, means, marker="o", linewidth=2.3, color=color, label=label)
        ax.fill_between(xs, lows, highs, color=color, alpha=0.16, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks([1, 4, 8, 24])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    style(ax, "Multi-horizon rollout RMSE with 95% bootstrap CI", "Prediction horizon (h)", "Window RMSE$_T$ (degC)")
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.03,
        "CI computed from window-level rollout RMSE distributions in window_errors.csv; no synthetic seed variance added.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    save(fig, "block1_q1_polish_multi_horizon_ci")


def normal_qq_points(values: np.ndarray, n: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) > n:
        values = rng.choice(values, size=n, replace=False)
    values = np.sort((values - values.mean()) / values.std(ddof=1))
    probs = (np.arange(1, len(values) + 1) - 0.5) / len(values)
    nd = NormalDist()
    theoretical = np.array([nd.inv_cdf(float(p)) for p in probs])
    return theoretical, values


def fig_residual_distribution_qq() -> None:
    specs = [
        ("v3", "outputs/surrogate_v3_rollout_prepared_15min/v3/all_full_rollouts.csv", BLUE),
        ("raw v3.5", "outputs/surrogate_v35_rollout_prepared_15min_episodeaware/raw_v35/all_full_rollouts.csv", ORANGE),
        ("calibrated v3.5", "outputs/surrogate_v35_rollout_prepared_15min_episodeaware/calibrated_v35/all_full_rollouts.csv", TEAL),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    residuals: dict[str, np.ndarray] = {}
    for label, rel, color in specs:
        df = read(rel)
        err = df["temp_error_c"].dropna().to_numpy()
        residuals[label] = err
        axes[0].hist(err, bins=90, density=True, histtype="step", linewidth=2.0, color=color, label=label)
        abs_sorted = np.sort(np.abs(err))
        cdf = np.arange(1, len(abs_sorted) + 1) / len(abs_sorted)
        axes[1].plot(abs_sorted, cdf, color=color, linewidth=2.1, label=label)
    axes[0].axvline(0, color="#222222", linewidth=1.0)
    style(axes[0], "Temperature residual distribution", "$T_{pred}-T_{BOPTEST}$ (degC)", "density")
    axes[0].legend(frameon=False)
    for x in [0.5, 1.0, 1.5]:
        axes[1].axvline(x, color="#777777", linestyle="--", linewidth=0.8)
    style(axes[1], "Absolute-error CDF with engineering thresholds", "$|T_{pred}-T_{BOPTEST}|$ (degC)", "fraction below threshold")
    axes[1].set_xlim(0, 5.0)
    axes[1].set_ylim(0, 1.01)
    inset = inset_axes(axes[0], width="42%", height="42%", loc="upper left", borderpad=1.2)
    tq, oq = normal_qq_points(residuals["calibrated v3.5"])
    inset.scatter(tq, oq, s=4, color=TEAL, alpha=0.45)
    lim = max(abs(tq).max(), abs(oq).max())
    inset.plot([-lim, lim], [-lim, lim], color="#333333", linewidth=1.0)
    inset.set_title("Q-Q: calibrated", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(True, color="#eeeeee", linewidth=0.6)
    fig.suptitle("Residual distribution, tolerance CDF, and calibrated-v3.5 Q-Q diagnostic", fontsize=13, weight="bold")
    save(fig, "block1_q1_polish_residual_distribution_qq_cdf")


def fig_matched_waterfall_accent() -> None:
    start = 1.5572
    mid = 0.8761
    end = 0.6441
    corpus = start - mid
    calib = mid - end
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    ax.bar([0], [start], color=BLUE, width=0.56, edgecolor="#222222", linewidth=0.5)
    ax.bar([1], [-corpus], bottom=[start], color="#5ba3d0", width=0.56, edgecolor="#222222", linewidth=0.5)
    ax.bar([2], [-calib], bottom=[mid], color=TEAL, width=0.56, edgecolor="#222222", linewidth=0.5)
    ax.bar([3], [end], color=PURPLE, width=0.56, edgecolor="#222222", linewidth=0.5)
    ax.annotate(
        "-0.681 degC\ncorpus shift",
        xy=(1, mid + corpus / 2),
        xytext=(1.35, 1.38),
        arrowprops=dict(arrowstyle="->", color="#5ba3d0", linewidth=1.5),
        fontsize=9,
        color="#24536d",
        weight="bold",
    )
    ax.annotate(
        "-0.232 degC\nStage A/B/C",
        xy=(2, end + calib / 2),
        xytext=(2.35, 0.98),
        arrowprops=dict(arrowstyle="->", color=TEAL, linewidth=1.5),
        fontsize=9,
        color="#135f55",
        weight="bold",
    )
    ax.plot([0.28, 0.72], [start, start], color="#777777", linewidth=1)
    ax.plot([1.28, 1.72], [mid, mid], color="#777777", linewidth=1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(
        [
            "v3 hourly\n1.557 degC",
            "15-min corpus\n74.6% of gain",
            "Stage A/B/C\n25.4% of gain",
            "calibrated v3.5\n0.644 degC",
        ]
    )
    style(ax, "Matched-corpus attribution of the predictive-fidelity gain", ylabel="24h RMSE$_T$ (degC)")
    ax.text(
        0.02,
        0.95,
        "Interpretation: calibration improves fidelity, but the claim is bounded by a corpus-controlled decomposition.",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        va="top",
    )
    save(fig, "block1_q1_polish_matched_waterfall")


def main() -> None:
    fig_czon_prior_band()
    fig_horizon_ci()
    fig_residual_distribution_qq()
    fig_matched_waterfall_accent()
    print("Wrote Q1-polished Block 1 figures to", OUT)


if __name__ == "__main__":
    main()
