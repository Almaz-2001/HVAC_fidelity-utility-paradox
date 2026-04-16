from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "figures" / "current_surrogate"

ROLLOUT_CSV = REPO_ROOT / "outputs" / "surrogate_v35_rollout_live" / "calibrated_v35" / "all_full_rollouts.csv"
ROLLOUT_SUMMARY_CSV = REPO_ROOT / "outputs" / "surrogate_v35_rollout_live" / "v35_compare_summary.csv"
INVERSE_CSV = REPO_ROOT / "outputs" / "surrogate_v35_inverse_boptest_prior420_heads_only" / "calibration_predictions.csv"
INVERSE_SUMMARY_JSON = REPO_ROOT / "outputs" / "surrogate_v35_inverse_boptest_prior420_heads_only" / "calibration_summary_boptest_v35.json"


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _bias(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a - b))


def export_comfort_trace() -> Path:
    df = pd.read_csv(ROLLOUT_CSV)
    summary = pd.read_csv(ROLLOUT_SUMMARY_CSV)
    cal_summary = summary.loc[summary["variant"] == "calibrated_v35"].iloc[0]

    selected = []
    for season in ["winter", "spring", "summer", "autumn"]:
        season_df = df[(df["season"] == season) & (df["policy"] == "mixed")]
        if season_df.empty:
            season_df = df[df["season"] == season]
        if season_df.empty:
            continue
        episode_id = season_df["episode_id"].iloc[0]
        selected.append((season, season_df[season_df["episode_id"] == episode_id].copy()))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
    axes = axes.flatten()

    for ax, (season, season_df) in zip(axes, selected):
        ax.fill_between(
            season_df["step"].to_numpy(),
            21.0,
            25.0,
            color="#dff3e4",
            alpha=0.8,
            label="Comfort band 21-25C",
        )
        ax.plot(
            season_df["step"],
            season_df["actual_t_zone"],
            color="#c0392b",
            linewidth=2.0,
            label="BOPTEST",
        )
        ax.plot(
            season_df["step"],
            season_df["pred_t_zone"],
            color="#2563eb",
            linewidth=2.0,
            label="Current surrogate v3.5",
        )
        rmse = _rmse(season_df["pred_t_zone"].to_numpy(), season_df["actual_t_zone"].to_numpy())
        bias = _bias(season_df["pred_t_zone"].to_numpy(), season_df["actual_t_zone"].to_numpy())
        ax.set_title(f"{season.capitalize()} mixed trace")
        ax.set_xlabel("Step (hours)")
        ax.set_ylabel("Zone temperature, C")
        ax.grid(alpha=0.25)
        ax.text(
            0.02,
            0.03,
            f"RMSE={rmse:.2f} C\nBias={bias:+.2f} C",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        "Current surrogate comfort trace vs BOPTEST\n"
        f"calibrated_v35 one-step RMSE={cal_summary['one_step_rmse_c']:.3f} C, "
        f"24h RMSE={cal_summary['longest_horizon_rmse_c']:.3f} C, "
        f"C_zon error={cal_summary['czon_error_pct']:.2f}%",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out_path = OUT_DIR / "current_v35_comfort_trace_vs_boptest.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_inverse_problem_figure() -> Path:
    df = pd.read_csv(INVERSE_CSV)
    summary = json.loads(INVERSE_SUMMARY_JSON.read_text(encoding="utf-8"))

    n_plot = min(240, len(df))
    view = df.iloc[:n_plot].copy()

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

    axes[0].plot(view["step"], view["t_target_used"], color="#c0392b", linewidth=2.0, label="BOPTEST target / clean trace")
    axes[0].plot(view["step"], view["t_pred_before"], color="#7f8c8d", linewidth=1.7, label="Before calibration")
    axes[0].plot(view["step"], view["t_pred_after"], color="#2563eb", linewidth=2.0, label="After calibration")
    axes[0].plot(view["step"], view["t_pred_surrogate_v35"], color="#16a085", linewidth=1.6, linestyle="--", label="Structural v3.5 core")
    axes[0].set_ylabel("Temperature, C")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].set_title("Inverse problem result on current v3.5 heads-only calibration")

    axes[1].plot(view["step"], view["p_target_used"], color="#c0392b", linewidth=2.0, label="BOPTEST target power")
    axes[1].plot(view["step"], view["p_pred_before"], color="#7f8c8d", linewidth=1.7, label="Before calibration")
    axes[1].plot(view["step"], view["p_pred_after"], color="#2563eb", linewidth=2.0, label="After calibration")
    axes[1].plot(view["step"], view["p_pred_surrogate_v35"], color="#16a085", linewidth=1.6, linestyle="--", label="Structural v3.5 core")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Power, W")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right", frameon=False)

    fig.suptitle(
        "Current inverse problem snapshot\n"
        f"RMSE {summary['baseline_rmse_c']:.4f} -> {summary['calibrated_rmse_c']:.4f} C, "
        f"C_zon={summary['c_zon_final_j_per_k']:.0f} J/K, "
        f"C_zon error={summary['czon_error_pct']:.2f}%",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = OUT_DIR / "current_v35_inverse_problem_result.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    _ensure_out_dir()
    comfort = export_comfort_trace()
    inverse = export_inverse_problem_figure()
    print("EXPORTED CURRENT SURROGATE FIGURES")
    print(f"Comfort trace: {comfort}")
    print(f"Inverse problem: {inverse}")


if __name__ == "__main__":
    main()
