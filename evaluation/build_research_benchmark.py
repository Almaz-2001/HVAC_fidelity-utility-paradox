from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "results" / "research_benchmark"
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"
MANIFEST_DIR = OUT_ROOT / "manifests"

CONTROLLER_SUMMARY_CSV = REPO_ROOT / "outputs" / "three_model_comparison" / "comparison_summary.csv"
SURROGATE_COMPARE_CSV = REPO_ROOT / "outputs" / "surrogate_v35_rollout_live" / "v35_compare_summary.csv"
RAW_HORIZON_CSV = REPO_ROOT / "outputs" / "surrogate_v35_rollout_live" / "raw_v35" / "horizon_metrics.csv"
CAL_HORIZON_CSV = REPO_ROOT / "outputs" / "surrogate_v35_rollout_live" / "calibrated_v35" / "horizon_metrics.csv"
INVERSE_SUMMARY_JSON = (
    REPO_ROOT / "outputs" / "surrogate_v35_inverse_boptest_prior420_heads_only" / "calibration_summary_boptest_v35.json"
)
MULTISTART_SUMMARY_CSV = REPO_ROOT / "outputs" / "surrogate_v35_inverse_boptest_multistart" / "multistart_summary_v35.csv"
MORL_PROGRESS_CSV = REPO_ROOT / "outputs" / "eval_multi_seed" / "summary.csv"


def ensure_dirs() -> None:
    for path in (FIG_DIR, TABLE_DIR, MANIFEST_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def load_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def extract_horizon_rmse(horizon_df: pd.DataFrame | None, horizon_h: int) -> float:
    if horizon_df is None or horizon_df.empty:
        return float("nan")
    match = horizon_df.loc[horizon_df["horizon_h"] == horizon_h]
    if match.empty:
        return float("nan")
    return float(match.iloc[0]["temp_rmse_c"])


def build_surrogate_rollout_table(
    compare_df: pd.DataFrame,
    raw_horizon_df: pd.DataFrame | None,
    calibrated_horizon_df: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    horizon_by_variant = {
        "raw_v35": raw_horizon_df,
        "calibrated_v35": calibrated_horizon_df,
    }

    for row in compare_df.to_dict(orient="records"):
        variant = str(row["variant"])
        horizon_df = horizon_by_variant.get(variant)
        rows.append(
            {
                "surrogate_variant": variant,
                "one_step_rmse_c": float(row["one_step_rmse_c"]),
                "rollout_4h_rmse_c": extract_horizon_rmse(horizon_df, 4),
                "rollout_8h_rmse_c": extract_horizon_rmse(horizon_df, 8),
                "rollout_24h_rmse_c": extract_horizon_rmse(horizon_df, 24),
                "mean_episode_rmse_c": float(row["mean_episode_rmse_c"]),
                "mean_episode_bias_c": float(row["mean_episode_bias_c"]),
                "mean_episode_power_rmse_w": float(row["mean_episode_power_rmse_w"]),
                "c_zon_final_j_per_k": float(row["c_zon_final_j_per_k"]),
                "c_zon_error_pct": float(row["czon_error_pct"]),
                "stage_c_mode": str(row["stage_c_mode"]),
            }
        )

    return pd.DataFrame(rows)


def build_inverse_table(inverse_summary: dict[str, Any], multistart_df: pd.DataFrame) -> pd.DataFrame:
    best_multistart = multistart_df.loc[multistart_df["czon_error_pct"].astype(float).idxmin()]

    rows = [
        {
            "experiment": "v35_heads_only",
            "baseline_rmse_c": float(inverse_summary["baseline_rmse_c"]),
            "calibrated_rmse_c": float(inverse_summary["calibrated_rmse_c"]),
            "improvement_rmse_pct": float(inverse_summary["improvement_rmse_pct"]),
            "c_zon_final_j_per_k": float(inverse_summary["c_zon_final_j_per_k"]),
            "c_zon_error_pct": float(inverse_summary["czon_error_pct"]),
            "stage_c_mode": str(inverse_summary["stage_c_mode"]),
            "excitation_mode": str(inverse_summary.get("excitation_summary", {}).get("excitation_mode", "")),
            "rows_train_selected": int(inverse_summary.get("excitation_summary", {}).get("rows_train_selected", 0)),
            "status": "implemented",
        },
        {
            "experiment": "v35_multistart_best_prior",
            "baseline_rmse_c": np.nan,
            "calibrated_rmse_c": float(best_multistart["calibrated_rmse_c"]),
            "improvement_rmse_pct": np.nan,
            "c_zon_final_j_per_k": float(best_multistart["c_zon_final_j_per_k"]),
            "c_zon_error_pct": float(best_multistart["czon_error_pct"]),
            "stage_c_mode": str(best_multistart.get("stage_c_mode", "")),
            "excitation_mode": "",
            "rows_train_selected": np.nan,
            "status": "implemented",
        },
    ]
    return pd.DataFrame(rows)


def build_controller_table(controller_df: pd.DataFrame, morl_progress_df: pd.DataFrame | None) -> pd.DataFrame:
    base = controller_df.copy()
    base["controller_family"] = ["PI", "Thermostatic PPO", "HDRL"]
    base["training_env"] = ["BOPTEST/native controller", "Surrogate", "Surrogate"]
    base["benchmark_status"] = "implemented"

    columns = [
        "controller_key",
        "controller_family",
        "training_env",
        "rmse22_mean",
        "mae22_mean",
        "within_1c_mean",
        "within_05c_mean",
        "viol_21_25_mean",
        "ms_fixed_mean",
        "energy_total_kwh",
        "energy_mean_kwh",
        "benchmark_status",
    ]
    out = base[columns].copy()

    if morl_progress_df is not None and not morl_progress_df.empty:
        row0 = morl_progress_df.iloc[0]
        morl_row = {
            "controller_key": "morl_safe_layer",
            "controller_family": "MORL/PPO + safety layer",
            "training_env": "Surrogate -> BOPTEST pipeline",
            "rmse22_mean": np.nan,
            "mae22_mean": np.nan,
            "within_1c_mean": np.nan,
            "within_05c_mean": np.nan,
            "viol_21_25_mean": float(row0["ppo_sf_viol_mean"]),
            "ms_fixed_mean": float(row0["ppo_sf_ms_mean"]),
            "energy_total_kwh": np.nan,
            "energy_mean_kwh": np.nan,
            "benchmark_status": "experimental",
        }
        out = pd.concat([out, pd.DataFrame([morl_row])], ignore_index=True)

    return out


def build_experiment_matrix(
    controller_table: pd.DataFrame,
    surrogate_rollout_table: pd.DataFrame,
    inverse_table: pd.DataFrame,
) -> pd.DataFrame:
    morl_status = "experimental" if "morl_safe_layer" in controller_table["controller_key"].values else "pipeline_ready"
    rows = [
        {
            "pillar": "digital_twin",
            "experiment": "Surrogate raw vs calibrated rollout realism",
            "status": "implemented",
            "evidence": "surrogate_rollout_benchmark.csv",
        },
        {
            "pillar": "inverse_problem",
            "experiment": "Stage A/B/C inverse calibration with explicit C_zon",
            "status": "implemented",
            "evidence": "inverse_calibration_benchmark.csv",
        },
        {
            "pillar": "controller_benchmark",
            "experiment": "PI vs Thermostatic PPO vs HDRL on unified BOPTEST benchmark",
            "status": "implemented",
            "evidence": "controller_benchmark.csv",
        },
        {
            "pillar": "morl_pipeline",
            "experiment": "Surrogate-pretrained MORL/PPO transfer to BOPTEST",
            "status": morl_status,
            "evidence": "controller_benchmark.csv",
        },
        {
            "pillar": "downstream_control_alignment",
            "experiment": "Effect of surrogate quality on downstream controller performance",
            "status": "pending",
            "evidence": "requires retraining/evaluating controllers on calibrated twin and a v3.5 backend adapter",
        },
    ]
    return pd.DataFrame(rows)


def plot_surrogate_dashboard(rollout_df: pd.DataFrame, inverse_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    plot_df = rollout_df.copy()
    x = np.arange(len(plot_df))
    width = 0.18

    axes[0].bar(x - 1.5 * width, plot_df["one_step_rmse_c"], width=width, label="1-step RMSE")
    axes[0].bar(x - 0.5 * width, plot_df["rollout_4h_rmse_c"], width=width, label="4h RMSE")
    axes[0].bar(x + 0.5 * width, plot_df["rollout_8h_rmse_c"], width=width, label="8h RMSE")
    axes[0].bar(x + 1.5 * width, plot_df["rollout_24h_rmse_c"], width=width, label="24h RMSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["surrogate_variant"])
    axes[0].set_ylabel("Temperature RMSE, C")
    axes[0].set_title("Surrogate rollout realism")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(plot_df["surrogate_variant"], plot_df["mean_episode_power_rmse_w"], color=["#7f8c8d", "#2563eb"])
    axes[1].set_ylabel("Power RMSE, W")
    axes[1].set_title("Power realism")
    axes[1].grid(axis="y", alpha=0.25)

    inv = inverse_df.loc[inverse_df["experiment"] == "v35_heads_only"].iloc[0]
    axes[2].bar(["baseline", "calibrated"], [inv["baseline_rmse_c"], inv["calibrated_rmse_c"]], color=["#7f8c8d", "#2563eb"])
    axes[2].set_title("Inverse calibration fit")
    axes[2].set_ylabel("RMSE, C")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].text(
        0.02,
        0.95,
        f"C_zon error={inv['c_zon_error_pct']:.2f}%\nStage C={inv['stage_c_mode']}",
        transform=axes[2].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    fig.suptitle("Calibrated surrogate benchmark snapshot", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = FIG_DIR / "surrogate_fidelity_dashboard.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_controller_dashboard(controller_df: pd.DataFrame) -> tuple[Path, Path]:
    plot_df = controller_df[controller_df["benchmark_status"] == "implemented"].copy()
    labels = plot_df["controller_family"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].bar(x, plot_df["rmse22_mean"], color=["#7f8c8d", "#2563eb", "#16a085"])
    axes[0].set_title("Comfort RMSE")
    axes[0].set_ylabel("RMSE around 22 C")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, plot_df["ms_fixed_mean"], color=["#7f8c8d", "#2563eb", "#16a085"])
    axes[1].set_title("Safety metric")
    axes[1].set_ylabel("m_s")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(x, plot_df["energy_mean_kwh"], color=["#7f8c8d", "#2563eb", "#16a085"])
    axes[2].set_title("Energy per scenario")
    axes[2].set_ylabel("Mean energy, kWh")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, ha="right")
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Unified BOPTEST controller benchmark", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bar_path = FIG_DIR / "controller_benchmark_dashboard.png"
    fig.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(6.5, 5.5))
    scatter = ax.scatter(
        plot_df["energy_mean_kwh"],
        plot_df["rmse22_mean"],
        s=700 * plot_df["ms_fixed_mean"].to_numpy() + 120,
        c=["#7f8c8d", "#2563eb", "#16a085"],
        alpha=0.85,
    )
    for _, row in plot_df.iterrows():
        ax.text(row["energy_mean_kwh"] + 2.0, row["rmse22_mean"] + 0.03, row["controller_family"], fontsize=9)
    ax.set_xlabel("Mean energy per scenario, kWh")
    ax.set_ylabel("Mean RMSE around 22 C")
    ax.set_title("Trade-off view (bubble size = m_s)")
    ax.grid(alpha=0.25)
    tradeoff_path = FIG_DIR / "controller_tradeoff_scatter.png"
    fig2.tight_layout()
    fig2.savefig(tradeoff_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    return bar_path, tradeoff_path


def write_manifest(files: dict[str, str], extra: dict[str, Any]) -> Path:
    manifest = {
        "generated_files": files,
        "source_files": {
            "controller_summary_csv": str(CONTROLLER_SUMMARY_CSV.relative_to(REPO_ROOT)),
            "surrogate_compare_csv": str(SURROGATE_COMPARE_CSV.relative_to(REPO_ROOT)),
            "raw_horizon_csv": str(RAW_HORIZON_CSV.relative_to(REPO_ROOT)) if RAW_HORIZON_CSV.exists() else None,
            "calibrated_horizon_csv": str(CAL_HORIZON_CSV.relative_to(REPO_ROOT)) if CAL_HORIZON_CSV.exists() else None,
            "inverse_summary_json": str(INVERSE_SUMMARY_JSON.relative_to(REPO_ROOT)),
            "multistart_summary_csv": str(MULTISTART_SUMMARY_CSV.relative_to(REPO_ROOT)),
            "morl_progress_csv": str(MORL_PROGRESS_CSV.relative_to(REPO_ROOT)) if MORL_PROGRESS_CSV.exists() else None,
        },
        "notes": extra,
    }
    path = MANIFEST_DIR / "research_benchmark_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()

    controller_summary = load_required_csv(CONTROLLER_SUMMARY_CSV)
    surrogate_compare = load_required_csv(SURROGATE_COMPARE_CSV)
    raw_horizon = load_optional_csv(RAW_HORIZON_CSV)
    calibrated_horizon = load_optional_csv(CAL_HORIZON_CSV)
    inverse_summary = load_required_json(INVERSE_SUMMARY_JSON)
    multistart_summary = load_required_csv(MULTISTART_SUMMARY_CSV)
    morl_progress = pd.read_csv(MORL_PROGRESS_CSV) if MORL_PROGRESS_CSV.exists() else None

    surrogate_rollout_table = build_surrogate_rollout_table(
        surrogate_compare,
        raw_horizon_df=raw_horizon,
        calibrated_horizon_df=calibrated_horizon,
    )
    inverse_table = build_inverse_table(inverse_summary, multistart_summary)
    controller_table = build_controller_table(controller_summary, morl_progress)
    experiment_matrix = build_experiment_matrix(controller_table, surrogate_rollout_table, inverse_table)

    surrogate_rollout_path = TABLE_DIR / "surrogate_rollout_benchmark.csv"
    inverse_path = TABLE_DIR / "inverse_calibration_benchmark.csv"
    controller_path = TABLE_DIR / "controller_benchmark.csv"
    experiment_path = TABLE_DIR / "experiment_matrix.csv"
    surrogate_rollout_table.to_csv(surrogate_rollout_path, index=False)
    inverse_table.to_csv(inverse_path, index=False)
    controller_table.to_csv(controller_path, index=False)
    experiment_matrix.to_csv(experiment_path, index=False)

    surrogate_fig = plot_surrogate_dashboard(surrogate_rollout_table, inverse_table)
    controller_fig, tradeoff_fig = plot_controller_dashboard(controller_table)

    manifest_path = write_manifest(
        files={
            "surrogate_rollout_table": str(surrogate_rollout_path.relative_to(REPO_ROOT)),
            "inverse_table": str(inverse_path.relative_to(REPO_ROOT)),
            "controller_table": str(controller_path.relative_to(REPO_ROOT)),
            "experiment_matrix": str(experiment_path.relative_to(REPO_ROOT)),
            "surrogate_figure": str(surrogate_fig.relative_to(REPO_ROOT)),
            "controller_figure": str(controller_fig.relative_to(REPO_ROOT)),
            "tradeoff_figure": str(tradeoff_fig.relative_to(REPO_ROOT)),
        },
        extra={
            "implemented_controller_families": controller_table["controller_family"].tolist(),
            "downstream_control_alignment_status": "pending calibrated-surrogate controller retrain",
        },
    )

    print("RESEARCH BENCHMARK SNAPSHOT COMPLETE")
    print(f"Surrogate table: {surrogate_rollout_path}")
    print(f"Inverse table:   {inverse_path}")
    print(f"Controller table:{controller_path}")
    print(f"Experiment map:  {experiment_path}")
    print(f"Manifest:        {manifest_path}")


if __name__ == "__main__":
    main()
