from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[SKIP] Missing: {path}")
        return None
    return pd.read_csv(path)


def build_tables(outputs_dir: Path, seed: int, artifact_dir: Path) -> None:
    rows = []

    standard = _load_csv(outputs_dir / "standard_controller_yearly_summary.csv")
    if standard is not None and not standard.empty:
        rows.append(
            {
                "controller": "standard_boptest",
                "metric_1": "schedule_ms_mean",
                "value_1": standard["schedule_m_s"].dropna().mean(),
                "metric_2": "energy_mean",
                "value_2": standard["energy_kwh"].dropna().mean(),
            }
        )

    thermo = _load_csv(outputs_dir / "thermostatic_yearly_summary.csv")
    if thermo is not None and not thermo.empty:
        rows.append(
            {
                "controller": "thermostatic",
                "metric_1": "rmse_mean",
                "value_1": thermo["rmse"].mean(),
                "metric_2": "energy_mean",
                "value_2": thermo["energy"].mean(),
            }
        )

    hdrl = _load_csv(outputs_dir / "hdrl_yearly_summary.csv")
    if hdrl is not None and not hdrl.empty:
        rows.append(
            {
                "controller": "hdrl",
                "metric_1": "viol_mean",
                "value_1": hdrl["viol"].dropna().mean(),
                "metric_2": "ms_mean",
                "value_2": hdrl["ms"].dropna().mean(),
            }
        )

    morl_eval = _load_csv(outputs_dir / f"seed{seed}" / "eval" / "ppo_eval.csv")
    if morl_eval is not None and not morl_eval.empty:
        rows.append(
            {
                "controller": "morl",
                "metric_1": "comfort_mean",
                "value_1": morl_eval["comfort"].dropna().mean(),
                "metric_2": "hvac_power_mean",
                "value_2": morl_eval["hvac_power"].dropna().mean(),
            }
        )

    safe_summary = _load_csv(outputs_dir / "eval_safe_morl" / "summary.csv")
    if safe_summary is not None and not safe_summary.empty:
        rows.append(
            {
                "controller": "safe_morl",
                "metric_1": "m_s",
                "value_1": safe_summary.loc[0, "m_s"],
                "metric_2": "violation_pct",
                "value_2": safe_summary.loc[0, "violation_pct"],
            }
        )

    if rows:
        pd.DataFrame(rows).to_csv(artifact_dir / "paper_summary_table.csv", index=False)


def plot_thermostatic(outputs_dir: Path, artifact_dir: Path) -> None:
    summary = _load_csv(outputs_dir / "thermostatic_yearly_summary.csv")
    if summary is None or summary.empty:
        return

    plt.figure(figsize=(10, 4))
    plt.bar(summary["name"], summary["rmse"], color="#2f6db2")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RMSE [C]")
    plt.title("Thermostatic Baseline: Scenario-wise RMSE")
    plt.tight_layout()
    plt.savefig(artifact_dir / "fig_thermostatic_rmse.png", dpi=160)
    plt.close()


def plot_hdrl(outputs_dir: Path, artifact_dir: Path) -> None:
    summary = _load_csv(outputs_dir / "hdrl_yearly_summary.csv")
    if summary is None or summary.empty:
        return

    valid = summary.dropna(subset=["energy", "ms"])
    if valid.empty:
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(valid["energy"], valid["ms"], s=70, color="#d95f02")
    for _, row in valid.iterrows():
        plt.text(row["energy"], row["ms"], row["name"], fontsize=8)
    plt.xlabel("Energy [kWh]")
    plt.ylabel("m_s")
    plt.title("HDRL: Energy vs Safety")
    plt.tight_layout()
    plt.savefig(artifact_dir / "fig_hdrl_energy_vs_ms.png", dpi=160)
    plt.close()


def plot_morl(outputs_dir: Path, seed: int, artifact_dir: Path) -> None:
    log_df = _load_csv(outputs_dir / f"seed{seed}" / "morl_log.csv")
    if log_df is None or log_df.empty:
        return

    rolled = log_df.copy()
    rolled["comfort_roll"] = rolled["comfort"].rolling(200, min_periods=1).mean()
    rolled["energy_roll"] = rolled["energy"].rolling(200, min_periods=1).mean()

    plt.figure(figsize=(10, 4))
    plt.plot(rolled["step"], rolled["comfort_roll"], label="comfort", color="#1b9e77")
    plt.plot(rolled["step"], rolled["energy_roll"], label="energy", color="#7570b3")
    plt.xlabel("Step")
    plt.ylabel("Rolling mean")
    plt.title("MORL Training Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact_dir / "fig_morl_training_tradeoff.png", dpi=160)
    plt.close()


def plot_safe_morl(outputs_dir: Path, artifact_dir: Path) -> None:
    safe_df = _load_csv(outputs_dir / "eval_safe_morl" / "eval_safe_morl.csv")
    if safe_df is None or safe_df.empty:
        return

    head = safe_df.head(500)
    plt.figure(figsize=(10, 4))
    plt.plot(head["step"], head["t_zone"], color="#e7298a", label="T_zone")
    fallback = head[head["source"] != "ppo"]
    if not fallback.empty:
        plt.scatter(fallback["step"], fallback["t_zone"], color="#000000", s=10, label="fallback")
    plt.xlabel("Step")
    plt.ylabel("Zone temperature [C]")
    plt.title("Safe MORL Intervention Timeline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact_dir / "fig_safe_morl_timeline.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Build paper-ready tables and plots from evaluation CSVs")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact_dir", default=None)
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else outputs_dir / "paper_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    build_tables(outputs_dir, args.seed, artifact_dir)
    plot_thermostatic(outputs_dir, artifact_dir)
    plot_hdrl(outputs_dir, artifact_dir)
    plot_morl(outputs_dir, args.seed, artifact_dir)
    plot_safe_morl(outputs_dir, artifact_dir)

    print(f"[DONE] Paper artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
