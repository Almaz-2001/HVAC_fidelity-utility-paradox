from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_C = 22.0
T_LOW = 21.0
T_HIGH = 25.0
BUILDING_AREA_M2 = 48.0
SCENARIO_ORDER = [
    "Jan_Winter",
    "Feb_Winter",
    "Mar_Spring",
    "Apr_Spring",
    "May_Spring",
    "Jun_Summer",
    "Jul_Summer",
    "Aug_Summer",
    "Sep_Autumn",
    "Oct_Autumn",
    "Nov_Autumn",
    "Dec_Winter",
]
SUMMER_SCENARIOS = {"Jun_Summer", "Jul_Summer", "Aug_Summer"}


@dataclass(frozen=True)
class ControllerSpec:
    key: str
    label: str
    pattern: str
    temp_col: str
    power_col: str
    color: str


CONTROLLERS = [
    ControllerSpec(
        key="standard_pi",
        label="Standard PI",
        pattern="standard_controller_scenario_*.csv",
        temp_col="t_zone",
        power_col="p_total",
        color="#4c78a8",
    ),
    ControllerSpec(
        key="thermostatic",
        label="Thermostatic PPO",
        pattern="thermostatic_scenario_*.csv",
        temp_col="t_zone",
        power_col="p_total",
        color="#f58518",
    ),
    ControllerSpec(
        key="hdrl",
        label="HDRL",
        pattern="hdrl_scenario_*.csv",
        temp_col="temp",
        power_col="power",
        color="#54a24b",
    ),
]


def _extract_scenario_name(path: Path, spec: ControllerSpec) -> str:
    prefix = spec.pattern.replace("*", "")
    return path.stem.replace(prefix.replace(".csv", ""), "")


def _compute_fixed_metrics(temps: np.ndarray) -> dict[str, float]:
    errors = np.abs(temps - TARGET_C)
    r_time = float(np.mean((temps < T_LOW) | (temps > T_HIGH)))
    over = float(np.maximum((temps - T_HIGH) / T_HIGH, 0.0).max())
    under = float(np.maximum((T_LOW - temps) / T_LOW, 0.0).max())
    return {
        "rmse22": float(np.sqrt(np.mean((temps - TARGET_C) ** 2))),
        "mae22": float(np.mean(errors)),
        "within_1c_pct": float(np.mean(errors < 1.0) * 100.0),
        "within_05c_pct": float(np.mean(errors < 0.5) * 100.0),
        "viol_21_25_pct": float(r_time * 100.0),
        "m_s_fixed": float(r_time + max(over, under)),
        "t_min": float(temps.min()),
        "t_max": float(temps.max()),
        "t_mean": float(temps.mean()),
    }


def load_controller_metrics(outputs_dir: Path, spec: ControllerSpec) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    paths = sorted(outputs_dir.glob(spec.pattern), key=lambda p: SCENARIO_ORDER.index(_extract_scenario_name(p, spec)))
    if not paths:
        raise FileNotFoundError(f"No files found for {spec.label}: {outputs_dir / spec.pattern}")

    for path in paths:
        scenario = _extract_scenario_name(path, spec)
        df = pd.read_csv(path)
        temps = df[spec.temp_col].to_numpy(dtype=float)
        power = df[spec.power_col].to_numpy(dtype=float)
        metrics = _compute_fixed_metrics(temps)
        metrics.update(
            {
                "controller_key": spec.key,
                "controller": spec.label,
                "scenario": scenario,
                "energy_kwh": float(power.sum() / 1000.0),
                "energy_kwh_m2": float(power.sum() / 1000.0 / BUILDING_AREA_M2),
            }
        )
        rows.append(metrics)

    out = pd.DataFrame(rows)
    out["scenario"] = pd.Categorical(out["scenario"], categories=SCENARIO_ORDER, ordered=True)
    return out.sort_values("scenario").reset_index(drop=True)


def aggregate_metrics(scenario_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        scenario_df.groupby(["controller_key", "controller"], as_index=False)
        .agg(
            rmse22_mean=("rmse22", "mean"),
            mae22_mean=("mae22", "mean"),
            within_1c_mean=("within_1c_pct", "mean"),
            within_05c_mean=("within_05c_pct", "mean"),
            viol_21_25_mean=("viol_21_25_pct", "mean"),
            ms_fixed_mean=("m_s_fixed", "mean"),
            energy_total_kwh=("energy_kwh", "sum"),
            energy_mean_kwh=("energy_kwh", "mean"),
            energy_total_kwh_m2=("energy_kwh_m2", "sum"),
        )
    )
    order = {spec.key: idx for idx, spec in enumerate(CONTROLLERS)}
    grouped["sort_key"] = grouped["controller_key"].map(order)
    grouped = grouped.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)
    return grouped


def build_conclusion(summary_df: pd.DataFrame, scenario_df: pd.DataFrame) -> str:
    def row(key: str) -> pd.Series:
        return summary_df.loc[summary_df["controller_key"] == key].iloc[0]

    pi = row("standard_pi")
    thermo = row("thermostatic")
    hdrl = row("hdrl")

    summer = scenario_df[scenario_df["scenario"].isin(SUMMER_SCENARIOS)]
    thermo_summer = float(summer.loc[summer["controller_key"] == "thermostatic", "energy_kwh"].sum())
    hdrl_summer = float(summer.loc[summer["controller_key"] == "hdrl", "energy_kwh"].sum())
    hdrl_summer_gain = 100.0 * (thermo_summer - hdrl_summer) / max(thermo_summer, 1e-6)

    thermo_vs_pi = 100.0 * (thermo["energy_total_kwh"] - pi["energy_total_kwh"]) / max(pi["energy_total_kwh"], 1e-6)
    hdrl_vs_pi = 100.0 * (hdrl["energy_total_kwh"] - pi["energy_total_kwh"]) / max(pi["energy_total_kwh"], 1e-6)
    hdrl_vs_thermo = 100.0 * (hdrl["energy_total_kwh"] - thermo["energy_total_kwh"]) / max(thermo["energy_total_kwh"], 1e-6)

    lines = [
        "FINAL CONCLUSION",
        "================",
        "",
        (
            f"1. Best fixed-target comfort: Thermostatic PPO "
            f"(mean RMSE22 = {thermo['rmse22_mean']:.3f} C, "
            f"mean m_s = {thermo['ms_fixed_mean']:.3f})."
        ),
        (
            f"2. Best energy efficiency: Standard PI "
            f"(total energy = {pi['energy_total_kwh']:.1f} kWh, "
            f"{pi['energy_total_kwh_m2']:.2f} kWh/m2)."
        ),
        (
            f"3. Best fixed-band violation rate among the two RL controllers: HDRL "
            f"(mean violation = {hdrl['viol_21_25_mean']:.1f}% vs "
            f"{thermo['viol_21_25_mean']:.1f}% for Thermostatic PPO), "
            f"but HDRL still has worse mean RMSE22 ({hdrl['rmse22_mean']:.3f} C)."
        ),
        (
            f"4. Annual trade-off status: HDRL does not yet beat Thermostatic PPO "
            f"on the yearly benchmark. Its annual energy is {hdrl_vs_thermo:+.1f}% "
            f"relative to Thermostatic PPO."
        ),
        (
            f"5. Seasonal nuance: HDRL reduces summer energy by {hdrl_summer_gain:.1f}% "
            f"relative to Thermostatic PPO across Jun-Aug, but that gain is not enough "
            f"to offset weaker non-summer performance."
        ),
        (
            f"6. Relative to the Standard PI baseline, both RL controllers remain much more "
            f"energy-intensive: Thermostatic PPO is {thermo_vs_pi:+.1f}% and HDRL is "
            f"{hdrl_vs_pi:+.1f}% versus PI."
        ),
        "",
        "Recommended interpretation:",
        "- Standard PI = low-energy reference baseline.",
        "- Thermostatic PPO = strongest comfort reference baseline.",
        "- HDRL = structured trade-off controller that is promising but not yet dominant annually.",
    ]
    return "\n".join(lines)


def plot_dashboard(summary_df: pd.DataFrame, artifact_dir: Path) -> None:
    labels = summary_df["controller"].tolist()
    color_map = {spec.key: spec.color for spec in CONTROLLERS}
    colors = [color_map[key] for key in summary_df["controller_key"]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Three-Model Comparison Dashboard", fontsize=16)

    metrics = [
        ("rmse22_mean", "Mean RMSE22 [C]", axes[0, 0]),
        ("viol_21_25_mean", "Mean Viol. 21-25 [%]", axes[0, 1]),
        ("ms_fixed_mean", "Mean Fixed-Band m_s", axes[1, 0]),
        ("energy_total_kwh", "Total Energy [kWh]", axes[1, 1]),
    ]

    for metric, ylabel, ax in metrics:
        values = summary_df[metric].to_numpy(dtype=float)
        ax.bar(labels, values, color=colors)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
        for idx, val in enumerate(values):
            ax.text(idx, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(artifact_dir / "comparison_dashboard.png", dpi=180)
    plt.close(fig)


def plot_monthly_metric(scenario_df: pd.DataFrame, metric: str, ylabel: str, filename: str, artifact_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for spec in CONTROLLERS:
        subset = scenario_df[scenario_df["controller_key"] == spec.key]
        ax.plot(
            subset["scenario"].astype(str),
            subset[metric].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            label=spec.label,
            color=spec.color,
        )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Scenario")
    ax.set_title(ylabel + " by Scenario")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifact_dir / filename, dpi=180)
    plt.close(fig)


def plot_tradeoff(summary_df: pd.DataFrame, artifact_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for spec in CONTROLLERS:
        row = summary_df.loc[summary_df["controller_key"] == spec.key].iloc[0]
        ax.scatter(
            row["energy_total_kwh"],
            row["rmse22_mean"],
            s=130,
            color=spec.color,
            label=spec.label,
        )
        ax.text(
            row["energy_total_kwh"],
            row["rmse22_mean"],
            f"  {spec.label}",
            va="center",
            fontsize=10,
        )
    ax.set_xlabel("Total Energy [kWh]")
    ax.set_ylabel("Mean RMSE22 [C]")
    ax.set_title("Comfort-Energy Trade-off Across Controllers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifact_dir / "comparison_tradeoff_scatter.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual comparison of the three validated HVAC controllers")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--artifact_dir", default=None)
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else outputs_dir / "three_model_comparison"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    scenario_frames = [load_controller_metrics(outputs_dir, spec) for spec in CONTROLLERS]
    scenario_df = pd.concat(scenario_frames, ignore_index=True)
    summary_df = aggregate_metrics(scenario_df)

    scenario_df.to_csv(artifact_dir / "comparison_scenario_metrics.csv", index=False)
    summary_df.to_csv(artifact_dir / "comparison_summary.csv", index=False)

    plot_dashboard(summary_df, artifact_dir)
    plot_monthly_metric(scenario_df, "rmse22", "RMSE22 [C]", "comparison_monthly_rmse22.png", artifact_dir)
    plot_monthly_metric(scenario_df, "energy_kwh", "Energy [kWh]", "comparison_monthly_energy.png", artifact_dir)
    plot_tradeoff(summary_df, artifact_dir)

    conclusion = build_conclusion(summary_df, scenario_df)
    conclusion_path = artifact_dir / "comparison_conclusion.txt"
    conclusion_path.write_text(conclusion, encoding="utf-8")

    print(conclusion)
    print(f"\nSaved artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
