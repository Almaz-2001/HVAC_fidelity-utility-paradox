from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"


def _read_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "controller" not in df.columns:
        df["controller"] = "thermostatic"
    return df


def _read_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time_days"] = df["sim_time_sec"] / 86400.0
    df["energy_kwh_cum"] = (df["p_total_w"] * (df["step"].diff().fillna(1) * 900.0) / 3600.0 / 1000.0).cumsum()
    return df


def build_comfort_plot(traces: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for ax, (scenario, trace) in zip(axes, traces.items()):
        ax.plot(trace["time_days"], trace["t_zone_c"], color="tab:red", linewidth=1.8, label="BOPTEST T_zone")
        ax.fill_between(trace["time_days"], 21.0, 24.0, color="#dff3e3", alpha=0.9, label="Comfort band 21-24 C")
        ax.plot(trace["time_days"], trace["t_amb_c"], color="tab:blue", linewidth=1.0, alpha=0.65, label="Ambient")
        ax.set_title(scenario.replace("_", " "))
        ax.set_ylabel("Temperature [C]")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time [days]")
    fig.suptitle("Hybrid-trained thermostatic controller on live BOPTEST")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_power_plot(traces: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    for row, (scenario, trace) in enumerate(traces.items()):
        axes[row, 0].plot(trace["time_days"], trace["p_total_w"], color="tab:orange", linewidth=1.5)
        axes[row, 0].set_title(f"{scenario.replace('_', ' ')} power")
        axes[row, 0].set_ylabel("HVAC power [W]")
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].plot(trace["time_days"], trace["energy_kwh_cum"], color="tab:green", linewidth=1.8)
        axes[row, 1].set_title(f"{scenario.replace('_', ' ')} cumulative energy")
        axes[row, 1].set_ylabel("Energy [kWh]")
        axes[row, 1].grid(alpha=0.25)

    axes[1, 0].set_xlabel("Time [days]")
    axes[1, 1].set_xlabel("Time [days]")
    fig.suptitle("Hybrid-trained thermostatic controller energy traces on live BOPTEST")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_bar_plot(df: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: Path) -> None:
    pivot = df.pivot(index="scenario", columns="controller", values=metric).loc[
        ["peak_heat_window", "typical_heat_window"]
    ]
    ax = pivot.plot(kind="bar", figsize=(10, 6), color=["#4e79a7", "#e15759"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def build_energy_bar_plot(df: pd.DataFrame, out_path: Path) -> None:
    build_bar_plot(
        df=df,
        metric="energy_kwh",
        ylabel="Energy [kWh]",
        title="Hybrid thermostatic vs PI: 14-day energy",
        out_path=out_path,
    )


def write_report(
    out_path: Path,
    hybrid_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    v3_df: pd.DataFrame,
    warm_df: pd.DataFrame,
) -> None:
    peak_h = hybrid_df[hybrid_df["scenario"] == "peak_heat_window"].iloc[0]
    typ_h = hybrid_df[hybrid_df["scenario"] == "typical_heat_window"].iloc[0]
    peak_v3 = v3_df[v3_df["scenario"] == "peak_heat_window"].iloc[0]
    typ_v3 = v3_df[v3_df["scenario"] == "typical_heat_window"].iloc[0]
    peak_w = warm_df[warm_df["scenario"] == "peak_heat_window"].iloc[0]
    typ_w = warm_df[warm_df["scenario"] == "typical_heat_window"].iloc[0]

    report = f"""# Block 2 Hybrid Surrogate Snapshot

Date: 2026-04-28

## Canonical Hybrid Setup

- training dynamics: `v3`
- physics regularizer: `v3.5 disagreement penalty`
- controller family: `thermostatic PPO`
- comfort band: `21-24 C`
- step size: `900 s`
- `lambda_temp_disagree = 0.10`
- `lambda_power_disagree = 5e-5`

## Main Result

The canonical hybrid branch uses `lambda_temp_disagree = 0.10`.

It is materially better than the failed direct `v3.5` warm-start path, and it gives the best overall tradeoff among the tested hybrid penalty values.

### Hybrid thermostatic on live BOPTEST

| scenario | m_s | violation_pct | rmse_center_c | energy_kwh |
| --- | ---: | ---: | ---: | ---: |
| peak_heat_window | {peak_h['m_s']:.4f} | {peak_h['violation_pct']:.2f} | {peak_h['rmse_center_c']:.3f} | {peak_h['energy_kwh']:.1f} |
| typical_heat_window | {typ_h['m_s']:.4f} | {typ_h['violation_pct']:.2f} | {typ_h['rmse_center_c']:.3f} | {typ_h['energy_kwh']:.1f} |

### Context against pure `v3` thermostatic

| scenario | v3 m_s | hybrid m_s | hybrid delta | v3 energy_kwh | hybrid energy_kwh |
| --- | ---: | ---: | ---: | ---: | ---: |
| peak_heat_window | {peak_v3['m_s']:.4f} | {peak_h['m_s']:.4f} | {peak_h['m_s'] - peak_v3['m_s']:.4f} | {peak_v3['energy_kwh']:.1f} | {peak_h['energy_kwh']:.1f} |
| typical_heat_window | {typ_v3['m_s']:.4f} | {typ_h['m_s']:.4f} | {typ_h['m_s'] - typ_v3['m_s']:.4f} | {typ_v3['energy_kwh']:.1f} | {typ_h['energy_kwh']:.1f} |

### Context against failed direct `v3.5` warm-start

| scenario | warm-start m_s | hybrid m_s | relative improvement |
| --- | ---: | ---: | ---: |
| peak_heat_window | {peak_w['m_s']:.4f} | {peak_h['m_s']:.4f} | {(1.0 - peak_h['m_s'] / peak_w['m_s']) * 100.0:.1f}% |
| typical_heat_window | {typ_w['m_s']:.4f} | {typ_h['m_s']:.4f} | {(1.0 - typ_h['m_s'] / typ_w['m_s']) * 100.0:.1f}% |

### Canonical interpretation of `lambda = 0.10`

- On `peak_heat_window`, the hybrid nearly matches pure `v3` comfort while using less energy.
- On `typical_heat_window`, the hybrid is better than pure `v3` on `m_s`, violation, and energy, with only a small RMSE penalty.
- Therefore `lambda = 0.10` is the current default hybrid setting for transfer to the next controller family.

## Interpretation

- The hybrid regularizer is useful: it rescues the `v3.5` branch from catastrophic Block 2 performance.
- The `0.10` setting is the current best compromise, not just a proof of concept.
- The next downstream question is no longer thermostatic tuning, but whether the same hybrid default helps `HDRL`, and then `MORL`.

## Figures

![Hybrid comfort traces](figures/hybrid_boptest_comfort_traces.png)

![Hybrid power and cumulative energy](figures/hybrid_boptest_power_energy_traces.png)

![Hybrid vs PI m_s](figures/hybrid_vs_pi_ms.png)

![Hybrid vs PI violation](figures/hybrid_vs_pi_violation.png)

![Hybrid vs PI energy](figures/hybrid_vs_pi_energy.png)
"""
    out_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build snapshot artifacts for the current hybrid surrogate branch.")
    parser.add_argument(
        "--hybrid-summary",
        default="outputs/block2_thermostatic_hybrid_v3_v35_l010/summary.csv",
    )
    parser.add_argument(
        "--hybrid-trace-dir",
        default="outputs/block2_thermostatic_hybrid_v3_v35_l010/traces",
    )
    parser.add_argument(
        "--pi-summary",
        default="outputs/block2_bestest_air_15min_thermostatic_v35/summary.csv",
    )
    parser.add_argument(
        "--v3-summary",
        default="outputs/bestest_air_article7_style_15min/summary.csv",
    )
    parser.add_argument(
        "--warmstart-summary",
        default="outputs/block2_thermostatic_warmstart_utility/comparison_summary.csv",
    )
    parser.add_argument(
        "--report-path",
        default="reports/block2_hybrid_surrogate_report.md",
    )
    parser.add_argument(
        "--metrics-path",
        default="reports/block2_hybrid_surrogate_metrics.csv",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/figures",
    )
    args = parser.parse_args()

    figures_dir = REPO_ROOT / args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    hybrid_df = _read_summary(REPO_ROOT / args.hybrid_summary)
    pi_source_df = _read_summary(REPO_ROOT / args.pi_summary)
    v3_source_df = _read_summary(REPO_ROOT / args.v3_summary)
    warm_df = _read_summary(REPO_ROOT / args.warmstart_summary)

    pi_df = pi_source_df[pi_source_df["controller"] == "pi"].copy()
    v3_df = v3_source_df[v3_source_df["controller"] == "thermostatic"].copy()
    warm_df = warm_df[warm_df["mode"] == "warmstart"].copy()

    compare_df = pd.concat(
        [
            pi_df.assign(controller="pi")[["scenario", "controller", "m_s", "violation_pct", "energy_kwh"]],
            hybrid_df.assign(controller="hybrid")[["scenario", "controller", "m_s", "violation_pct", "energy_kwh"]],
        ],
        ignore_index=True,
    )
    compare_df.to_csv(REPO_ROOT / args.metrics_path, index=False)

    traces = {
        "peak_heat_window": _read_trace(REPO_ROOT / args.hybrid_trace_dir / "peak_heat_window_thermostatic.csv"),
        "typical_heat_window": _read_trace(
            REPO_ROOT / args.hybrid_trace_dir / "typical_heat_window_thermostatic.csv"
        ),
    }

    build_comfort_plot(traces, figures_dir / "hybrid_boptest_comfort_traces.png")
    build_power_plot(traces, figures_dir / "hybrid_boptest_power_energy_traces.png")
    build_bar_plot(
        compare_df,
        metric="m_s",
        ylabel="m_s",
        title="Hybrid thermostatic vs PI: comfort-safety score",
        out_path=figures_dir / "hybrid_vs_pi_ms.png",
    )
    build_bar_plot(
        compare_df,
        metric="violation_pct",
        ylabel="Violation [%]",
        title="Hybrid thermostatic vs PI: comfort violations",
        out_path=figures_dir / "hybrid_vs_pi_violation.png",
    )
    build_energy_bar_plot(compare_df, figures_dir / "hybrid_vs_pi_energy.png")
    write_report(REPO_ROOT / args.report_path, hybrid_df, compare_df, v3_df, warm_df)


if __name__ == "__main__":
    main()
