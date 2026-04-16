from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

SCENARIO_MAP = {
    "peak_heat_window": "peak_heat_day",
    "typical_heat_window": "typical_heat_day",
}

SCENARIO_LABELS = {
    "peak_heat_window": "Peak heat window",
    "typical_heat_window": "Typical heat window",
}

CONTROLLER_ORDER = ["thermostatic", "hdrl", "surrogate_mpc"]
CONTROLLER_LABELS = {
    "thermostatic": "Thermostatic PPO",
    "hdrl": "HDRL",
    "surrogate_mpc": "Surrogate MPC",
}

METRICS = [
    ("m_s", "m_s", True),
    ("r_time", "r_time", True),
    ("r_sev", "r_sev", True),
    ("violation_pct", "Violation, %", True),
    ("rmse_22_c", "RMSE to 22C", True),
    ("mean_power_w", "Mean power, W", True),
    ("energy_kwh", "Energy, kWh", True),
    ("within_band_pct", "Within band, %", False),
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected YAML mapping in {path}")
    return payload


def build_article7_ref_df(config: dict[str, Any]) -> pd.DataFrame:
    refs = config.get("references", {}).get("article7_table5", {})
    rows: list[dict[str, Any]] = []
    for scenario_key in ["peak_heat_day", "typical_heat_day"]:
        row = {
            "article7_scenario": scenario_key,
            "article7_pi_m_s": refs.get("pi", {}).get(scenario_key),
            "article7_mpc_m_s": refs.get("mpc", {}).get(scenario_key),
            "article7_safe_drl_m_s": 0.0 if scenario_key == "peak_heat_day" else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def enrich_summary(summary_df: pd.DataFrame, article7_df: pd.DataFrame) -> pd.DataFrame:
    enriched = summary_df.copy()
    enriched["article7_scenario"] = enriched["scenario"].map(SCENARIO_MAP)
    enriched["scenario_label"] = enriched["scenario"].map(SCENARIO_LABELS).fillna(enriched["scenario"])
    enriched["controller_label"] = enriched["controller"].map(CONTROLLER_LABELS).fillna(enriched["controller"])
    enriched = enriched.merge(article7_df, on="article7_scenario", how="left")
    return enriched


def plot_ms_vs_article7(enriched: pd.DataFrame, out_path: Path) -> None:
    scenario_order = ["peak_heat_window", "typical_heat_window"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    article7_palette = {
        "Article7 PI": "#c0392b",
        "Article7 MPC": "#2980b9",
        "Article7 Safe DRL": "#16a085",
    }
    controller_colors = {
        "thermostatic": "#1f77b4",
        "hdrl": "#ff7f0e",
        "surrogate_mpc": "#2ca02c",
    }

    for ax, scenario in zip(axes, scenario_order):
        subset = enriched[enriched["scenario"] == scenario].copy()
        subset["controller_rank"] = subset["controller"].map(
            {name: idx for idx, name in enumerate(CONTROLLER_ORDER)}
        )
        subset = subset.sort_values("controller_rank")
        x = np.arange(len(subset))
        values = subset["m_s"].to_numpy(dtype=float)
        labels = subset["controller_label"].tolist()
        colors = [controller_colors.get(name, "#7f8c8d") for name in subset["controller"]]

        ax.bar(x, values, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        ax.set_ylabel("m_s")
        ax.grid(axis="y", alpha=0.25)

        ref_row = subset.iloc[0]
        pi_ref = ref_row.get("article7_pi_m_s")
        mpc_ref = ref_row.get("article7_mpc_m_s")
        safe_ref = ref_row.get("article7_safe_drl_m_s")
        if pd.notna(pi_ref):
            ax.axhline(float(pi_ref), color=article7_palette["Article7 PI"], linestyle="--", linewidth=1.8, label="Article7 PI")
        if pd.notna(mpc_ref):
            ax.axhline(float(mpc_ref), color=article7_palette["Article7 MPC"], linestyle="--", linewidth=1.8, label="Article7 MPC")
        if pd.notna(safe_ref):
            ax.axhline(float(safe_ref), color=article7_palette["Article7 Safe DRL"], linestyle="--", linewidth=1.8, label="Article7 Safe DRL")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("bestest_air controllers vs Article 7 safety references", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_all_metrics_for_scenario(enriched: pd.DataFrame, scenario: str, out_path: Path) -> None:
    subset = enriched[enriched["scenario"] == scenario].copy()
    subset["controller_rank"] = subset["controller"].map(
        {name: idx for idx, name in enumerate(CONTROLLER_ORDER)}
    )
    subset = subset.sort_values("controller_rank")
    labels = subset["controller_label"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, (metric_key, metric_label, lower_is_better) in zip(axes, METRICS):
        values = subset[metric_key].to_numpy(dtype=float)
        ax.bar(x, values, color=palette[: len(values)], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        suffix = "lower is better" if lower_is_better else "higher is better"
        ax.set_title(f"{metric_label}\n({suffix})", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

        for idx, value in enumerate(values):
            fmt = f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
            ax.text(idx, value, fmt, ha="center", va="bottom", fontsize=8)

        if metric_key == "m_s" and not subset.empty:
            ref_row = subset.iloc[0]
            if pd.notna(ref_row.get("article7_pi_m_s")):
                ax.axhline(float(ref_row["article7_pi_m_s"]), color="#c0392b", linestyle="--", linewidth=1.5)
            if pd.notna(ref_row.get("article7_mpc_m_s")):
                ax.axhline(float(ref_row["article7_mpc_m_s"]), color="#2980b9", linestyle="--", linewidth=1.5)
            if pd.notna(ref_row.get("article7_safe_drl_m_s")):
                ax.axhline(float(ref_row["article7_safe_drl_m_s"]), color="#16a085", linestyle="--", linewidth=1.5)

    fig.suptitle(f"All controller metrics | {SCENARIO_LABELS.get(scenario, scenario)}", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_scatter(enriched: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    controller_markers = {
        "thermostatic": "o",
        "hdrl": "s",
        "surrogate_mpc": "^",
    }
    scenario_colors = {
        "peak_heat_window": "#8e44ad",
        "typical_heat_window": "#27ae60",
    }

    for _, row in enriched.iterrows():
        controller = row["controller"]
        scenario = row["scenario"]
        ax.scatter(
            row["mean_power_w"],
            row["m_s"],
            s=110,
            marker=controller_markers.get(controller, "o"),
            color=scenario_colors.get(scenario, "#34495e"),
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
        )
        label = f"{CONTROLLER_LABELS.get(controller, controller)}\n{SCENARIO_LABELS.get(scenario, scenario)}"
        ax.annotate(
            label,
            (row["mean_power_w"], row["m_s"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.set_xlabel("Mean HVAC power, W")
    ax.set_ylabel("m_s")
    ax.set_title("Trade-off map on bestest_air article7-style windows")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_notes(enriched: pd.DataFrame, out_path: Path) -> None:
    best_rows = (
        enriched.sort_values(["scenario", "m_s", "mean_power_w"])
        .groupby("scenario", as_index=False)
        .first()
    )

    lines = [
        "Context",
        "- Direct literature overlay is only available for m_s.",
        "- Article 7 references come from the hydronic benchmark config, so they are contextual, not apples-to-apples with bestest_air.",
        "- All other metrics in this bundle are internal comparisons across our controllers on the same bestest_air windows.",
        "",
        "Best controller by m_s per scenario",
    ]

    for _, row in best_rows.iterrows():
        lines.append(
            f"- {SCENARIO_LABELS.get(row['scenario'], row['scenario'])}: "
            f"{row['controller_label']} | m_s={row['m_s']:.4f} | "
            f"viol={row['violation_pct']:.1f}% | power={row['mean_power_w']:.1f} W"
        )

    lines.extend(
        [
            "",
            "Why surrogate-trained PPO is stronger than current surrogate-MPC",
            "- Thermostatic PPO and HDRL were trained offline over many surrogate episodes and learned robust policy priors, not just one-step optimization.",
            "- They use richer observations, including time, forecasts, previous action and delta-T history.",
            "- HDRL adds seasonal gating and emergency logic; surrogate-MPC currently does not.",
            "- The current surrogate-MPC trusts the surrogate inside the online optimizer, so model bias translates directly into suboptimal real actions on BOPTEST.",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build comparison figures for bestest_air article7-style benchmark."
    )
    parser.add_argument(
        "--benchmark-summary",
        default="outputs/bestest_air_article7_style/summary.csv",
    )
    parser.add_argument(
        "--article7-config",
        default="configs/article7_hydronic.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="results/article7_bestest_air_comparison",
    )
    args = parser.parse_args()

    summary_path = REPO_ROOT / args.benchmark_summary
    config_path = REPO_ROOT / args.article7_config
    out_dir = REPO_ROOT / args.out_dir
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_path)
    article7_cfg = load_yaml(config_path)
    article7_df = build_article7_ref_df(article7_cfg)
    enriched = enrich_summary(summary_df, article7_df)

    enriched.to_csv(tables_dir / "bestest_air_article7_enriched_summary.csv", index=False)
    article7_df.to_csv(tables_dir / "article7_reference_context.csv", index=False)
    (tables_dir / "bestest_air_article7_enriched_summary.json").write_text(
        enriched.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    plot_ms_vs_article7(enriched, figures_dir / "ms_vs_article7_context.png")
    plot_all_metrics_for_scenario(
        enriched,
        "peak_heat_window",
        figures_dir / "all_metrics_peak_heat_window.png",
    )
    plot_all_metrics_for_scenario(
        enriched,
        "typical_heat_window",
        figures_dir / "all_metrics_typical_heat_window.png",
    )
    plot_tradeoff_scatter(enriched, figures_dir / "tradeoff_power_vs_ms.png")
    build_notes(enriched, out_dir / "README.txt")

    manifest = {
        "benchmark_summary": str(summary_path),
        "article7_config": str(config_path),
        "outputs": {
            "figures": sorted([path.name for path in figures_dir.iterdir() if path.is_file()]),
            "tables": sorted([path.name for path in tables_dir.iterdir() if path.is_file()]),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved comparison bundle: {out_dir}")
    print("Figures:")
    for path in sorted(figures_dir.iterdir()):
        if path.is_file():
            print(f"  {path}")
    print("Tables:")
    for path in sorted(tables_dir.iterdir()):
        if path.is_file():
            print(f"  {path}")


if __name__ == "__main__":
    main()
