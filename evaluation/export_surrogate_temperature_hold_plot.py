from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ARTIFACT_DIR = "outputs/surrogate_comfort_traces"
DEFAULT_OUT_DIR = "outputs/surrogate_comfort_traces/mobile_plot"
T_TARGET = 22.0
T_LOW = 21.0
T_HIGH = 25.0
Y_MIN = 17.0
Y_MAX = 29.0

CONTROLLER_ORDER = [
    ("thermostatic", "Thermostatic PPO", "#f58518"),
    ("hdrl", "HDRL", "#54a24b"),
]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a graph-ready surrogate vs BOPTEST temperature-hold plot and mobile-friendly logs."
    )
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_logs(artifact_dir: Path) -> pd.DataFrame:
    boptest_path = artifact_dir / "boptest_comfort_trace_log.csv"
    surrogate_path = artifact_dir / "surrogate_v3_log.csv"
    if not boptest_path.exists() or not surrogate_path.exists():
        raise FileNotFoundError(
            "Missing surrogate comfort-trace logs. Run "
            "`evaluation/validate_surrogate_comfort_traces.py` first."
        )

    boptest = pd.read_csv(boptest_path)
    surrogate = pd.read_csv(surrogate_path)
    merge_keys = [
        "controller_key",
        "controller",
        "scenario",
        "step",
        "global_step",
        "prev_time",
        "prev_t_zone",
        "prev_t_amb",
        "a0",
        "a1",
        "fan_u",
        "t_supply",
    ]
    merged = boptest.merge(surrogate, on=merge_keys, how="inner").sort_values(
        ["controller_key", "global_step"], kind="stable"
    )
    if merged.empty:
        raise RuntimeError("Merged surrogate/BOPTEST comfort-trace log is empty.")
    return merged


def build_long_plot_log(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    color_map = {
        "boptest": "#3266ad",
        "thermostatic": "#f58518",
        "hdrl": "#54a24b",
    }

    for controller_key, controller_label, surrogate_color in CONTROLLER_ORDER:
        controller_df = merged[merged["controller_key"] == controller_key].copy()
        if controller_df.empty:
            continue

        controller_df["scenario"] = pd.Categorical(
            controller_df["scenario"], categories=SCENARIO_ORDER, ordered=True
        )
        controller_df = controller_df.sort_values(["scenario", "step"], kind="stable")

        for row in controller_df.itertuples(index=False):
            rows.append(
                {
                    "subplot_key": controller_key,
                    "subplot_label": controller_label,
                    "scenario": row.scenario,
                    "step": int(row.step),
                    "global_step": int(row.global_step),
                    "series_key": "boptest",
                    "series_label": "BOPTEST",
                    "temperature_c": float(row.boptest_t_zone),
                    "color_hex": color_map["boptest"],
                    "line_width": 1.7,
                    "prev_time_s": float(row.prev_time),
                    "prev_t_zone_c": float(row.prev_t_zone),
                    "prev_t_amb_c": float(row.prev_t_amb),
                    "a0": float(row.a0),
                    "a1": float(row.a1),
                    "fan_u": float(row.fan_u),
                    "t_supply_c": float(row.t_supply),
                }
            )
            rows.append(
                {
                    "subplot_key": controller_key,
                    "subplot_label": controller_label,
                    "scenario": row.scenario,
                    "step": int(row.step),
                    "global_step": int(row.global_step),
                    "series_key": "surrogate_v3",
                    "series_label": "Surrogate v3",
                    "temperature_c": float(row.surrogate_t_zone),
                    "color_hex": surrogate_color,
                    "line_width": 1.5,
                    "prev_time_s": float(row.prev_time),
                    "prev_t_zone_c": float(row.prev_t_zone),
                    "prev_t_amb_c": float(row.prev_t_amb),
                    "a0": float(row.a0),
                    "a1": float(row.a1),
                    "fan_u": float(row.fan_u),
                    "t_supply_c": float(row.t_supply),
                }
            )

    return pd.DataFrame(rows)


def build_meta(merged: pd.DataFrame, png_path: Path) -> dict:
    subplots = []
    for controller_key, controller_label, surrogate_color in CONTROLLER_ORDER:
        controller_df = merged[merged["controller_key"] == controller_key].copy()
        if controller_df.empty:
            continue

        controller_df["scenario"] = pd.Categorical(
            controller_df["scenario"], categories=SCENARIO_ORDER, ordered=True
        )
        controller_df = controller_df.sort_values(["scenario", "step"], kind="stable")

        boundaries = []
        ticks = []
        for scenario in SCENARIO_ORDER:
            scenario_df = controller_df[controller_df["scenario"] == scenario]
            if scenario_df.empty:
                continue
            start = int(scenario_df["global_step"].min())
            end = int(scenario_df["global_step"].max()) + 1
            boundaries.append(end)
            ticks.append(
                {
                    "x": 0.5 * (start + end),
                    "label": scenario.replace("_", "\n"),
                    "scenario": scenario,
                }
            )

        subplots.append(
            {
                "subplot_key": controller_key,
                "subplot_label": controller_label,
                "series": [
                    {"series_key": "boptest", "label": "BOPTEST", "color_hex": "#3266ad", "line_width": 1.7},
                    {
                        "series_key": "surrogate_v3",
                        "label": "Surrogate v3",
                        "color_hex": surrogate_color,
                        "line_width": 1.5,
                    },
                ],
                "boundaries": boundaries[:-1],
                "ticks": ticks,
            }
        )

    return {
        "title": "Surrogate v3 vs BOPTEST on comfort-oriented control traces",
        "x_label": "Concatenated yearly scenarios",
        "y_label": "T_zone (C)",
        "target_temperature_c": T_TARGET,
        "comfort_band_low_c": T_LOW,
        "comfort_band_high_c": T_HIGH,
        "y_min_c": Y_MIN,
        "y_max_c": Y_MAX,
        "subplot_order": [item[0] for item in CONTROLLER_ORDER],
        "subplots": subplots,
        "output_png": str(png_path),
    }


def plot_temperature_hold(merged: pd.DataFrame, png_path: Path) -> None:
    fig, axes = plt.subplots(len(CONTROLLER_ORDER), 1, figsize=(16, 8), sharex=False)
    if len(CONTROLLER_ORDER) == 1:
        axes = [axes]

    for ax, (controller_key, controller_label, surrogate_color) in zip(axes, CONTROLLER_ORDER):
        controller_df = merged[merged["controller_key"] == controller_key].copy()
        if controller_df.empty:
            ax.set_visible(False)
            continue

        controller_df["scenario"] = pd.Categorical(
            controller_df["scenario"], categories=SCENARIO_ORDER, ordered=True
        )
        controller_df = controller_df.sort_values(["scenario", "step"], kind="stable")

        ax.axhspan(T_LOW, T_HIGH, color="#cfe8ff", alpha=0.30)
        ax.axhline(T_TARGET, color="#666666", linestyle="--", linewidth=1.0)
        ax.plot(
            controller_df["global_step"].to_numpy(dtype=float),
            controller_df["boptest_t_zone"].to_numpy(dtype=float),
            color="#3266ad",
            linewidth=1.7,
            label="BOPTEST",
        )
        ax.plot(
            controller_df["global_step"].to_numpy(dtype=float),
            controller_df["surrogate_t_zone"].to_numpy(dtype=float),
            color=surrogate_color,
            linewidth=1.5,
            label="Surrogate v3",
        )

        tick_positions = []
        tick_labels = []
        for scenario in SCENARIO_ORDER:
            scenario_df = controller_df[controller_df["scenario"] == scenario]
            if scenario_df.empty:
                continue
            start = int(scenario_df["global_step"].min())
            end = int(scenario_df["global_step"].max()) + 1
            if scenario != SCENARIO_ORDER[-1]:
                ax.axvline(end, color="#999999", linestyle=":", linewidth=0.8, alpha=0.7)
            tick_positions.append(0.5 * (start + end))
            tick_labels.append(scenario.replace("_", "\n"))

        ax.set_ylabel(f"{controller_label}\nT_zone (C)")
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Concatenated yearly scenarios")
    fig.suptitle("Surrogate v3 vs BOPTEST on comfort-oriented control traces", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_logs(artifact_dir)

    png_path = out_dir / "surrogate_temperature_hold_vs_boptest.png"
    long_csv_path = out_dir / "surrogate_temperature_hold_plot_long.csv"
    wide_csv_path = out_dir / "surrogate_temperature_hold_plot_wide.csv"
    meta_path = out_dir / "surrogate_temperature_hold_plot_meta.json"

    plot_temperature_hold(merged, png_path)

    long_df = build_long_plot_log(merged)
    long_df.to_csv(long_csv_path, index=False)
    merged.to_csv(wide_csv_path, index=False)

    meta = build_meta(merged, png_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("SURROGATE TEMPERATURE HOLD EXPORT COMPLETE")
    print("==========================================")
    print(f"Saved plot: {png_path}")
    print(f"Saved long plot log: {long_csv_path}")
    print(f"Saved wide plot log: {wide_csv_path}")
    print(f"Saved plot meta: {meta_path}")


if __name__ == "__main__":
    main()
