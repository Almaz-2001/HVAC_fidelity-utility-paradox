"""
evaluation/validate_surrogate_comfort_traces.py

Validate the current direct-TSup surrogate on comfort-oriented controller
traces instead of exploratory random/heat/cool/mixed collection episodes.

This script replays the saved action traces from:
  - evaluation/eval_thermostatic.py
  - evaluation/yearly_validation_hdrl.py

and compares recursive surrogate predictions against the BOPTEST traces that
those evaluators already saved to CSV.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from surrogate.rc_node_v2 import RCNeuralODEv2


MODEL_PATH = "outputs/surrogate_v2/rc_node_v3_tsupply.pt"
OUTPUT_DIR = "outputs/surrogate_comfort_traces"
T_TARGET = 22.0
T_LOW = 21.0
T_HIGH = 25.0
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


@dataclass(frozen=True)
class TraceSpec:
    key: str
    label: str
    pattern: str
    temp_col: str
    power_col: str
    color: str


TRACE_SPECS = [
    TraceSpec(
        key="thermostatic",
        label="Thermostatic PPO",
        pattern="thermostatic_scenario_*.csv",
        temp_col="t_zone",
        power_col="p_total",
        color="#f58518",
    ),
    TraceSpec(
        key="hdrl",
        label="HDRL",
        pattern="hdrl_scenario_*.csv",
        temp_col="temp",
        power_col="power",
        color="#54a24b",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate surrogate on thermostatic/HDRL comfort traces")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--artifact_dir", default=OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_surrogate(model_path: str, device: str) -> tuple[RCNeuralODEv2, torch.device]:
    torch_device = torch.device(device)
    checkpoint = torch.load(model_path, map_location=torch_device, weights_only=False)
    model = RCNeuralODEv2(hidden_dim=checkpoint.get("hidden_dim", 64))
    model.load_state_dict(checkpoint["model_state"])
    model.to(torch_device)
    model.eval()
    return model, torch_device


def _scenario_name(path: Path, spec: TraceSpec) -> str:
    prefix = spec.pattern.replace("*", "").replace(".csv", "")
    return path.stem.replace(prefix, "")


def _require_trace_columns(df: pd.DataFrame, path: Path, spec: TraceSpec) -> None:
    required = {"prev_time", "prev_t_zone", "prev_t_amb", "a0", "a1", spec.temp_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(
            f"{path.name} is missing columns {missing}. "
            "Rerun eval_thermostatic.py and yearly_validation_hdrl.py after the trace-format update."
        )


def build_rollout_trace(
    df: pd.DataFrame,
    spec: TraceSpec,
    model: RCNeuralODEv2,
    device: torch.device,
) -> pd.DataFrame:
    _require_trace_columns(df, Path("trace.csv"), spec)
    if df.empty:
        return pd.DataFrame()

    t_curr = float(df.loc[0, "prev_t_zone"])
    rows: list[dict[str, float | int | str]] = []

    for row in df.itertuples(index=False):
        prev_time = float(row.prev_time)
        hour = (prev_time / 3600.0) % 24.0
        day = (prev_time / 86400.0) % 365.0
        prev_t_amb = float(row.prev_t_amb)
        a0 = float(row.a0)
        a1 = float(row.a1)

        with torch.no_grad():
            t_next, p_pred = model(
                torch.tensor([t_curr], dtype=torch.float32, device=device),
                torch.tensor([prev_t_amb], dtype=torch.float32, device=device),
                torch.tensor([hour], dtype=torch.float32, device=device),
                torch.tensor([day], dtype=torch.float32, device=device),
                torch.tensor([a0], dtype=torch.float32, device=device),
                torch.tensor([a1], dtype=torch.float32, device=device),
            )
        pred_t = float(t_next[0].detach().cpu())
        pred_p = float(p_pred[0].detach().cpu())
        actual_t = float(getattr(row, spec.temp_col))
        actual_p = float(getattr(row, spec.power_col))

        rows.append(
            {
                "step": int(row.step),
                "prev_time": prev_time,
                "prev_t_zone": float(row.prev_t_zone),
                "prev_t_amb": prev_t_amb,
                "a0": a0,
                "a1": a1,
                "fan_u": float(getattr(row, "fan_u", np.nan)),
                "t_supply": float(getattr(row, "t_supply", np.nan)),
                "actual_t_zone": actual_t,
                "pred_t_zone": pred_t,
                "temp_error_c": pred_t - actual_t,
                "actual_power_w": actual_p,
                "pred_power_w": pred_p,
                "power_error_w": pred_p - actual_p,
            }
        )
        t_curr = pred_t

    return pd.DataFrame(rows)


def scenario_metrics(trace_df: pd.DataFrame) -> dict[str, float]:
    actual = trace_df["actual_t_zone"].to_numpy(dtype=float)
    pred = trace_df["pred_t_zone"].to_numpy(dtype=float)
    err = pred - actual
    actual_in_band = ((actual >= T_LOW) & (actual <= T_HIGH)).mean() * 100.0
    pred_in_band = ((pred >= T_LOW) & (pred <= T_HIGH)).mean() * 100.0
    return {
        "rmse_c": float(np.sqrt(np.mean(err ** 2))),
        "mae_c": float(np.mean(np.abs(err))),
        "bias_c": float(np.mean(err)),
        "within_05c_pct": float((np.abs(err) < 0.5).mean() * 100.0),
        "within_1c_pct": float((np.abs(err) < 1.0).mean() * 100.0),
        "actual_in_band_pct": float(actual_in_band),
        "pred_in_band_pct": float(pred_in_band),
        "band_gap_pct": float(pred_in_band - actual_in_band),
        "actual_mean_abs_to_target": float(np.mean(np.abs(actual - T_TARGET))),
        "pred_mean_abs_to_target": float(np.mean(np.abs(pred - T_TARGET))),
    }


def load_and_rollout(
    outputs_dir: Path,
    spec: TraceSpec,
    model: RCNeuralODEv2,
    device: torch.device,
    artifact_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    scenario_rows = []
    traces: dict[str, pd.DataFrame] = {}

    paths = sorted(outputs_dir.glob(spec.pattern), key=lambda p: SCENARIO_ORDER.index(_scenario_name(p, spec)))
    if not paths:
        raise FileNotFoundError(f"No files found for {spec.label}: {outputs_dir / spec.pattern}")

    trace_dir = artifact_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        scenario = _scenario_name(path, spec)
        df = pd.read_csv(path)
        _require_trace_columns(df, path, spec)
        trace_df = build_rollout_trace(df, spec, model, device)
        trace_df.insert(0, "scenario", scenario)
        trace_df.insert(0, "controller", spec.label)
        trace_df.to_csv(trace_dir / f"{spec.key}_{scenario}.csv", index=False)
        traces[scenario] = trace_df
        metrics = scenario_metrics(trace_df)
        metrics.update({"controller_key": spec.key, "controller": spec.label, "scenario": scenario})
        scenario_rows.append(metrics)

    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df["scenario"] = pd.Categorical(scenario_df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    scenario_df = scenario_df.sort_values("scenario").reset_index(drop=True)
    return scenario_df, traces


def aggregate(scenario_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scenario_df.groupby(["controller_key", "controller"], as_index=False)
        .agg(
            rmse_mean=("rmse_c", "mean"),
            mae_mean=("mae_c", "mean"),
            bias_mean=("bias_c", "mean"),
            within_05_mean=("within_05c_pct", "mean"),
            within_1_mean=("within_1c_pct", "mean"),
            actual_in_band_mean=("actual_in_band_pct", "mean"),
            pred_in_band_mean=("pred_in_band_pct", "mean"),
            band_gap_mean=("band_gap_pct", "mean"),
            actual_abs_target_mean=("actual_mean_abs_to_target", "mean"),
            pred_abs_target_mean=("pred_mean_abs_to_target", "mean"),
        )
    )
    return summary


def build_controller_full_trace(traces: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    offset = 0
    for scenario in SCENARIO_ORDER:
        trace_df = traces.get(scenario)
        if trace_df is None or trace_df.empty:
            continue
        frame = trace_df.copy()
        frame["global_step"] = np.arange(offset, offset + len(frame))
        frames.append(frame)
        offset += len(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_controller_grid(spec: TraceSpec, traces: dict[str, pd.DataFrame], artifact_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    y_min, y_max = 17.0, 29.0

    for ax, scenario in zip(axes, SCENARIO_ORDER):
        trace_df = traces.get(scenario)
        if trace_df is None or trace_df.empty:
            ax.set_visible(False)
            continue
        x = trace_df["step"].to_numpy(dtype=float)
        ax.axhspan(T_LOW, T_HIGH, color="#cfe8ff", alpha=0.30)
        ax.axhline(T_TARGET, color="#666666", linestyle="--", linewidth=1.0)
        ax.plot(x, trace_df["actual_t_zone"].to_numpy(dtype=float), color="#3266ad", linewidth=1.8, label="BOPTEST")
        ax.plot(x, trace_df["pred_t_zone"].to_numpy(dtype=float), color=spec.color, linewidth=1.6, label="Surrogate")
        ax.set_title(scenario)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Comfort-oriented surrogate rollout: {spec.label}", fontsize=16, fontweight="bold")
    fig.text(0.03, 0.5, "Zone temperature (C)", va="center", rotation="vertical")
    fig.text(0.5, 0.04, "Step", ha="center")
    fig.tight_layout(rect=[0.03, 0.05, 1.0, 0.94])
    out_path = artifact_dir / f"comfort_trace_grid_{spec.key}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_summary(summary_df: pd.DataFrame, artifact_dir: Path) -> Path:
    color_map = {spec.key: spec.color for spec in TRACE_SPECS}
    labels = summary_df["controller"].tolist()
    colors = [color_map[key] for key in summary_df["controller_key"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    metrics = [
        ("rmse_mean", "Mean rollout RMSE (C)"),
        ("within_1_mean", "Error within ±1C (%)"),
        ("band_gap_mean", "Predicted band % - actual band %"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        values = summary_df[metric].to_numpy(dtype=float)
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Surrogate realism on comfort-oriented controller traces", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out_path = artifact_dir / "comfort_trace_summary.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_combined_yearly(
    controller_traces: dict[str, dict[str, pd.DataFrame]],
    artifact_dir: Path,
) -> Path:
    fig, axes = plt.subplots(len(TRACE_SPECS), 1, figsize=(16, 8), sharex=False)
    if len(TRACE_SPECS) == 1:
        axes = [axes]

    for ax, spec in zip(axes, TRACE_SPECS):
        traces = controller_traces[spec.key]
        frames = []
        offset = 0
        boundaries = []
        centers = []

        for scenario in SCENARIO_ORDER:
            trace_df = traces.get(scenario)
            if trace_df is None or trace_df.empty:
                continue
            frame = trace_df.copy()
            frame["global_step"] = np.arange(offset, offset + len(frame))
            frames.append(frame)
            start = offset
            offset += len(frame)
            end = offset
            boundaries.append(end)
            centers.append((0.5 * (start + end), scenario))

        if not frames:
            continue

        full_df = pd.concat(frames, ignore_index=True)
        ax.axhspan(T_LOW, T_HIGH, color="#cfe8ff", alpha=0.30)
        ax.axhline(T_TARGET, color="#666666", linestyle="--", linewidth=1.0)
        ax.plot(
            full_df["global_step"].to_numpy(dtype=float),
            full_df["actual_t_zone"].to_numpy(dtype=float),
            color="#3266ad",
            linewidth=1.7,
            label="BOPTEST",
        )
        ax.plot(
            full_df["global_step"].to_numpy(dtype=float),
            full_df["pred_t_zone"].to_numpy(dtype=float),
            color=spec.color,
            linewidth=1.5,
            label="Surrogate v3",
        )

        for boundary in boundaries[:-1]:
            ax.axvline(boundary, color="#999999", linestyle=":", linewidth=0.8, alpha=0.7)

        ax.set_ylabel(f"{spec.label}\nT_zone (C)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
        ax.set_ylim(17.0, 29.0)

        if centers:
            ax.set_xticks([c[0] for c in centers])
            ax.set_xticklabels([c[1].replace("_", "\n") for c in centers], fontsize=8)

    axes[-1].set_xlabel("Concatenated yearly scenarios")
    fig.suptitle("Surrogate v3 vs BOPTEST on comfort-oriented control traces", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = artifact_dir / "surrogate_v3_vs_boptest_yearly.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_parity(
    scenario_frames: list[pd.DataFrame],
    artifact_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, len(TRACE_SPECS), figsize=(12, 5), sharex=True, sharey=True)
    if len(TRACE_SPECS) == 1:
        axes = [axes]

    for ax, spec in zip(axes, TRACE_SPECS):
        trace_df = scenario_frames[[frame["controller_key"].iloc[0] for frame in scenario_frames].index(spec.key)]
        x = trace_df["actual_t_zone"].to_numpy(dtype=float)
        y = trace_df["pred_t_zone"].to_numpy(dtype=float)
        ax.scatter(x, y, s=10, alpha=0.25, color=spec.color)
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        ax.plot([lo, hi], [lo, hi], color="#444444", linestyle="--", linewidth=1.0)
        ax.set_title(spec.label)
        ax.set_xlabel("BOPTEST T_zone (C)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Surrogate v3 T_zone (C)")
    fig.suptitle("Surrogate v3 parity vs BOPTEST on comfort traces", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = artifact_dir / "surrogate_v3_vs_boptest_parity.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_standalone_surrogate(
    boptest_log: pd.DataFrame,
    surrogate_log: pd.DataFrame,
    artifact_dir: Path,
) -> Path:
    merged = boptest_log.merge(
        surrogate_log,
        on=[
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
        ],
        how="inner",
    ).sort_values(["controller_key", "global_step"], kind="stable")

    merged["combined_step"] = np.arange(len(merged))

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    axes[0].axhspan(T_LOW, T_HIGH, color="#cfe8ff", alpha=0.30)
    axes[0].axhline(T_TARGET, color="#666666", linestyle="--", linewidth=1.0)
    axes[0].plot(
        merged["combined_step"].to_numpy(dtype=float),
        merged["boptest_t_zone"].to_numpy(dtype=float),
        color="#3266ad",
        linewidth=1.6,
        label="BOPTEST",
    )
    axes[0].plot(
        merged["combined_step"].to_numpy(dtype=float),
        merged["surrogate_t_zone"].to_numpy(dtype=float),
        color="#f58518",
        linewidth=1.5,
        label="Surrogate v3",
    )
    axes[0].set_ylabel("T_zone (C)")
    axes[0].set_ylim(17.0, 29.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        merged["combined_step"].to_numpy(dtype=float),
        merged["boptest_power_w"].to_numpy(dtype=float),
        color="#3266ad",
        linewidth=1.6,
        label="BOPTEST",
    )
    axes[1].plot(
        merged["combined_step"].to_numpy(dtype=float),
        merged["surrogate_power_w"].to_numpy(dtype=float),
        color="#54a24b",
        linewidth=1.5,
        label="Surrogate v3",
    )
    axes[1].set_ylabel("Power (W)")
    axes[1].set_xlabel("Concatenated comfort-action replay")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.suptitle("Standalone surrogate v3 replay vs BOPTEST on comfort-action traces", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = artifact_dir / "surrogate_v3_standalone_replay.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def build_text_summary(summary_df: pd.DataFrame, artifact_dir: Path) -> Path:
    lines = [
        "SURROGATE COMFORT-TRACE VALIDATION",
        "=================================",
        "",
        "This validation uses comfort-oriented action traces from:",
        "- evaluation/eval_thermostatic.py",
        "- evaluation/yearly_validation_hdrl.py",
        "",
        "It is intended to answer how realistic the surrogate is under controllers",
        "that actually try to maintain comfortable indoor temperatures.",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['controller']}: "
            f"mean RMSE = {row['rmse_mean']:.3f} C, "
            f"mean bias = {row['bias_mean']:+.3f} C, "
            f"within ±1C = {row['within_1_mean']:.1f}%, "
            f"band-gap = {row['band_gap_mean']:+.1f} pp"
        )
    out_path = artifact_dir / "comfort_trace_summary.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def build_explicit_logs(full_trace_frames: list[pd.DataFrame], artifact_dir: Path) -> tuple[Path, Path]:
    combined = pd.concat(full_trace_frames, ignore_index=True)
    if "global_step" not in combined.columns:
        rebuilt_frames = []
        for controller_key, controller_df in combined.groupby("controller_key", sort=False):
            offset = 0
            local_frames = []
            for scenario in SCENARIO_ORDER:
                scenario_df = controller_df[controller_df["scenario"] == scenario].copy()
                if scenario_df.empty:
                    continue
                scenario_df["global_step"] = np.arange(offset, offset + len(scenario_df))
                local_frames.append(scenario_df)
                offset += len(scenario_df)
            if local_frames:
                rebuilt_frames.append(pd.concat(local_frames, ignore_index=True))
        combined = pd.concat(rebuilt_frames, ignore_index=True) if rebuilt_frames else combined

    boptest_log = combined[
        [
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
            "actual_t_zone",
            "actual_power_w",
        ]
    ].rename(
        columns={
            "actual_t_zone": "boptest_t_zone",
            "actual_power_w": "boptest_power_w",
        }
    )
    surrogate_log = combined[
        [
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
            "pred_t_zone",
            "pred_power_w",
        ]
    ].rename(
        columns={
            "pred_t_zone": "surrogate_t_zone",
            "pred_power_w": "surrogate_power_w",
        }
    )

    boptest_path = artifact_dir / "boptest_comfort_trace_log.csv"
    surrogate_path = artifact_dir / "surrogate_v3_log.csv"
    boptest_log.to_csv(boptest_path, index=False)
    surrogate_log.to_csv(surrogate_path, index=False)
    return boptest_path, surrogate_path


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model, device = load_surrogate(args.model_path, args.device)

    scenario_frames = []
    full_trace_frames = []
    controller_traces: dict[str, dict[str, pd.DataFrame]] = {}
    grid_paths = []
    for spec in TRACE_SPECS:
        scenario_df, traces = load_and_rollout(outputs_dir, spec, model, device, artifact_dir)
        scenario_df.to_csv(artifact_dir / f"comfort_trace_metrics_{spec.key}.csv", index=False)
        scenario_frames.append(scenario_df)
        full_trace_df = build_controller_full_trace(traces)
        full_trace_df.insert(0, "controller_key", spec.key)
        full_trace_df.to_csv(artifact_dir / f"comfort_trace_full_{spec.key}.csv", index=False)
        full_trace_frames.append(full_trace_df)
        controller_traces[spec.key] = traces
        grid_paths.append(plot_controller_grid(spec, traces, artifact_dir))

    all_scenarios = pd.concat(scenario_frames, ignore_index=True)
    summary_df = aggregate(all_scenarios)
    summary_df.to_csv(artifact_dir / "comfort_trace_summary.csv", index=False)

    summary_plot = plot_summary(summary_df, artifact_dir)
    combined_yearly_plot = plot_combined_yearly(controller_traces, artifact_dir)
    parity_plot = plot_parity(full_trace_frames, artifact_dir)
    summary_txt = build_text_summary(summary_df, artifact_dir)
    boptest_log_path, surrogate_log_path = build_explicit_logs(full_trace_frames, artifact_dir)
    boptest_log_df = pd.read_csv(boptest_log_path)
    surrogate_log_df = pd.read_csv(surrogate_log_path)
    standalone_plot = plot_standalone_surrogate(boptest_log_df, surrogate_log_df, artifact_dir)

    print("COMFORT-TRACE SURROGATE VALIDATION COMPLETE")
    print("===========================================")
    print(f"Saved: {summary_plot}")
    print(f"Saved: {combined_yearly_plot}")
    print(f"Saved: {parity_plot}")
    print(f"Saved: {standalone_plot}")
    print(f"Saved: {boptest_log_path}")
    print(f"Saved: {surrogate_log_path}")
    for path in grid_paths:
        print(f"Saved: {path}")
    print(f"Saved: {summary_txt}")


if __name__ == "__main__":
    main()
