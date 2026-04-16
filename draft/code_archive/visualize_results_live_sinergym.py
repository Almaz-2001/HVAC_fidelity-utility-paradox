from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = ROOT / "outputs" / "legacy_sinergym"
DEFAULT_OUT_DIR = Path(
    os.environ.get("LIVE_FIGURE_OUTPUT_DIR")
    or os.environ.get("FIGURE_OUTPUT_DIR")
    or (ROOT / "outputs" / "legacy_sinergym" / "live_figures")
)

RUN_ORDER = [
    "comfort_only",
    "comfort_dominant",
    "balanced",
    "energy_dominant",
    "energy_only",
]
POLICY_ORDER = ["ppo", "rule_based", "random", "zero_hold"]
POLICY_LABELS = {
    "ppo": "Learned PPO",
    "rule_based": "Rule-based",
    "random": "Random",
    "zero_hold": "Zero-hold",
}
RUN_LABELS = {
    "comfort_only": "Comfort only",
    "comfort_dominant": "Comfort dominant",
    "balanced": "Balanced",
    "energy_dominant": "Energy dominant",
    "energy_only": "Energy only",
}
COLORS = {
    "ppo": "#3266ad",
    "rule_based": "#1D9E75",
    "random": "#D85A30",
    "zero_hold": "#73726c",
    "comfort_only": "#5B8FF9",
    "comfort_dominant": "#61DDAA",
    "balanced": "#65789B",
    "energy_dominant": "#F6BD16",
    "energy_only": "#E86452",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build live Sinergym figures from real CSV outputs instead of hard-coded report values."
    )
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--representative-seed", type=int, default=42)
    return parser.parse_args()


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _compute_trace_metrics(df: pd.DataFrame) -> dict[str, float]:
    zone_temp = pd.to_numeric(df["zone_temp"], errors="coerce")
    hvac_power = pd.to_numeric(df["hvac_power"], errors="coerce")
    comfort = pd.to_numeric(df["comfort"], errors="coerce")

    comfort_penalty = np.clip(-comfort.to_numpy(dtype=float), 0.0, None)
    zone_arr = zone_temp.to_numpy(dtype=float)
    power_arr = hvac_power.to_numpy(dtype=float)

    in_band_pct = float((comfort_penalty <= 1e-12).mean() * 100.0) if len(comfort_penalty) else np.nan

    return {
        "steps": float(len(df)),
        "mean_hvac_power_w": float(np.nanmean(power_arr)),
        "mean_zone_temp_c": float(np.nanmean(zone_arr)),
        "min_zone_temp_c": float(np.nanmin(zone_arr)),
        "max_zone_temp_c": float(np.nanmax(zone_arr)),
        "temp_p05_c": float(np.nanpercentile(zone_arr, 5)),
        "temp_p95_c": float(np.nanpercentile(zone_arr, 95)),
        "mean_comfort_penalty": float(np.nanmean(comfort_penalty)),
        "p95_comfort_penalty": float(np.nanpercentile(comfort_penalty, 95)),
        "comfort_violation_pct": float((comfort_penalty > 1e-12).mean() * 100.0),
        "comfort_in_band_pct": in_band_pct,
    }


def _seed_from_dir(seed_dir: Path) -> int:
    return int(seed_dir.name.replace("seed", ""))


def collect_default_policy_metrics(base_dir: Path) -> tuple[pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    rows: list[dict[str, float | int | str]] = []
    traces: dict[tuple[str, int], pd.DataFrame] = {}

    for seed_dir in sorted(base_dir.glob("seed*")):
        if not seed_dir.is_dir():
            continue
        seed = _seed_from_dir(seed_dir)

        ppo_path = seed_dir / "eval" / "ppo_eval.csv"
        df = _safe_read_csv(ppo_path)
        if df is not None and not df.empty:
            rows.append(
                {
                    "seed": seed,
                    "policy": "ppo",
                    **_compute_trace_metrics(df),
                }
            )
            traces[("ppo", seed)] = df

        baseline_dir = seed_dir / "baselines"
        for baseline_path in sorted(baseline_dir.glob("*.csv")):
            df = _safe_read_csv(baseline_path)
            if df is None or df.empty:
                continue
            policy = baseline_path.stem
            rows.append(
                {
                    "seed": seed,
                    "policy": policy,
                    **_compute_trace_metrics(df),
                }
            )
            traces[(policy, seed)] = df

    return pd.DataFrame(rows), traces


def collect_live_pareto_metrics(base_dir: Path) -> pd.DataFrame:
    pareto_csv = base_dir / "pareto" / "pareto_results.csv"
    pareto_df = _safe_read_csv(pareto_csv)
    if pareto_df is None or pareto_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for item in pareto_df.to_dict(orient="records"):
        run_name = str(item["run_name"])
        seed = int(item["seed"])
        eval_path = base_dir / "pareto" / run_name / f"seed{seed}" / "eval" / "ppo_eval.csv"
        df = _safe_read_csv(eval_path)
        if df is None or df.empty:
            continue
        rows.append(
            {
                "run_name": run_name,
                "seed": seed,
                "w_comfort": float(item["w_comfort"]),
                "w_energy": float(item["w_energy"]),
                **_compute_trace_metrics(df),
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    metrics = [
        "mean_hvac_power_w",
        "mean_comfort_penalty",
        "p95_comfort_penalty",
        "mean_zone_temp_c",
        "min_zone_temp_c",
        "max_zone_temp_c",
        "comfort_violation_pct",
        "comfort_in_band_pct",
    ]
    grouped = df.groupby(group_col)[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_")
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


def plot_baseline_comparison(summary_df: pd.DataFrame, out_dir: Path) -> Path | None:
    if summary_df.empty:
        return None

    available = [p for p in POLICY_ORDER if p in set(summary_df["policy"])]
    if not available:
        return None

    frame = summary_df.set_index("policy").loc[available].reset_index()
    labels = [POLICY_LABELS.get(p, p) for p in frame["policy"]]
    x = np.arange(len(frame))
    colors = [COLORS.get(p, "#888888") for p in frame["policy"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    specs = [
        ("mean_hvac_power_w", "Mean HVAC power (W)"),
        ("mean_comfort_penalty", "Mean comfort penalty"),
        ("p95_comfort_penalty", "95th percentile comfort penalty"),
    ]

    for ax, (metric, title) in zip(axes, specs):
        means = frame[f"{metric}_mean"].to_numpy(dtype=float)
        stds = np.nan_to_num(frame.get(f"{metric}_std", pd.Series([0] * len(frame))).to_numpy(dtype=float))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.9, capsize=4, edgecolor="white")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        for i, value in enumerate(means):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Live Sinergym Baseline Comparison (real CSVs)", fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "live_fig1_baseline_comparison.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_representative_trajectories(
    traces: dict[tuple[str, int], pd.DataFrame],
    representative_seed: int,
    out_dir: Path,
) -> Path | None:
    available = [
        policy
        for policy in POLICY_ORDER
        if (policy, representative_seed) in traces
    ]
    if not available:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for policy in available:
        df = traces[(policy, representative_seed)]
        steps = df["step"].to_numpy(dtype=float)
        axes[0].plot(
            steps,
            pd.to_numeric(df["zone_temp"], errors="coerce"),
            linewidth=1.4,
            label=POLICY_LABELS.get(policy, policy),
            color=COLORS.get(policy, "#888888"),
        )
        axes[1].plot(
            steps,
            pd.to_numeric(df["hvac_power"], errors="coerce"),
            linewidth=1.4,
            label=POLICY_LABELS.get(policy, policy),
            color=COLORS.get(policy, "#888888"),
        )

    axes[0].axhspan(20.0, 26.0, color="#BFD7EA", alpha=0.25, label="Comfort band 20–26C")
    axes[0].set_ylabel("Zone temperature (C)")
    axes[0].set_title(f"Representative seed {representative_seed}: zone temperature")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].set_ylabel("HVAC power (W)")
    axes[1].set_xlabel("Step")
    axes[1].set_title(f"Representative seed {representative_seed}: HVAC power")

    fig.suptitle("Live Sinergym Trajectories (real eval and baseline CSVs)", fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "live_fig2_representative_trajectories.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_live_pareto(summary_df: pd.DataFrame, out_dir: Path) -> Path | None:
    if summary_df.empty:
        return None

    available = [r for r in RUN_ORDER if r in set(summary_df["run_name"])]
    frame = summary_df.set_index("run_name").loc[available].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    specs = [
        ("mean_comfort_penalty_mean", "Mean comfort penalty"),
        ("p95_comfort_penalty_mean", "95th percentile comfort penalty"),
    ]

    for ax, (y_col, y_title) in zip(axes, specs):
        for _, row in frame.iterrows():
            run_name = row["run_name"]
            ax.errorbar(
                row["mean_hvac_power_w_mean"],
                row[y_col],
                xerr=np.nan_to_num(row.get("mean_hvac_power_w_std", 0.0)),
                yerr=np.nan_to_num(row.get(y_col.replace("_mean", "_std"), 0.0)),
                fmt="o",
                color=COLORS.get(run_name, "#888888"),
                markersize=8,
                capsize=4,
            )
            ax.annotate(
                RUN_LABELS.get(run_name, run_name),
                (row["mean_hvac_power_w_mean"], row[y_col]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )

        ax.set_xlabel("Mean HVAC power (W)")
        ax.set_ylabel(y_title)
        ax.set_title(y_title + " vs power")

    fig.suptitle("Live Pareto Sweep from real eval CSVs", fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "live_fig3_pareto_tradeoff.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_summary(
    base_dir: Path,
    baseline_summary: pd.DataFrame,
    pareto_summary: pd.DataFrame,
    out_dir: Path,
) -> Path:
    lines: list[str] = []
    lines.append("LIVE SINERGYM SUMMARY")
    lines.append("====================")
    lines.append("")
    lines.append(f"Source directory: {base_dir}")
    lines.append("This report is built from real CSV outputs, not hard-coded figure values.")
    lines.append("")

    available_policies = sorted(baseline_summary["policy"].tolist()) if not baseline_summary.empty else []
    if "rule_based" not in available_policies:
        lines.append("Caveat: no rule-based baseline CSV was found. Current live comparison can only use PPO, random and zero-hold.")
        lines.append("")

    if not baseline_summary.empty:
        lines.append("Baseline comparison (mean across available seeds):")
        ordered = [p for p in POLICY_ORDER if p in set(baseline_summary["policy"])]
        frame = baseline_summary.set_index("policy").loc[ordered].reset_index()
        for _, row in frame.iterrows():
            lines.append(
                f"- {POLICY_LABELS.get(row['policy'], row['policy'])}: "
                f"mean HVAC power = {row['mean_hvac_power_w_mean']:.2f} W, "
                f"mean comfort penalty = {row['mean_comfort_penalty_mean']:.3f}, "
                f"p95 comfort penalty = {row['p95_comfort_penalty_mean']:.3f}, "
                f"mean zone temp = {row['mean_zone_temp_c_mean']:.3f} C"
            )
        lines.append("")

    if not pareto_summary.empty:
        lines.append("Live Pareto observations:")
        frame = pareto_summary.copy()
        best_power = frame.loc[frame["mean_hvac_power_w_mean"].idxmin()]
        best_comfort = frame.loc[frame["mean_comfort_penalty_mean"].idxmin()]
        lines.append(
            f"- Lowest live mean HVAC power: {RUN_LABELS.get(best_power['run_name'], best_power['run_name'])} "
            f"at {best_power['mean_hvac_power_w_mean']:.2f} W"
        )
        lines.append(
            f"- Lowest live mean comfort penalty: {RUN_LABELS.get(best_comfort['run_name'], best_comfort['run_name'])} "
            f"at {best_comfort['mean_comfort_penalty_mean']:.3f}"
        )

        collapsed = frame[
            np.isclose(frame["mean_hvac_power_w_mean"], frame["mean_hvac_power_w_mean"].iloc[0], atol=1e-6)
            & np.isclose(frame["mean_comfort_penalty_mean"], frame["mean_comfort_penalty_mean"].iloc[0], atol=1e-6)
        ]
        if len(collapsed) >= 2:
            runs = ", ".join(RUN_LABELS.get(r, r) for r in collapsed["run_name"].tolist())
            lines.append(
                f"- Warning: multiple Pareto settings collapse to the same live point: {runs}. "
                "This indicates the current legacy sweep is not producing a meaningful front."
            )
        lines.append("")

    lines.append("Interpretation:")
    lines.append("- This live artifact is honest with respect to the available CSVs.")
    lines.append("- It does not reproduce hidden article values from hard-coded dictionaries.")
    lines.append("- If the thesis requires a rule-based baseline, that baseline must still be implemented and re-run.")

    out_path = out_dir / "live_sinergym_summary.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_honest_reproduction_report(
    base_dir: Path,
    baseline_summary: pd.DataFrame,
    pareto_summary: pd.DataFrame,
    out_dir: Path,
) -> Path:
    available_policies = set(baseline_summary["policy"]) if not baseline_summary.empty else set()
    has_rule_based = "rule_based" in available_policies

    lines: list[str] = []
    lines.append("# Legacy Sinergym Honest Reproduction")
    lines.append("")
    lines.append("## Position")
    lines.append("")
    lines.append("- Current live legacy reproduction differs from the article-era figures.")
    lines.append("- Article-era figures were not fully generated from the current CSV pipeline.")
    lines.append("- We rebuilt an honest live evaluation stack that reads real CSV outputs instead of hard-coded plotting dictionaries.")
    lines.append("")
    lines.append("## What This Report Uses")
    lines.append("")
    lines.append(f"- Source directory: `{base_dir}`")
    lines.append("- Live PPO evaluation CSVs from `seedXX/eval/ppo_eval.csv`")
    lines.append("- Live baseline CSVs from `seedXX/baselines/*.csv`")
    lines.append("- Live Pareto sweep CSVs from `pareto/*/seed*/eval/ppo_eval.csv` and `pareto/pareto_results.csv`")
    lines.append("")
    lines.append("## Why The Legacy Article Figures Are Not A Faithful Live Reproduction")
    lines.append("")
    lines.append("- `evaluation/visualize_results.py` uses hard-coded report dictionaries rather than recomputing the figures from current run outputs.")
    lines.append("- The current legacy Pareto sweep does not produce a meaningful front: several weight settings collapse to the same live point.")
    if not has_rule_based:
        lines.append("- A rule-based baseline is still missing from the currently available live CSV outputs, so any claim that depends on that baseline is not yet fully supported by current artifacts.")
    else:
        lines.append("- A rule-based baseline is now available in the live CSV pipeline and is included in the baseline comparison.")
    lines.append("")
    lines.append("## Current Live Baseline Metrics")
    lines.append("")

    if baseline_summary.empty:
        lines.append("No baseline summary CSVs were found.")
    else:
        ordered = [p for p in POLICY_ORDER if p in available_policies]
        frame = baseline_summary.set_index("policy").loc[ordered].reset_index()
        lines.append("| Policy | Mean HVAC power (W) | Mean comfort penalty | 95th percentile comfort penalty | Mean zone temp (C) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for _, row in frame.iterrows():
            lines.append(
                f"| {POLICY_LABELS.get(row['policy'], row['policy'])} | "
                f"{row['mean_hvac_power_w_mean']:.2f} | "
                f"{row['mean_comfort_penalty_mean']:.3f} | "
                f"{row['p95_comfort_penalty_mean']:.3f} | "
                f"{row['mean_zone_temp_c_mean']:.3f} |"
            )
    lines.append("")
    lines.append("## Current Live Pareto Observation")
    lines.append("")

    if pareto_summary.empty:
        lines.append("No live Pareto summary CSV was found.")
    else:
        ordered = [r for r in RUN_ORDER if r in set(pareto_summary["run_name"])]
        frame = pareto_summary.set_index("run_name").loc[ordered].reset_index()
        lines.append("| Run | Mean HVAC power (W) | Mean comfort penalty | 95th percentile comfort penalty | Mean zone temp (C) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for _, row in frame.iterrows():
            lines.append(
                f"| {RUN_LABELS.get(row['run_name'], row['run_name'])} | "
                f"{row['mean_hvac_power_w_mean']:.2f} | "
                f"{row['mean_comfort_penalty_mean']:.3f} | "
                f"{row['p95_comfort_penalty_mean']:.3f} | "
                f"{row['mean_zone_temp_c_mean']:.3f} |"
            )

        collapsed = frame[
            np.isclose(frame["mean_hvac_power_w_mean"], frame["mean_hvac_power_w_mean"].iloc[0], atol=1e-6)
            & np.isclose(frame["mean_comfort_penalty_mean"], frame["mean_comfort_penalty_mean"].iloc[0], atol=1e-6)
        ]
        lines.append("")
        if len(collapsed) >= 2:
            runs = ", ".join(RUN_LABELS.get(r, r) for r in collapsed["run_name"].tolist())
            lines.append(f"Current live evidence shows Pareto collapse for: **{runs}**.")
        else:
            lines.append("Current live evidence shows distinct Pareto points.")

    lines.append("")
    lines.append("## Honest Interpretation")
    lines.append("")
    lines.append("- The current legacy Sinergym branch can be reproduced honestly with the live CSV pipeline.")
    lines.append("- That honest reproduction should be reported as distinct from the older article-era figure package.")
    lines.append("- The correct wording is:")
    lines.append("  - current live legacy reproduction differs from article figures")
    lines.append("  - article-era figures were not fully generated from current CSV pipeline")
    lines.append("  - we rebuilt an honest live evaluation stack")
    lines.append("")
    lines.append("## Regeneration Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("python legacy_sinergym_main.py --mode baselines --seeds 42,43,44 --baseline-steps 2000")
    lines.append("python evaluation/visualize_results_live_sinergym.py --base-dir /app/outputs/legacy_sinergym --out-dir /app/outputs/legacy_sinergym/live_figures")
    lines.append("```")

    out_path = out_dir / "live_sinergym_honest_reproduction.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_metrics, traces = collect_default_policy_metrics(base_dir)
    baseline_summary = summarize_metrics(default_metrics, "policy") if not default_metrics.empty else pd.DataFrame()
    if not default_metrics.empty:
        default_metrics.to_csv(out_dir / "live_baseline_seed_metrics.csv", index=False)
    if not baseline_summary.empty:
        baseline_summary.to_csv(out_dir / "live_baseline_summary.csv", index=False)

    pareto_metrics = collect_live_pareto_metrics(base_dir)
    pareto_summary = summarize_metrics(pareto_metrics, "run_name") if not pareto_metrics.empty else pd.DataFrame()
    if not pareto_metrics.empty:
        pareto_metrics.to_csv(out_dir / "live_pareto_seed_metrics.csv", index=False)
    if not pareto_summary.empty:
        pareto_summary.to_csv(out_dir / "live_pareto_summary.csv", index=False)

    generated: list[Path] = []
    for path in (
        plot_baseline_comparison(baseline_summary, out_dir),
        plot_representative_trajectories(traces, args.representative_seed, out_dir),
        plot_live_pareto(pareto_summary, out_dir),
    ):
        if path is not None:
            generated.append(path)

    summary_path = write_summary(base_dir, baseline_summary, pareto_summary, out_dir)
    generated.append(summary_path)
    report_path = write_honest_reproduction_report(base_dir, baseline_summary, pareto_summary, out_dir)
    generated.append(report_path)

    print("LIVE SINERGYM FIGURES COMPLETE")
    print("==============================")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
