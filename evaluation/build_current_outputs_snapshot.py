from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "outputs" / "current"
SURROGATE_DIR = OUT_ROOT / "surrogate"
BENCHMARK_DIR = OUT_ROOT / "benchmark"
MORL_DIR = OUT_ROOT / "morl"

T_LOW = 21.0
T_HIGH = 25.0
STEP_SEC = 3600.0


def ensure_dirs() -> None:
    for path in (OUT_ROOT, SURROGATE_DIR, BENCHMARK_DIR, MORL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def compute_violation_and_ms(temps: pd.Series) -> tuple[float, float]:
    values = temps.astype(float).to_numpy()
    below = values < T_LOW
    above = values > T_HIGH
    violation = below | above
    r_time = float(violation.mean())
    under = ((T_LOW - values) / T_LOW).clip(min=0.0)
    over = ((values - T_HIGH) / T_HIGH).clip(min=0.0)
    r_sev = float(max(under.max(initial=0.0), over.max(initial=0.0)))
    return r_time * 100.0, r_time + r_sev


def load_yearly_trace_metrics(controller: str) -> pd.DataFrame:
    if controller == "pi":
        patterns = [
            REPO_ROOT / "outputs" / "standard_controller_scenario_*.csv",
            REPO_ROOT / "draft" / "output_archive" / "standard_controller_scenario_*.csv",
        ]
        temp_col = "t_zone"
        power_col = "p_total"
        name_from = lambda path: path.stem.replace("standard_controller_scenario_", "")
    elif controller == "thermostatic":
        patterns = [
            REPO_ROOT / "outputs" / "thermostatic_scenario_*.csv",
            REPO_ROOT / "draft" / "output_archive" / "thermostatic_scenario_*.csv",
        ]
        temp_col = "t_zone"
        power_col = "p_total"
        name_from = lambda path: path.stem.replace("thermostatic_scenario_", "")
    elif controller == "hdrl":
        patterns = [
            REPO_ROOT / "outputs" / "hdrl_scenario_*.csv",
            REPO_ROOT / "draft" / "output_archive" / "hdrl_scenario_*.csv",
        ]
        temp_col = "temp"
        power_col = "power"
        name_from = lambda path: path.stem.replace("hdrl_scenario_", "")
    else:
        raise ValueError(f"Unsupported controller: {controller}")

    rows: list[dict[str, float | str]] = []
    for pattern in patterns:
        matches = sorted(pattern.parent.glob(pattern.name))
        if not matches:
            continue
        for path in matches:
            df = pd.read_csv(path)
            viol_pct, ms = compute_violation_and_ms(df[temp_col])
            energy_kwh = float(df[power_col].astype(float).sum() * (STEP_SEC / 3600.0) / 1000.0)
            rows.append(
                {
                    "controller": controller,
                    "scenario": name_from(path),
                    "energy_kwh": energy_kwh,
                    "viol_pct": viol_pct,
                    "ms": ms,
                }
            )
        break
    if not rows:
        raise FileNotFoundError(f"No yearly trace files found for controller={controller}")
    return pd.DataFrame(rows)


def load_morl_yearly_metrics() -> pd.DataFrame:
    path = REPO_ROOT / "outputs" / "morl_surrogate_ppo_v35_calibrated" / "seed42" / "yearly_eval" / "morl_yearly_summary.csv"
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "controller": "morl",
            "scenario": df["name"],
            "energy_kwh": df["energy_kwh"].astype(float),
            "viol_pct": df["viol_pct"].astype(float),
            "ms": df["ms"].astype(float),
        }
    )


def build_morl_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    pi_df = load_yearly_trace_metrics("pi")
    thermostatic_df = load_yearly_trace_metrics("thermostatic")
    hdrl_df = load_yearly_trace_metrics("hdrl")
    morl_df = load_morl_yearly_metrics()

    all_df = pd.concat([pi_df, thermostatic_df, hdrl_df, morl_df], ignore_index=True)
    summary = (
        all_df.groupby("controller", as_index=False)[["energy_kwh", "viol_pct", "ms"]]
        .mean()
        .sort_values("controller")
        .reset_index(drop=True)
    )

    morl_row = summary.loc[summary["controller"] == "morl"].iloc[0]
    compare_rows = []
    for baseline in ("pi", "thermostatic", "hdrl"):
        base_row = summary.loc[summary["controller"] == baseline].iloc[0]
        compare_rows.append(
            {
                "baseline_controller": baseline,
                "morl_energy_kwh_mean": float(morl_row["energy_kwh"]),
                "baseline_energy_kwh_mean": float(base_row["energy_kwh"]),
                "energy_saved_kwh_mean": float(base_row["energy_kwh"] - morl_row["energy_kwh"]),
                "energy_saved_pct": float((base_row["energy_kwh"] - morl_row["energy_kwh"]) / base_row["energy_kwh"] * 100.0),
                "morl_viol_pct_mean": float(morl_row["viol_pct"]),
                "baseline_viol_pct_mean": float(base_row["viol_pct"]),
                "viol_reduction_pct_points": float(base_row["viol_pct"] - morl_row["viol_pct"]),
                "viol_reduction_pct_relative": float((base_row["viol_pct"] - morl_row["viol_pct"]) / base_row["viol_pct"] * 100.0),
                "morl_ms_mean": float(morl_row["ms"]),
                "baseline_ms_mean": float(base_row["ms"]),
                "ms_reduction_abs": float(base_row["ms"] - morl_row["ms"]),
                "ms_reduction_pct": float((base_row["ms"] - morl_row["ms"]) / base_row["ms"] * 100.0),
            }
        )
    return summary, pd.DataFrame(compare_rows)


def write_current_readme() -> None:
    readme = """# Current Outputs

This folder is the canonical snapshot of the current active research line.

## Structure

- `surrogate/`
  Canonical Block 1 fidelity outputs and inverse-calibration summaries.
- `benchmark/`
  Current benchmark-facing summaries, including MORL yearly comparison against PI / thermostatic / HDRL.
- `morl/`
  Current calibrated-twin MORL pipeline artifacts and stage summaries.

## Intent

This folder does not replace the original experiment folders. It gathers the current active outputs in one stable location for reporting and presentation.
"""
    (OUT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "surrogate_sources": {
            "surrogate_rollout_benchmark": "results/research_benchmark/tables/surrogate_rollout_benchmark.csv",
            "inverse_calibration_benchmark": "results/research_benchmark/tables/inverse_calibration_benchmark.csv",
            "rollout_compare_summary": "outputs/surrogate_v35_rollout_live/v35_compare_summary.csv",
            "calibration_summary": "outputs/surrogate_v35_inverse_boptest_prior420_heads_only/calibration_summary_boptest_v35.json",
        },
        "benchmark_sources": {
            "article7_style_15min": "outputs/bestest_air_article7_style_15min/summary.csv",
            "controller_benchmark": "results/research_benchmark/tables/controller_benchmark.csv",
            "pi_yearly_trace_dir": "outputs/standard_controller_scenario_*.csv or draft/output_archive/standard_controller_scenario_*.csv",
            "thermostatic_yearly_trace_dir": "outputs/thermostatic_scenario_*.csv or draft/output_archive/thermostatic_scenario_*.csv",
            "hdrl_yearly_trace_dir": "outputs/hdrl_scenario_*.csv or draft/output_archive/hdrl_scenario_*.csv",
            "morl_yearly_summary": "outputs/morl_surrogate_ppo_v35_calibrated/seed42/yearly_eval/morl_yearly_summary.csv",
        },
        "morl_sources": {
            "pipeline_manifest": "outputs/morl_surrogate_ppo_v35_calibrated/seed42/pipeline_manifest.json",
            "pretrain_config": "outputs/morl_surrogate_ppo_v35_calibrated/seed42/pretrain/config_snapshot.json",
            "finetune_config": "outputs/morl_surrogate_ppo_v35_calibrated/seed42/finetune_boptest/config_snapshot.json",
            "yearly_summary": "outputs/morl_surrogate_ppo_v35_calibrated/seed42/yearly_eval/morl_yearly_summary.csv",
        },
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()

    # Surrogate block
    copy_file(
        REPO_ROOT / "results" / "research_benchmark" / "tables" / "surrogate_rollout_benchmark.csv",
        SURROGATE_DIR / "surrogate_rollout_benchmark.csv",
    )
    copy_file(
        REPO_ROOT / "results" / "research_benchmark" / "tables" / "inverse_calibration_benchmark.csv",
        SURROGATE_DIR / "inverse_calibration_benchmark.csv",
    )
    copy_file(
        REPO_ROOT / "outputs" / "surrogate_v35_rollout_live" / "v35_compare_summary.csv",
        SURROGATE_DIR / "v35_compare_summary.csv",
    )
    copy_file(
        REPO_ROOT / "outputs" / "surrogate_v35_inverse_boptest_prior420_heads_only" / "calibration_summary_boptest_v35.json",
        SURROGATE_DIR / "calibration_summary_boptest_v35.json",
    )

    # Benchmark block
    copy_file(
        REPO_ROOT / "outputs" / "bestest_air_article7_style_15min" / "summary.csv",
        BENCHMARK_DIR / "article7_style_15min_summary.csv",
    )
    copy_file(
        REPO_ROOT / "results" / "research_benchmark" / "tables" / "controller_benchmark.csv",
        BENCHMARK_DIR / "controller_benchmark.csv",
    )

    yearly_means_df, morl_compare_df = build_morl_comparison()
    yearly_means_df.to_csv(BENCHMARK_DIR / "yearly_controller_means_aligned.csv", index=False)
    morl_compare_df.to_csv(BENCHMARK_DIR / "morl_vs_baselines_yearly.csv", index=False)

    # MORL block
    copy_file(
        REPO_ROOT / "outputs" / "morl_surrogate_ppo_v35_calibrated" / "seed42" / "pipeline_manifest.json",
        MORL_DIR / "pipeline_manifest.json",
    )
    copy_file(
        REPO_ROOT / "outputs" / "morl_surrogate_ppo_v35_calibrated" / "seed42" / "pretrain" / "config_snapshot.json",
        MORL_DIR / "pretrain_config_snapshot.json",
    )
    copy_file(
        REPO_ROOT / "outputs" / "morl_surrogate_ppo_v35_calibrated" / "seed42" / "finetune_boptest" / "config_snapshot.json",
        MORL_DIR / "finetune_boptest_config_snapshot.json",
    )
    copy_file(
        REPO_ROOT / "outputs" / "morl_surrogate_ppo_v35_calibrated" / "seed42" / "yearly_eval" / "morl_yearly_summary.csv",
        MORL_DIR / "morl_yearly_summary.csv",
    )

    write_current_readme()
    write_manifest()

    print(f"Current outputs snapshot built at: {OUT_ROOT}")


if __name__ == "__main__":
    main()
