from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from surrogate.inverse_problem_boptest_v35 import _build_training_index_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final Hou-and-Evins closure artifacts for sample size, splits, and positioning."
    )
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _block1_metric(df: pd.DataFrame, category: str, variant: str, metric: str) -> float:
    row = df[(df["category"] == category) & (df["variant"] == variant) & (df["metric"] == metric)]
    return float(row["value"].iloc[0])


def build_sample_size_justification_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    block1_df = pd.read_csv(REPO_ROOT / "reports" / "block1_surrogate_final_metrics.csv")

    v3_df = pd.read_csv(REPO_ROOT / "data" / "surrogate_v2" / "boptest_v2_tsupply.csv")
    v3_summary = pd.read_csv(REPO_ROOT / "outputs" / "bestest_air_article7_style_15min" / "summary.csv")
    v3_therm = v3_summary[v3_summary["controller"] == "thermostatic"].copy()
    rows.append(
        {
            "dataset_id": "v3_hourly_direct_tsup",
            "rows": int(len(v3_df)),
            "episodes_or_groups": int(v3_df[["season", "policy"]].drop_duplicates().shape[0]),
            "step_sec": 3600,
            "new_boptest_collection_minutes": np.nan,
            "primary_metric_name": "thermostatic_m_s_mean",
            "primary_metric_value": float(v3_therm["m_s"].mean()),
            "secondary_metric_name": "thermostatic_energy_kwh_mean",
            "secondary_metric_value": float(v3_therm["energy_kwh"].mean()),
            "decision": "retain",
            "justification": (
                "Broadest seasonal-policy coverage and strongest pure-control baseline; "
                "retained as the control-oriented surrogate corpus."
            ),
        }
    )

    prepared_summary = _read_json(REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "prepared_15min_dataset" / "dataset_summary.json")
    rows.append(
        {
            "dataset_id": "v35_prepared_15min_bootstrap",
            "rows": int(prepared_summary["rows"]),
            "episodes_or_groups": int(prepared_summary["episodes"]),
            "step_sec": int(float(prepared_summary["step_sec_unique"][0])),
            "new_boptest_collection_minutes": 0.0,
            "primary_metric_name": "calibrated_rollout_24h_rmse_c",
            "primary_metric_value": _block1_metric(block1_df, "prepared_rollout", "calibrated_v35", "rollout_24h_rmse"),
            "secondary_metric_name": "calibrated_inverse_rmse_c",
            "secondary_metric_value": _block1_metric(block1_df, "inverse_calibration", "best_temp_alignment", "calibrated_rmse"),
            "decision": "retain",
            "justification": (
                "Smallest 15-minute corpus that already supports explicit C_zon identification "
                "and the canonical calibrated v3.5 backend without additional live collection cost."
            ),
        }
    )

    collected_summary = _read_json(REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "collected_15min_dataset" / "dataset_summary.json")
    collected_launcher = _read_json(REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "collected_15min_baseline" / "launcher_summary.json")
    rows.append(
        {
            "dataset_id": "collected_15min_exploration",
            "rows": int(collected_summary["rows"]),
            "episodes_or_groups": int(collected_summary["episodes"]),
            "step_sec": int(float(collected_summary["step_sec_unique"][0])),
            "new_boptest_collection_minutes": float(collected_summary["elapsed_min"]),
            "primary_metric_name": "h4_rmse_c",
            "primary_metric_value": float(collected_launcher["safety_metrics"]["h4_rmse"]),
            "secondary_metric_name": "h4_false_safe_pct",
            "secondary_metric_value": float(collected_launcher["safety_metrics"]["h4_false_safe_pct"]),
            "decision": "reject_as_canonical",
            "justification": (
                "Much larger 15-minute corpus required new BOPTEST collection time, "
                "but still underperformed the prepared bootstrap corpus on rollout-safety metrics."
            ),
        }
    )

    focus_summary = _read_json(REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "collected_15min_focus_dataset" / "dataset_summary.json")
    focus_launcher = _read_json(REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "collected_15min_focus" / "launcher_summary.json")
    rows.append(
        {
            "dataset_id": "collected_15min_focus",
            "rows": int(focus_summary["rows"]),
            "episodes_or_groups": int(focus_summary["episodes"]),
            "step_sec": int(float(focus_summary["step_sec_unique"][0])),
            "new_boptest_collection_minutes": float(focus_summary["elapsed_min"]),
            "primary_metric_name": "h4_rmse_c",
            "primary_metric_value": float(focus_launcher["safety_metrics"]["h4_rmse"]),
            "secondary_metric_name": "h4_false_safe_pct",
            "secondary_metric_value": float(focus_launcher["safety_metrics"]["h4_false_safe_pct"]),
            "decision": "reject_as_canonical",
            "justification": (
                "More targeted heating-focused recollection reduced cost relative to the full collected corpus, "
                "but still did not beat the prepared bootstrap corpus on 1-hour safety fidelity."
            ),
        }
    )

    return pd.DataFrame(rows)


def build_split_representativeness_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    v3_df = pd.read_csv(REPO_ROOT / "data" / "surrogate_v2" / "boptest_v2_tsupply.csv")
    n_v3 = len(v3_df)
    n_train_v3 = max(1, int(n_v3 * 0.8))
    v3_train = v3_df.iloc[:n_train_v3].copy()
    v3_val = v3_df.iloc[n_train_v3:].copy()
    rows.append(
        {
            "pipeline": "v3_hourly_control_backbone",
            "split_mode": "row_index_contiguous_80_20",
            "train_rows": int(len(v3_train)),
            "val_rows": int(len(v3_val)),
            "test_definition": "external 15-minute BOPTEST benchmark scenarios",
            "train_group_coverage": int(v3_train[["season", "policy"]].drop_duplicates().shape[0]),
            "val_group_coverage": int(v3_val[["season", "policy"]].drop_duplicates().shape[0]),
            "train_season_values": ",".join(sorted(v3_train["season"].astype(str).unique().tolist())),
            "val_season_values": ",".join(sorted(v3_val["season"].astype(str).unique().tolist())),
            "train_policy_values": ",".join(sorted(v3_train["policy"].astype(str).unique().tolist())),
            "val_policy_values": ",".join(sorted(v3_val["policy"].astype(str).unique().tolist())),
            "representativeness_assessment": "limited",
            "assessment_note": (
                "Validation keeps all control policies but only covers the tail of the ordered corpus "
                "(autumn-only by season), so final claims rely on external BOPTEST benchmarks, not this split alone."
            ),
        }
    )

    prepared_df = pd.read_csv(REPO_ROOT / "data" / "block_1_2_surrogate_rmse" / "boptest_block12_15min_prepared.csv")
    _, train_all_idx, val_idx, split_summary = _build_training_index_sets(
        prepared_df,
        val_split=0.2,
        excitation_quantile=0.95,
        excitation_mix_ratio=1.0,
        excitation_mode="dt_only",
        seed=42,
    )
    prepared_train = prepared_df.iloc[train_all_idx].copy()
    prepared_val = prepared_df.iloc[val_idx].copy()
    rows.append(
        {
            "pipeline": "v35_inverse_canonical_power_head_only",
            "split_mode": "episode_id_80_20_plus_excitation_subselection",
            "train_rows": int(len(prepared_train)),
            "val_rows": int(len(prepared_val)),
            "test_definition": "prepared rollout validation plus live thermostatic BOPTEST transfer benchmark",
            "train_group_coverage": int(prepared_train["episode_id"].astype(str).nunique()),
            "val_group_coverage": int(prepared_val["episode_id"].astype(str).nunique()),
            "train_season_values": ",".join(sorted(prepared_train["season"].astype(str).unique().tolist())),
            "val_season_values": ",".join(sorted(prepared_val["season"].astype(str).unique().tolist())),
            "train_policy_values": ",".join(sorted(prepared_train["policy"].astype(str).unique().tolist())),
            "val_policy_values": ",".join(sorted(prepared_val["policy"].astype(str).unique().tolist())),
            "representativeness_assessment": "moderate",
            "assessment_note": (
                "Both heating scenarios are present in train and val; val is smaller and thermostatic-only, "
                "but the final branch is still tested downstream on live BOPTEST transfer."
            ),
        }
    )

    rows.append(
        {
            "pipeline": "hybrid_thermostatic_downstream",
            "split_mode": "no_internal_dataset_split",
            "train_rows": np.nan,
            "val_rows": np.nan,
            "test_definition": "two unseen live BOPTEST heating scenarios: peak_heat_window and typical_heat_window",
            "train_group_coverage": np.nan,
            "val_group_coverage": np.nan,
            "train_season_values": "",
            "val_season_values": "",
            "train_policy_values": "",
            "val_policy_values": "",
            "representativeness_assessment": "external_tested",
            "assessment_note": (
                "Downstream control claims are based on explicit out-of-training BOPTEST evaluation rather than a held-out "
                "supervised split, which is the more relevant test for controller utility."
            ),
        }
    )

    return pd.DataFrame(rows)


def build_targeted_sensitivity_table() -> pd.DataFrame:
    feature_df = pd.read_csv(REPO_ROOT / "reports" / "hou_evins_feature_justification_table.csv")
    hybrid_paths = {
        "l005": _resolve_existing_path(
            [
                REPO_ROOT / "outputs" / "block2_thermostatic_hybrid_v3_v35_l005" / "summary.csv",
                REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_hybrid_v3_v35_l005" / "summary.csv",
            ]
        ),
        "l010": _resolve_existing_path(
            [
                REPO_ROOT / "outputs" / "block2_thermostatic_hybrid_v3_v35_l010" / "summary.csv",
                REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_hybrid_v3_v35_l010" / "summary.csv",
            ]
        ),
        "l015": _resolve_existing_path(
            [
                REPO_ROOT / "outputs" / "block2_thermostatic_hybrid_v3_v35_l015" / "summary.csv",
                REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_hybrid_v3_v35_l015" / "summary.csv",
            ]
        ),
    }

    def _mean_ms(path: Path) -> float:
        df = pd.read_csv(path)
        return float(df["m_s"].mean())

    hybrid_scores = {label: _mean_ms(path) for label, path in hybrid_paths.items()}
    best_hybrid = min(hybrid_scores, key=hybrid_scores.get)

    feature_best = feature_df.sort_values(["peak_m_s", "typical_m_s"]).iloc[0]

    rows = [
        {
            "sensitivity_axis": "observation_ablation",
            "tested_values": "none,no_power,no_delta_t,no_power_no_delta_t,no_prev_action,no_temp_history",
            "selection_metric": "thermostatic transfer/control stability",
            "winner": str(feature_best["obs_ablation"]),
            "numerical_reason": (
                f"Best direct v3.5 diagnostic row was {feature_best['variant']} "
                f"with peak m_s={feature_best['peak_m_s']:.4f} and typical m_s={feature_best['typical_m_s']:.4f}."
            ),
            "positioning": "targeted_sensitivity_analysis",
        },
        {
            "sensitivity_axis": "feature_encoding",
            "tested_values": "raw,causal_smooth,clipped_log,comfort_centered",
            "selection_metric": "reduced transfer-gap without reopening full model search",
            "winner": "power=clipped_log,t_zone=raw,delta=removed",
            "numerical_reason": (
                "Encoding tests were evaluated through direct v3.5 transfer diagnostics and then reused in the "
                "canonical hybrid branch instead of running a global architecture search."
            ),
            "positioning": "targeted_sensitivity_analysis",
        },
        {
            "sensitivity_axis": "hybrid_lambda_temp_disagree",
            "tested_values": "0.05,0.10,0.15",
            "selection_metric": "mean thermostatic m_s across peak/typical",
            "winner": best_hybrid,
            "numerical_reason": (
                f"lambda {best_hybrid.replace('l', '0.')} achieved the best mean m_s among the tested values "
                f"({hybrid_scores[best_hybrid]:.4f})."
            ),
            "positioning": "targeted_sensitivity_analysis",
        },
    ]
    return pd.DataFrame(rows)


def write_positioning_note(out_path: Path, sample_df: pd.DataFrame, split_df: pd.DataFrame, sensitivity_df: pd.DataFrame) -> None:
    prepared = sample_df[sample_df["dataset_id"] == "v35_prepared_15min_bootstrap"].iloc[0]
    collected = sample_df[sample_df["dataset_id"] == "collected_15min_exploration"].iloc[0]
    split_v3 = split_df[split_df["pipeline"] == "v3_hourly_control_backbone"].iloc[0]
    hybrid_row = sensitivity_df[sensitivity_df["sensitivity_axis"] == "hybrid_lambda_temp_disagree"].iloc[0]
    text = f"""# Hou-and-Evins Final Open-Items Closure

Date: 2026-04-30

## Scope

This note closes the last previously open methodology items:

1. explicit sample-size justification
2. explicit split representativeness table
3. explicit paper positioning: formal HPO vs targeted sensitivity analysis

## Generated Artifacts

- [hou_evins_sample_size_justification_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_sample_size_justification_table.csv)
- [hou_evins_split_representativeness_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_split_representativeness_table.csv)
- [hou_evins_targeted_sensitivity_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_targeted_sensitivity_table.csv)

## 1. Sample-Size Justification

The project now makes the dataset-size decision explicit rather than implicit.

Key justification:

- the prepared 15-minute bootstrap corpus uses only {int(prepared['rows'])} rows and no new live BOPTEST collection time
- despite that smaller size, it supports the canonical calibrated `v3.5` backend with:
  - inverse RMSE `{prepared['secondary_metric_value']:.4f} C`
  - 24-hour rollout RMSE `{prepared['primary_metric_value']:.4f} C`
- the much larger collected 15-minute exploration corpus uses {int(collected['rows'])} rows and {collected['new_boptest_collection_minutes']:.2f} minutes of new BOPTEST collection
- yet it still underperforms on 1-hour safety fidelity:
  - `h4_rmse = {collected['primary_metric_value']:.4f} C`

This closes the sample-size argument as a cost-vs-accuracy decision:

- broad hourly data for `v3`
- compact prepared 15-minute bootstrap for canonical `v3.5`
- larger collected 15-minute corpora kept as robustness experiments, not canonical corpora

## 2. Split Representativeness

The project now makes split strategy and representativeness explicit.

Important nuance:

- the legacy `v3` hourly control corpus uses a contiguous row split
- this leaves validation with all policies but only the autumn tail of the corpus
- that limitation is now frozen explicitly in the split table

Therefore the paper should state clearly:

- one-step supervised validation is not the only judge
- final surrogate usefulness is validated through external prepared-rollout checks and live BOPTEST transfer benchmarks

## 3. HPO Positioning

The paper should **not** claim formal HPO.

The correct and defensible framing is:

- **targeted sensitivity analysis**

This is now explicit in the sensitivity table:

- observation ablations
- feature-encoding sweeps
- hybrid `lambda_temp_disagree` sweep

The canonical hybrid winner remains:

- `{hybrid_row['winner']}`

## Result

The Hou-and-Evins packaging is now complete enough for the thermostatic branch.

The remaining work is no longer methodology closure.

The remaining work is controller-family promotion:

1. `HDRL` on canonical `hybrid_l010`
2. then `MORL` on the same hybrid default

"""
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    reports_dir = REPO_ROOT / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    sample_df = build_sample_size_justification_table()
    split_df = build_split_representativeness_table()
    sensitivity_df = build_targeted_sensitivity_table()

    sample_df.to_csv(reports_dir / "hou_evins_sample_size_justification_table.csv", index=False)
    split_df.to_csv(reports_dir / "hou_evins_split_representativeness_table.csv", index=False)
    sensitivity_df.to_csv(reports_dir / "hou_evins_targeted_sensitivity_table.csv", index=False)
    write_positioning_note(
        reports_dir / "hou_evins_final_open_items_closure.md",
        sample_df=sample_df,
        split_df=split_df,
        sensitivity_df=sensitivity_df,
    )


if __name__ == "__main__":
    main()
