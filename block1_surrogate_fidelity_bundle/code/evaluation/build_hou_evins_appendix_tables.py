from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Hou-and-Evins-style appendix tables from current artifacts.")
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_sample_generation_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    v3_csv = REPO_ROOT / "data" / "surrogate_v2" / "boptest_v2_tsupply.csv"
    v3_df = pd.read_csv(v3_csv)
    rows.append(
        {
            "dataset_id": "v3_hourly_direct_tsup",
            "source_file": str(v3_csv.relative_to(REPO_ROOT)),
            "rows": int(len(v3_df)),
            "episodes_or_groups": int(v3_df[["policy", "season"]].drop_duplicates().shape[0]),
            "step_sec": 3600,
            "controller_or_policy_mix": ",".join(sorted(v3_df["policy"].astype(str).unique().tolist())),
            "season_or_scenario_mix": ",".join(sorted(v3_df["season"].astype(str).unique().tolist())),
            "t_zone_min_c": float(v3_df["t_zone"].min()),
            "t_zone_max_c": float(v3_df["t_zone"].max()),
            "t_amb_min_c": float(v3_df["t_amb"].min()),
            "t_amb_max_c": float(v3_df["t_amb"].max()),
            "power_min_w": float(v3_df["p_total"].min()),
            "power_max_w": float(v3_df["p_total"].max()),
            "intended_role": "broad seasonal direct-TSup control surrogate training corpus",
            "article_status": "control-oriented base corpus",
        }
    )

    prepared_json = REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "prepared_15min_dataset" / "dataset_summary.json"
    prepared = _read_json(prepared_json)
    rows.append(
        {
            "dataset_id": "v35_prepared_15min_bootstrap",
            "source_file": "data/block_1_2_surrogate_rmse/boptest_block12_15min_prepared.csv",
            "rows": int(prepared["rows"]),
            "episodes_or_groups": int(prepared["episodes"]),
            "step_sec": int(float(prepared["step_sec_unique"][0])),
            "controller_or_policy_mix": ",".join(prepared["controllers"]),
            "season_or_scenario_mix": ",".join(prepared["scenarios"]),
            "t_zone_min_c": float(prepared["t_zone_range_c"][0]),
            "t_zone_max_c": float(prepared["t_zone_range_c"][1]),
            "t_amb_min_c": float(prepared["t_amb_range_c"][0]),
            "t_amb_max_c": float(prepared["t_amb_range_c"][1]),
            "power_min_w": float(prepared["power_range_w"][0]),
            "power_max_w": float(prepared["power_range_w"][1]),
            "intended_role": "15-minute closed-loop bootstrap dataset for initial rollout-oriented calibration",
            "article_status": "controller-biased bootstrap corpus",
        }
    )

    collected_json = REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse" / "collected_15min_dataset" / "dataset_summary.json"
    collected = _read_json(collected_json)
    rows.append(
        {
            "dataset_id": "v35_collected_15min_exploration",
            "source_file": "data/block_1_2_surrogate_rmse/boptest_block12_15min_collected.csv",
            "rows": int(collected["rows"]),
            "episodes_or_groups": int(collected["episodes"]),
            "step_sec": int(float(collected["step_sec_unique"][0])),
            "controller_or_policy_mix": ",".join(collected["policy_values"]),
            "season_or_scenario_mix": ",".join(collected["season_values"]),
            "t_zone_min_c": float(collected["t_zone_range_c"][0]),
            "t_zone_max_c": float(collected["t_zone_range_c"][1]),
            "t_amb_min_c": float(collected["t_amb_range_c"][0]),
            "t_amb_max_c": float(collected["t_amb_range_c"][1]),
            "power_min_w": float(collected["p_total_range_w"][0]),
            "power_max_w": float(collected["p_total_range_w"][1]),
            "intended_role": "broader 15-minute exploration dataset for transfer-stability analysis",
            "article_status": "exploratory robustness corpus",
        }
    )
    return pd.DataFrame(rows)


def build_stage_a_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage_a_operation": "latency compensation search",
                "implementation": "_undo_delay_array over candidate lags",
                "parameter_or_rule": "lag in [0, max_latency_search]",
                "selection_criterion": "choose lag minimizing surrogate one-step RMSE against observed next-step temperature",
                "purpose": "undo thermostat/telemetry delay before calibration",
            },
            {
                "stage_a_operation": "temperature bias removal",
                "implementation": "median residual offset subtraction",
                "parameter_or_rule": "bias_est = median(t_obs_next - t_pred)",
                "selection_criterion": "center residual distribution before Stage B/C calibration",
                "purpose": "avoid fitting static sensor bias as building physics",
            },
            {
                "stage_a_operation": "power affine normalization",
                "implementation": "least-squares fit p_obs ≈ scale * p_pred + bias",
                "parameter_or_rule": "scale_est, bias_p_est from linear solve",
                "selection_criterion": "reduce power-channel mismatch before nonlinear head refinement",
                "purpose": "separate systematic meter scaling from dynamic error",
            },
            {
                "stage_a_operation": "rolling denoise",
                "implementation": "rolling median + exponential moving average",
                "parameter_or_rule": "smooth_window = 5",
                "selection_criterion": "suppress high-frequency telemetry noise without changing regime identity",
                "purpose": "stabilize downstream inverse calibration",
            },
            {
                "stage_a_operation": "causal delta recomputation",
                "implementation": "delta_t = t_zone_next - t_zone after preprocessing",
                "parameter_or_rule": "derived from corrected current and next temperature",
                "selection_criterion": "preserve state-transition consistency after artifact correction",
                "purpose": "avoid propagating artifact-driven temperature-history features",
            },
        ]
    )


def build_feature_justification_table() -> pd.DataFrame:
    variants = [
        {
            "variant": "with_power_raw",
            "summary": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_with_power" / "summary.csv",
            "divergence": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block13_obs_gap_with_power" / "first_divergence_summary.csv",
            "obs_ablation": "none",
            "delta_feature_mode": "raw",
            "power_feature_mode": "raw",
            "t_zone_feature_mode": "raw",
            "decision": "reference-only",
        },
        {
            "variant": "no_power",
            "summary": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_no_power" / "summary.csv",
            "divergence": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block13_obs_gap_no_power" / "first_divergence_summary.csv",
            "obs_ablation": "no_power",
            "delta_feature_mode": "raw",
            "power_feature_mode": "raw",
            "t_zone_feature_mode": "raw",
            "decision": "rejected",
        },
        {
            "variant": "no_delta_t",
            "summary": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_no_delta_t" / "summary.csv",
            "divergence": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block13_obs_gap_no_delta_t" / "first_divergence_summary.csv",
            "obs_ablation": "no_delta_t",
            "delta_feature_mode": "raw",
            "power_feature_mode": "raw",
            "t_zone_feature_mode": "raw",
            "decision": "retained-as-direction",
        },
        {
            "variant": "no_power_no_delta_t",
            "summary": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_no_power_no_delta_t" / "summary.csv",
            "divergence": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block13_obs_gap_no_power_no_delta_t" / "first_divergence_summary.csv",
            "obs_ablation": "no_power_no_delta_t",
            "delta_feature_mode": "raw",
            "power_feature_mode": "raw",
            "t_zone_feature_mode": "raw",
            "decision": "rejected",
        },
        {
            "variant": "no_prev_action",
            "summary": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_no_prev_action" / "summary.csv",
            "divergence": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block13_obs_gap_no_prev_action" / "first_divergence_summary.csv",
            "obs_ablation": "no_prev_action",
            "delta_feature_mode": "raw",
            "power_feature_mode": "raw",
            "t_zone_feature_mode": "raw",
            "decision": "rejected",
        },
        {
            "variant": "no_temp_history",
            "summary": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_no_temp_history" / "summary.csv",
            "divergence": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block13_obs_gap_no_temp_history" / "first_divergence_summary.csv",
            "obs_ablation": "no_temp_history",
            "delta_feature_mode": "raw",
            "power_feature_mode": "raw",
            "t_zone_feature_mode": "raw",
            "decision": "rejected",
        },
        {
            "variant": "causal_smooth",
            "summary": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_causal_smooth" / "summary.csv",
            "divergence": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block13_obs_gap_causal_smooth" / "first_divergence_summary.csv",
            "obs_ablation": "none",
            "delta_feature_mode": "causal_smooth",
            "power_feature_mode": "raw",
            "t_zone_feature_mode": "raw",
            "decision": "rejected",
        },
        {
            "variant": "no_delta_t_powerlog",
            "summary": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_no_delta_t_powerlog" / "summary.csv",
            "divergence": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block13_obs_gap_no_delta_t_powerlog" / "first_divergence_summary.csv",
            "obs_ablation": "no_delta_t",
            "delta_feature_mode": "raw",
            "power_feature_mode": "clipped_log",
            "t_zone_feature_mode": "raw",
            "decision": "retained-intermediate",
        },
        {
            "variant": "no_delta_t_powerlog_tzone",
            "summary": REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_no_delta_t_powerlog_tzone" / "summary.csv",
            "divergence": REPO_ROOT / "outputs" / "block13_obs_gap_no_delta_t_powerlog_tzone" / "first_divergence_summary.csv",
            "obs_ablation": "no_delta_t",
            "delta_feature_mode": "raw",
            "power_feature_mode": "clipped_log",
            "t_zone_feature_mode": "comfort_centered",
            "decision": "retained-diagnostic",
        },
    ]

    rows: list[dict[str, object]] = []
    for spec in variants:
        summary_df = pd.read_csv(spec["summary"])
        divergence_df = pd.read_csv(spec["divergence"]).set_index("scenario")
        peak = summary_df[summary_df["scenario"] == "peak_heat_window"].iloc[0]
        typical = summary_df[summary_df["scenario"] == "typical_heat_window"].iloc[0]
        peak_div = divergence_df.loc["peak_heat_window"]
        typ_div = divergence_df.loc["typical_heat_window"]
        rows.append(
            {
                "variant": spec["variant"],
                "obs_ablation": spec["obs_ablation"],
                "delta_feature_mode": spec["delta_feature_mode"],
                "power_feature_mode": spec["power_feature_mode"],
                "t_zone_feature_mode": spec["t_zone_feature_mode"],
                "peak_m_s": float(peak["m_s"]),
                "typical_m_s": float(typical["m_s"]),
                "peak_violation_pct": float(peak["violation_pct"]),
                "typical_violation_pct": float(typical["violation_pct"]),
                "peak_first_divergence_step": int(peak_div["first_divergence_step"]),
                "typical_first_divergence_step": int(typ_div["first_divergence_step"]),
                "peak_action_gap_norm": float(peak_div["action_gap_norm"]),
                "typical_action_gap_norm": float(typ_div["action_gap_norm"]),
                "peak_top_feature": str(peak_div["top_feature"]),
                "typical_top_feature": str(typ_div["top_feature"]),
                "decision": spec["decision"],
            }
        )
    return pd.DataFrame(rows)


def build_architecture_justification_table() -> pd.DataFrame:
    v3_bench = pd.read_csv(REPO_ROOT / "outputs" / "bestest_air_article7_style_15min" / "summary.csv")
    v3_transfer = pd.read_csv(REPO_ROOT / "outputs" / "block13_closed_loop_transfer_pure_v3" / "summary.csv")
    v3_div = pd.read_csv(REPO_ROOT / "outputs" / "block13_obs_gap_pure_v3" / "first_divergence_summary.csv").set_index("scenario")

    v35_block1 = pd.read_csv(REPO_ROOT / "reports" / "block1_surrogate_final_metrics.csv")
    v35_bench = pd.read_csv(
        REPO_ROOT / "draft" / "legacy_archive" / "outputs" / "block2_thermostatic_no_delta_t_powerlog_tzone" / "summary.csv"
    )
    v35_transfer = pd.read_csv(REPO_ROOT / "outputs" / "block13_closed_loop_transfer_no_delta_t_powerlog_tzone" / "summary.csv")
    v35_div = pd.read_csv(REPO_ROOT / "outputs" / "block13_obs_gap_no_delta_t_powerlog_tzone" / "first_divergence_summary.csv").set_index("scenario")

    hybrid_bench = pd.read_csv(REPO_ROOT / "outputs" / "block2_thermostatic_hybrid_v3_v35_l010" / "summary.csv")
    hybrid_transfer = pd.read_csv(REPO_ROOT / "outputs" / "block13_closed_loop_transfer_hybrid_l010" / "summary.csv")
    hybrid_div = pd.read_csv(REPO_ROOT / "outputs" / "block13_obs_gap_hybrid_l010" / "first_divergence_summary.csv").set_index("scenario")

    def _bench_row(df: pd.DataFrame, scenario: str) -> pd.Series:
        return df[df["scenario"] == scenario].iloc[0]

    rows = []
    for variant in ("v3", "v35_calibrated", "hybrid_l010"):
        if variant == "v3":
            peak_b = _bench_row(v3_bench, "peak_heat_window")
            typ_b = _bench_row(v3_bench, "typical_heat_window")
            peak_t = _bench_row(v3_transfer, "peak_heat_window")
            typ_t = _bench_row(v3_transfer, "typical_heat_window")
            rows.append(
                {
                    "variant": variant,
                    "role": "control-oriented direct-TSup surrogate",
                    "explicit_physics": "no",
                    "explicit_c_zon": "no",
                    "block1_temp_alignment_rmse_c": None,
                    "block1_rollout_24h_rmse_c": None,
                    "peak_control_m_s": float(peak_b["m_s"]),
                    "typical_control_m_s": float(typ_b["m_s"]),
                    "peak_energy_kwh": float(peak_b["energy_kwh"]),
                    "typical_energy_kwh": float(typ_b["energy_kwh"]),
                    "peak_transfer_temp_rmse_c": float(peak_t["temp_rmse_c"]),
                    "typical_transfer_temp_rmse_c": float(typ_t["temp_rmse_c"]),
                    "peak_first_divergence_step": int(v3_div.loc["peak_heat_window", "first_divergence_step"]),
                    "typical_first_divergence_step": int(v3_div.loc["typical_heat_window", "first_divergence_step"]),
                    "article_position": "best pure control baseline",
                }
            )
        elif variant == "v35_calibrated":
            peak_b = _bench_row(v35_bench, "peak_heat_window")
            typ_b = _bench_row(v35_bench, "typical_heat_window")
            peak_t = _bench_row(v35_transfer, "peak_heat_window")
            typ_t = _bench_row(v35_transfer, "typical_heat_window")
            rows.append(
                {
                    "variant": variant,
                    "role": "physics-oriented calibrated twin",
                    "explicit_physics": "yes",
                    "explicit_c_zon": "yes",
                    "block1_temp_alignment_rmse_c": float(
                        v35_block1[
                            (v35_block1["category"] == "inverse_calibration")
                            & (v35_block1["metric"] == "calibrated_rmse")
                        ]["value"].iloc[0]
                    ),
                    "block1_rollout_24h_rmse_c": float(
                        v35_block1[
                            (v35_block1["category"] == "prepared_rollout")
                            & (v35_block1["variant"] == "calibrated_v35")
                            & (v35_block1["metric"] == "rollout_24h_rmse")
                        ]["value"].iloc[0]
                    ),
                    "peak_control_m_s": float(peak_b["m_s"]),
                    "typical_control_m_s": float(typ_b["m_s"]),
                    "peak_energy_kwh": float(peak_b["energy_kwh"]),
                    "typical_energy_kwh": float(typ_b["energy_kwh"]),
                    "peak_transfer_temp_rmse_c": float(peak_t["temp_rmse_c"]),
                    "typical_transfer_temp_rmse_c": float(typ_t["temp_rmse_c"]),
                    "peak_first_divergence_step": int(v35_div.loc["peak_heat_window", "first_divergence_step"]),
                    "typical_first_divergence_step": int(v35_div.loc["typical_heat_window", "first_divergence_step"]),
                    "article_position": "best physical baseline, poor control surrogate",
                }
            )
        else:
            peak_b = _bench_row(hybrid_bench, "peak_heat_window")
            typ_b = _bench_row(hybrid_bench, "typical_heat_window")
            peak_t = _bench_row(hybrid_transfer, "peak_heat_window")
            typ_t = _bench_row(hybrid_transfer, "typical_heat_window")
            rows.append(
                {
                    "variant": variant,
                    "role": "v3 dynamics regularized by calibrated v3.5",
                    "explicit_physics": "partial",
                    "explicit_c_zon": "regularizer-only",
                    "block1_temp_alignment_rmse_c": None,
                    "block1_rollout_24h_rmse_c": None,
                    "peak_control_m_s": float(peak_b["m_s"]),
                    "typical_control_m_s": float(typ_b["m_s"]),
                    "peak_energy_kwh": float(peak_b["energy_kwh"]),
                    "typical_energy_kwh": float(typ_b["energy_kwh"]),
                    "peak_transfer_temp_rmse_c": float(peak_t["temp_rmse_c"]),
                    "typical_transfer_temp_rmse_c": float(typ_t["temp_rmse_c"]),
                    "peak_first_divergence_step": int(hybrid_div.loc["peak_heat_window", "first_divergence_step"]),
                    "typical_first_divergence_step": int(hybrid_div.loc["typical_heat_window", "first_divergence_step"]),
                    "article_position": "best verified compromise",
                }
            )
    return pd.DataFrame(rows)


def write_report(
    out_path: Path,
    sample_df: pd.DataFrame,
    stage_a_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    arch_df: pd.DataFrame,
) -> None:
    best_feature = feature_df.sort_values(["peak_m_s", "typical_m_s"]).iloc[0]
    best_arch = arch_df[arch_df["variant"] == "hybrid_l010"].iloc[0]
    report = f"""# Hou-and-Evins Partial Closure

Date: 2026-04-30

## Scope

This document closes the previously partial items:

1. sample generation as a paper-facing table
2. Stage A preprocessing as an article-facing block
3. feature significance and encoding justification as a numerical table
4. architecture justification as a comparative table

## Generated Tables

- [hou_evins_sample_generation_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_sample_generation_table.csv)
- [hou_evins_stage_a_processing_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_stage_a_processing_table.csv)
- [hou_evins_feature_justification_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_feature_justification_table.csv)
- [hou_evins_architecture_justification_table.csv](C:/Users/user/Desktop/HVAC_DRL_MORL/reports/hou_evins_architecture_justification_table.csv)

## 1. Sample Generation

The project now has an explicit sample-generation table covering:

- hourly `v3` direct-TSup corpus
- prepared 15-minute `v3.5` bootstrap corpus
- collected 15-minute exploration corpus

This closes the packaging gap around:

- dataset size
- step size
- excitation/controller mix
- seasonal/scenario coverage
- state and power ranges

## 2. Stage A Preprocessing

Stage A is now described as a concrete sequence:

- latency compensation search
- temperature bias removal
- power affine normalization
- rolling denoise
- causal delta recomputation

Each operation is now tied to:

- an explicit implementation rule
- a numerical selection criterion
- a purpose in the inverse-calibration pipeline

## 3. Feature Significance and Encoding Justification

The feature/encoding table now makes the numerical logic explicit.

Key result:

- the strongest direct `v3.5` diagnostic branch came from removing `delta_t_zone_norm`, then clipping/log-scaling power, then testing comfort-centered temperature encoding
- however the final best downstream result is not direct `v3.5`, but `hybrid_l010`

The best direct-`v3.5` diagnostic row in the table is:

- variant: `{best_feature['variant']}`
- peak `m_s = {best_feature['peak_m_s']:.4f}`
- typical `m_s = {best_feature['typical_m_s']:.4f}`

This closes the packaging gap around observation/encoding justification.

## 4. Architecture Justification

The architecture comparison is now explicit:

- `v3`: best pure control surrogate
- `v3.5 calibrated`: best physical twin, poor standalone control surrogate
- `hybrid_l010`: best verified compromise

For the canonical hybrid row:

- peak `m_s = {best_arch['peak_control_m_s']:.4f}`
- typical `m_s = {best_arch['typical_control_m_s']:.4f}`
- peak transfer RMSE = `{best_arch['peak_transfer_temp_rmse_c']:.3f} C`
- typical `first_divergence_step = {int(best_arch['typical_first_divergence_step'])}`

## Result

These four items should no longer be treated as partially documented.

They are now explicit, numerical, and article-facing.

What still remains open is the separate set of truly open Hou-and-Evins items:

- formal HPO or an explicit decision to frame the work as targeted sensitivity analysis
- sample-size justification as a cost-vs-accuracy argument
- data split representativeness as a separate paper table
"""
    out_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    reports_dir = REPO_ROOT / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    sample_df = build_sample_generation_table()
    stage_a_df = build_stage_a_table()
    feature_df = build_feature_justification_table()
    arch_df = build_architecture_justification_table()

    sample_df.to_csv(reports_dir / "hou_evins_sample_generation_table.csv", index=False)
    stage_a_df.to_csv(reports_dir / "hou_evins_stage_a_processing_table.csv", index=False)
    feature_df.to_csv(reports_dir / "hou_evins_feature_justification_table.csv", index=False)
    arch_df.to_csv(reports_dir / "hou_evins_architecture_justification_table.csv", index=False)
    write_report(
        reports_dir / "hou_evins_partial_closure.md",
        sample_df=sample_df,
        stage_a_df=stage_a_df,
        feature_df=feature_df,
        arch_df=arch_df,
    )


if __name__ == "__main__":
    main()
