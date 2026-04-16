from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

REPORT_MILESTONES = {
    "phase0_energy_reduction_pct": 28.0,
    "phase1_speedup_x": 32313.0,
    "phase1_rmse_c": 0.163,
    "phase1_r2": 0.991,
    "phase2_rmse_improvement_pct": 79.0,
    "phase2_czon_error_pct": 7.9,
    "phase3_best_ms_mean": 0.697,
    "phase3_best_ms_std": 0.368,
    "phase3_summer_ms_best_low": 0.140,
    "phase3_summer_ms_best_high": 0.187,
    "tsup_v4_mean_rmse_c": 0.84,
    "tsup_v4_winter_rmse_c": 0.78,
}

COPY_SPECS = [
    ("reports/Current_HVAC_PPO_MORL.pdf", "docs/reports/Current_HVAC_PPO_MORL.pdf", "Report"),
    ("reports/defensePart1.pdf", "docs/reports/defensePart1.pdf", "Report"),
    ("reports/defensePart2.pdf", "docs/reports/defensePart2.pdf", "Report"),
    ("outputs/figures/fig1_ms_comparison.png", "figures/paper/fig1_ms_comparison.png", "Figure"),
    ("outputs/figures/fig2_surrogate_comparison.png", "figures/paper/fig2_surrogate_comparison.png", "Figure"),
    ("outputs/figures/fig3_rollout_validation.png", "figures/paper/fig3_rollout_validation.png", "Figure"),
    ("outputs/figures/fig4_multi_seed.png", "figures/paper/fig4_multi_seed.png", "Figure"),
    ("outputs/figures/fig5_calibration.png", "figures/paper/fig5_calibration.png", "Figure"),
    ("outputs/figures/fig6_pareto_front.png", "figures/paper/fig6_pareto_front.png", "Figure"),
    ("outputs/three_model_comparison/comparison_dashboard.png", "figures/controllers/comparison_dashboard.png", "Figure"),
    ("outputs/three_model_comparison/comparison_monthly_energy.png", "figures/controllers/comparison_monthly_energy.png", "Figure"),
    ("outputs/three_model_comparison/comparison_monthly_rmse22.png", "figures/controllers/comparison_monthly_rmse22.png", "Figure"),
    ("outputs/three_model_comparison/comparison_tradeoff_scatter.png", "figures/controllers/comparison_tradeoff_scatter.png", "Figure"),
    ("outputs/surrogate_rollout_live/live_rollout_temp_metrics.png", "figures/surrogate/live_rollout_temp_metrics.png", "Figure"),
    ("outputs/surrogate_rollout_live/live_rollout_energy_metrics.png", "figures/surrogate/live_rollout_energy_metrics.png", "Figure"),
    ("outputs/surrogate_rollout_live/live_rollout_trajectory_grid.png", "figures/surrogate/live_rollout_trajectory_grid.png", "Figure"),
    ("outputs/surrogate_comfort_traces/comfort_trace_summary.png", "figures/surrogate/comfort_trace_summary.png", "Figure"),
    ("outputs/surrogate_comfort_traces/comfort_trace_grid_hdrl.png", "figures/surrogate/comfort_trace_grid_hdrl.png", "Figure"),
    ("outputs/surrogate_comfort_traces/comfort_trace_grid_thermostatic.png", "figures/surrogate/comfort_trace_grid_thermostatic.png", "Figure"),
    ("outputs/surrogate_comfort_traces/surrogate_v3_vs_boptest_parity.png", "figures/surrogate/surrogate_v3_vs_boptest_parity.png", "Figure"),
    ("outputs/surrogate_comfort_traces/surrogate_v3_vs_boptest_yearly.png", "figures/surrogate/surrogate_v3_vs_boptest_yearly.png", "Figure"),
    ("outputs/surrogate_v35_rollout_live/raw_v35/live_rollout_temp_metrics.png", "figures/calibration/v35_raw_temp_metrics.png", "Figure"),
    ("outputs/surrogate_v35_rollout_live/raw_v35/live_rollout_energy_metrics.png", "figures/calibration/v35_raw_energy_metrics.png", "Figure"),
    ("outputs/surrogate_v35_rollout_live/raw_v35/live_rollout_trajectory_grid.png", "figures/calibration/v35_raw_trajectory_grid.png", "Figure"),
    ("outputs/surrogate_v35_rollout_live/calibrated_v35/live_rollout_temp_metrics.png", "figures/calibration/v35_calibrated_temp_metrics.png", "Figure"),
    ("outputs/surrogate_v35_rollout_live/calibrated_v35/live_rollout_energy_metrics.png", "figures/calibration/v35_calibrated_energy_metrics.png", "Figure"),
    ("outputs/surrogate_v35_rollout_live/calibrated_v35/live_rollout_trajectory_grid.png", "figures/calibration/v35_calibrated_trajectory_grid.png", "Figure"),
    ("outputs/three_model_comparison/comparison_summary.csv", "tables/raw/comparison_summary.csv", "Table"),
    ("outputs/three_model_comparison/comparison_scenario_metrics.csv", "tables/raw/comparison_scenario_metrics.csv", "Table"),
    ("outputs/three_model_comparison/comparison_conclusion.txt", "tables/raw/comparison_conclusion.txt", "Text"),
    ("outputs/thermostatic_yearly_summary.csv", "tables/raw/thermostatic_yearly_summary.csv", "Table"),
    ("outputs/hdrl_yearly_summary.csv", "tables/raw/hdrl_yearly_summary.csv", "Table"),
    ("outputs/standard_controller_yearly_summary.csv", "tables/raw/standard_controller_yearly_summary.csv", "Table"),
    ("outputs/standard_controller_energy_comparison.csv", "tables/raw/standard_controller_energy_comparison.csv", "Table"),
    ("outputs/pareto/pareto_summary.csv", "tables/raw/pareto_summary.csv", "Table"),
    ("outputs/eval_multi_seed/summary.csv", "tables/raw/eval_multi_seed_summary.csv", "Table"),
    ("outputs/surrogate_rollout_live/horizon_metrics.csv", "tables/raw/surrogate_rollout_horizon_metrics.csv", "Table"),
    ("outputs/surrogate_rollout_live/episode_summary.csv", "tables/raw/surrogate_rollout_episode_summary.csv", "Table"),
    ("outputs/surrogate_comfort_traces/comfort_trace_summary.csv", "tables/raw/comfort_trace_summary.csv", "Table"),
    ("outputs/surrogate_comfort_traces/comfort_trace_metrics_hdrl.csv", "tables/raw/comfort_trace_metrics_hdrl.csv", "Table"),
    ("outputs/surrogate_comfort_traces/comfort_trace_metrics_thermostatic.csv", "tables/raw/comfort_trace_metrics_thermostatic.csv", "Table"),
    ("outputs/surrogate_v2_inverse_boptest/calibration_summary_boptest_v3.json", "tables/raw/calibration_summary_boptest_v3.json", "JSON"),
    ("outputs/surrogate_v35_inverse_boptest_multistart/multistart_summary_v35.csv", "tables/raw/multistart_summary_v35.csv", "Table"),
    ("outputs/surrogate_v35_inverse_boptest_prior420_heads_only/calibration_summary_boptest_v35.json", "tables/raw/calibration_summary_boptest_v35.json", "JSON"),
    ("outputs/surrogate_v35_inverse_boptest_prior420_heads_only/excitation_summary.json", "tables/raw/v35_excitation_summary.json", "JSON"),
    ("outputs/surrogate_v35_rollout_live/v35_compare_summary.csv", "tables/raw/v35_compare_summary.csv", "Table"),
]

TEMP_GRAPH_DIRS = [
    REPO_ROOT / "outputs" / "surrogate_v35_rollout_live_prior420_rollout_heads",
    REPO_ROOT / "outputs" / "surrogate_v35_rollout_live_prior420_rollout_heads_nonlinear",
    REPO_ROOT / "outputs" / "surrogate_v35_rollout_live_prior420_rollout_heads_mixed",
]

TEMP_GRAPH_RELATIVE_PATHS = [
    Path("raw_v35/live_rollout_energy_metrics.png"),
    Path("raw_v35/live_rollout_temp_metrics.png"),
    Path("raw_v35/live_rollout_trajectory_grid.png"),
    Path("calibrated_v35/live_rollout_energy_metrics.png"),
    Path("calibrated_v35/live_rollout_temp_metrics.png"),
    Path("calibrated_v35/live_rollout_trajectory_grid.png"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _copy_assets() -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for src_rel, dst_rel, kind in COPY_SPECS:
        src = REPO_ROOT / src_rel
        dst = RESULTS_DIR / dst_rel
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(
            {
                "kind": kind,
                "source": src_rel.replace("\\", "/"),
                "destination": dst_rel.replace("\\", "/"),
            }
        )
    return copied


def _cleanup_temp_graphs() -> list[dict[str, str]]:
    deleted: list[dict[str, str]] = []
    for base_dir in TEMP_GRAPH_DIRS:
        if not base_dir.exists():
            continue
        for rel_path in TEMP_GRAPH_RELATIVE_PATHS:
            png_path = base_dir / rel_path
            rel = png_path.relative_to(REPO_ROOT).as_posix()
            if png_path.exists():
                png_path.unlink()
                deleted.append({"graph_path": rel, "status": "deleted"})
            else:
                deleted.append({"graph_path": rel, "status": "already_absent"})
    return deleted


def _build_code_entrypoints() -> list[dict[str, str]]:
    return [
        {
            "subsystem": "Data collection",
            "file": "data/collect_tsupply_data.py",
            "role": "Collects direct-TSup BOPTEST data on testcase bestest_air across 4 seasons and 4 action policies.",
            "status": "Current source of surrogate_v2 tsupply dataset.",
        },
        {
            "subsystem": "Surrogate training",
            "file": "surrogate/train_surrogate_v2.py",
            "role": "Builds BOPTESTDatasetV2 and trains RC Neural ODE surrogate with one-step and multi-step losses.",
            "status": "Core surrogate training entry point.",
        },
        {
            "subsystem": "Inverse calibration v3",
            "file": "surrogate/inverse_problem_boptest_v3.py",
            "role": "Artifact injection and calibration on real BOPTEST traces with preprocessing of noise, latency, and bias.",
            "status": "Improves RMSE strongly but does not recover physically correct C_zon.",
        },
        {
            "subsystem": "Inverse calibration v3.5",
            "file": "surrogate/inverse_problem_boptest_v35.py",
            "role": "Explicit structural C_zon, Stage A/B/C calibration, excitation-window selection, and rollout-aware head tuning.",
            "status": "Best physical C_zon recovery so far, rollout realism still unresolved.",
        },
        {
            "subsystem": "Thermostatic PPO",
            "file": "training/train_thermostatic.py",
            "role": "Comfort-first PPO baseline on direct TSup with 17-feature observation, forecasts, time cycles, and action history.",
            "status": "Current strongest validated comfort controller.",
        },
        {
            "subsystem": "Thermostatic eval",
            "file": "evaluation/eval_thermostatic.py",
            "role": "12-month BOPTEST validation for the thermostatic direct-TSup controller.",
            "status": "Produces yearly summary and scenario traces.",
        },
        {
            "subsystem": "HDRL",
            "file": "training/train_hdrl.py",
            "role": "Trains winter and summer PPO experts with a threshold-based gate and emergency logic.",
            "status": "Structured trade-off controller, not yet dominant annually.",
        },
        {
            "subsystem": "HDRL eval",
            "file": "evaluation/yearly_validation_hdrl.py",
            "role": "Yearly BOPTEST validation with seasonal gate, emergency heating, and direct TSup actuation.",
            "status": "Best RL violation rate, but weaker RMSE than thermostatic PPO.",
        },
        {
            "subsystem": "MORL entry point",
            "file": "main.py",
            "role": "Original MORL PPO orchestration for train, eval, multi-seed, and pareto sweep.",
            "status": "Architecturally present, but not yet fully revalidated after direct-TSup redesign.",
        },
        {
            "subsystem": "MORL yearly eval",
            "file": "evaluation/yearly_validation_morl.py",
            "role": "BOPTEST yearly validation script for MORL PPO models on testcase bestest_air.",
            "status": "Guarded by observation-dimension checks; direct-TSup retraining is still pending.",
        },
        {
            "subsystem": "Safety layer",
            "file": "layers/safety/action_filter.py",
            "role": "Surrogate-based action filter with fallback MPC for short-horizon safety screening.",
            "status": "Implemented, but current multi-seed metrics do not yet justify it as the main result.",
        },
    ]


def _load_controller_metrics() -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    rows = _read_csv(REPO_ROOT / "outputs" / "three_model_comparison" / "comparison_summary.csv")
    by_key = {row["controller_key"]: row for row in rows}
    return by_key, rows


def _load_pareto_summary() -> dict[str, dict[str, float]]:
    path = REPO_ROOT / "outputs" / "pareto" / "pareto_summary.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    data_rows = rows[3:]
    result: dict[str, dict[str, float]] = {}
    for row in data_rows:
        if not row or not row[0]:
            continue
        result[row[0]] = {
            "mean_comfort_mean": float(row[1]),
            "mean_comfort_std": float(row[2]),
            "mean_energy_mean": float(row[3]),
            "mean_energy_std": float(row[4]),
            "total_energy_kwh_mean": float(row[5]),
            "total_energy_kwh_std": float(row[6]),
            "m_s_mean": float(row[7]),
            "m_s_std": float(row[8]),
            "violation_pct_mean": float(row[9]),
            "violation_pct_std": float(row[10]),
        }
    return result


def _build_surrogate_progress() -> list[dict[str, str | float]]:
    live_rollout_rows = _read_csv(REPO_ROOT / "outputs" / "surrogate_rollout_live" / "horizon_metrics.csv")
    live_rollout_map = {int(float(row["horizon_h"])): row for row in live_rollout_rows}

    v3 = json.loads((REPO_ROOT / "outputs" / "surrogate_v2_inverse_boptest" / "calibration_summary_boptest_v3.json").read_text(encoding="utf-8"))
    v35_heads = json.loads((REPO_ROOT / "outputs" / "surrogate_v35_inverse_boptest_prior420_heads_only" / "calibration_summary_boptest_v35.json").read_text(encoding="utf-8"))
    v35_compare_rows = _read_csv(REPO_ROOT / "outputs" / "surrogate_v35_rollout_live" / "v35_compare_summary.csv")
    v35_compare = {row["variant"]: row for row in v35_compare_rows}
    multistart_rows = _read_csv(REPO_ROOT / "outputs" / "surrogate_v35_inverse_boptest_multistart" / "multistart_summary_v35.csv")
    best_multistart = min(multistart_rows, key=lambda row: float(row["czon_error_pct"]))

    return [
        {
            "stage": "Report Phase 1 surrogate",
            "source_type": "reported_in_reports",
            "temp_rmse_c": REPORT_MILESTONES["phase1_rmse_c"],
            "long_rollout_rmse_c": "",
            "r2_temp": REPORT_MILESTONES["phase1_r2"],
            "speedup_x": REPORT_MILESTONES["phase1_speedup_x"],
            "c_zon_error_pct": "",
            "note": "Physics-informed RC Neural ODE milestone from reports/Current_HVAC_PPO_MORL.pdf.",
        },
        {
            "stage": "Current live rollout on surrogate",
            "source_type": "measured_from_outputs",
            "temp_rmse_c": float(live_rollout_map[1]["temp_rmse_c"]),
            "long_rollout_rmse_c": float(live_rollout_map[24]["temp_rmse_c"]),
            "r2_temp": "",
            "speedup_x": "",
            "c_zon_error_pct": "",
            "note": f"24h rollout RMSE = {float(live_rollout_map[24]['temp_rmse_c']):.3f} C.",
        },
        {
            "stage": "Inverse problem v3",
            "source_type": "measured_from_outputs",
            "temp_rmse_c": float(v3["calibrated_rmse_c"]),
            "long_rollout_rmse_c": "",
            "r2_temp": "",
            "speedup_x": "",
            "c_zon_error_pct": float(v3["czon_error_pct"]),
            "note": f"Baseline RMSE {float(v3['baseline_rmse_c']):.4f} -> {float(v3['calibrated_rmse_c']):.4f} C.",
        },
        {
            "stage": "Inverse problem v3.5 multistart best prior",
            "source_type": "measured_from_outputs",
            "temp_rmse_c": float(best_multistart["calibrated_rmse_c"]),
            "long_rollout_rmse_c": "",
            "r2_temp": "",
            "speedup_x": "",
            "c_zon_error_pct": float(best_multistart["czon_error_pct"]),
            "note": f"Best prior tested = {float(best_multistart['c_zon_prior_tested_j_per_k']):.0f} J/K.",
        },
        {
            "stage": "Inverse problem v3.5 heads-only",
            "source_type": "measured_from_outputs",
            "temp_rmse_c": float(v35_heads["calibrated_rmse_c"]),
            "long_rollout_rmse_c": "",
            "r2_temp": "",
            "speedup_x": "",
            "c_zon_error_pct": float(v35_heads["czon_error_pct"]),
            "note": f"Stage C mode = {v35_heads['stage_c_mode']}.",
        },
        {
            "stage": "v3.5 rollout raw model",
            "source_type": "measured_from_outputs",
            "temp_rmse_c": float(v35_compare["raw_v35"]["one_step_rmse_c"]),
            "long_rollout_rmse_c": float(v35_compare["raw_v35"]["longest_horizon_rmse_c"]),
            "r2_temp": "",
            "speedup_x": "",
            "c_zon_error_pct": float(v35_compare["raw_v35"]["czon_error_pct"]),
            "note": f"24h rollout RMSE = {float(v35_compare['raw_v35']['longest_horizon_rmse_c']):.3f} C.",
        },
        {
            "stage": "v3.5 rollout calibrated heads-only",
            "source_type": "measured_from_outputs",
            "temp_rmse_c": float(v35_compare["calibrated_v35"]["one_step_rmse_c"]),
            "long_rollout_rmse_c": float(v35_compare["calibrated_v35"]["longest_horizon_rmse_c"]),
            "r2_temp": "",
            "speedup_x": "",
            "c_zon_error_pct": float(v35_compare["calibrated_v35"]["czon_error_pct"]),
            "note": f"24h rollout RMSE = {float(v35_compare['calibrated_v35']['longest_horizon_rmse_c']):.3f} C.",
        },
    ]


def _build_morl_progress() -> list[dict[str, str | float]]:
    pareto = _load_pareto_summary()
    multi_seed_summary = _read_csv(REPO_ROOT / "outputs" / "eval_multi_seed" / "summary.csv")[0]

    return [
        {
            "experiment": "Phase 0 reported baseline",
            "source_type": "reported_in_reports",
            "metric_primary": "energy_reduction_pct",
            "value_primary": REPORT_MILESTONES["phase0_energy_reduction_pct"],
            "metric_secondary": "best_ms_mean",
            "value_secondary": REPORT_MILESTONES["phase3_best_ms_mean"],
            "note": "Reported milestones from reports/Current_HVAC_PPO_MORL.pdf.",
        },
        {
            "experiment": "Pareto balanced",
            "source_type": "measured_from_outputs",
            "metric_primary": "m_s",
            "value_primary": float(pareto["balanced"]["m_s_mean"]),
            "metric_secondary": "total_energy_kwh",
            "value_secondary": float(pareto["balanced"]["total_energy_kwh_mean"]),
            "note": "Current raw Pareto point.",
        },
        {
            "experiment": "Pareto energy_only",
            "source_type": "measured_from_outputs",
            "metric_primary": "m_s",
            "value_primary": float(pareto["energy_only"]["m_s_mean"]),
            "metric_secondary": "total_energy_kwh",
            "value_secondary": float(pareto["energy_only"]["total_energy_kwh_mean"]),
            "note": "Energy-dominant point with comfort collapse.",
        },
        {
            "experiment": "Safe MORL multi-seed",
            "source_type": "measured_from_outputs",
            "metric_primary": "ppo_sf_ms_mean",
            "value_primary": float(multi_seed_summary["ppo_sf_ms_mean"]),
            "metric_secondary": "ppo_sf_accept_mean",
            "value_secondary": float(multi_seed_summary["ppo_sf_accept_mean"]),
            "note": "Safety layer summary from outputs/eval_multi_seed/summary.csv.",
        },
    ]


def _build_summary_markdown(
    controllers: dict[str, dict[str, str]],
    surrogate_progress: list[dict[str, str | float]],
    morl_progress: list[dict[str, str | float]],
) -> str:
    pi = controllers["standard_pi"]
    th = controllers["thermostatic"]
    hd = controllers["hdrl"]

    current_rollout = next(row for row in surrogate_progress if row["stage"] == "Current live rollout on surrogate")
    v3 = next(row for row in surrogate_progress if row["stage"] == "Inverse problem v3")
    v35_best = next(row for row in surrogate_progress if row["stage"] == "Inverse problem v3.5 heads-only")
    v35_raw = next(row for row in surrogate_progress if row["stage"] == "v3.5 rollout raw model")
    v35_cal = next(row for row in surrogate_progress if row["stage"] == "v3.5 rollout calibrated heads-only")

    lines = [
        "# Текущий срез проекта HVAC_DRL_MORL",
        "",
        "Пакет `results/` собирает текущие валидированные результаты по surrogate, thermostatic PPO, HDRL и MORL/safety layer.",
        "",
        "Важно:",
        "- `reported_in_reports` означает milestone-числа из `reports/Current_HVAC_PPO_MORL.pdf` и `reports/defensePart2.pdf`.",
        "- `measured_from_outputs` означает текущие фактические числа из `outputs/*.csv` и `outputs/*.json`.",
        "- Для презентации текущего состояния репозитория главным источником нужно считать именно `outputs/*`.",
        "",
        "## 1. Короткий итог по состоянию проекта",
        "",
        f"- Лучшая валидированная comfort-модель: Thermostatic PPO, mean RMSE22 = {float(th['rmse22_mean']):.3f} C, mean m_s = {float(th['ms_fixed_mean']):.3f}, total energy = {float(th['energy_total_kwh']):.1f} kWh.",
        f"- Энергетический reference: Standard PI, total energy = {float(pi['energy_total_kwh']):.1f} kWh, но RMSE22 = {float(pi['rmse22_mean']):.3f} C.",
        f"- Лучшая RL-модель по violation rate: HDRL, violation [21,25]C = {float(hd['viol_21_25_mean']):.2f}%, но mean RMSE22 = {float(hd['rmse22_mean']):.3f} C.",
        "- MORL и safety layer уже реализованы, но ещё не являются strongest validated result после direct-TSup redesign.",
        "",
        "## 2. Суррогатный digital twin",
        "",
        "### 2.1 Откуда взялись данные",
        "- Сбор идёт через `data/collect_tsupply_data.py` на BOPTEST testcase `bestest_air`.",
        "- Для dataset generation используется прямое управление `fcu_oveTSup_u` и `fcu_oveFan_u`.",
        "- Внутренний PI-контур BOPTEST нейтрализуется фиксированием `Tset,cool = 40C` и `Tset,heat = 15C`.",
        "- Датасет строится как 4 сезона x 4 политики x 3200 шагов = 51 200 переходов.",
        "- Основные поля: `t_zone`, `t_amb`, `hour`, `day`, `a0_raw`, `a1_raw`, `t_zone_next`, `delta_t`, `p_total`.",
        "",
        "### 2.2 Обучение суррогата",
        "- `surrogate/train_surrogate_v2.py` использует `BOPTESTDatasetV2` и multi-step loss по горизонтам `[2,4]`.",
        f"- Зафиксированный report milestone: RMSE = {REPORT_MILESTONES['phase1_rmse_c']:.3f} C, R2 = {REPORT_MILESTONES['phase1_r2']:.3f}, speedup = {REPORT_MILESTONES['phase1_speedup_x']:.0f}x.",
        f"- Текущий live-rollout baseline: 1h RMSE = {float(current_rollout['temp_rmse_c']):.3f} C, 24h rollout RMSE = 0.754 C.",
        "",
        "### 2.3 Что означают Stage A / B / C",
        "- Stage A: preprocessing observed trace, компенсация noise/latency/bias/scale артефактов.",
        "- Stage B: физическая идентификация `C_zon` на возбуждённых окнах с высоким `|dT/dt|` при почти замороженном backbone.",
        "- Stage C: мягкая калибровка heads или части модели, чтобы улучшить fit без разрушения найденной физики.",
        "",
        "### 2.4 Текущий статус inverse calibration",
        f"- v3: calibrated RMSE = {float(v3['temp_rmse_c']):.4f} C, но C_zon error = {float(v3['c_zon_error_pct']):.2f}%.",
        f"- v3.5 heads-only: calibrated RMSE = {float(v35_best['temp_rmse_c']):.4f} C, C_zon error = {float(v35_best['c_zon_error_pct']):.2f}%.",
        f"- raw v3.5 rollout: 24h RMSE = {float(v35_raw['long_rollout_rmse_c']):.3f} C.",
        f"- calibrated v3.5 rollout: 24h RMSE = {float(v35_cal['long_rollout_rmse_c']):.3f} C.",
        "- Вывод: физическая идентификация уже стала качественной, но free-run rollout realism для калиброванного twin ещё не улучшен.",
        "",
        "## 3. Thermostatic PPO",
        "- Обучается в `training/train_thermostatic.py`.",
        "- Работает на direct-TSup surrogate с observation budget 17 признаков: физическое состояние, cyclic time, forecast, previous action, delta-T history.",
        "- Reward ориентирован на tracking 22C, с повышенным штрафом за winter underheating и только слабой energy regularization около target.",
        "- Валидация идёт в `evaluation/eval_thermostatic.py` по 12 месячным сценариям BOPTEST `bestest_air`.",
        f"- Текущий итог: RMSE22 = {float(th['rmse22_mean']):.3f} C, MAE22 = {float(th['mae22_mean']):.3f} C, within ±1C = {float(th['within_1c_mean']):.1f}%, total energy = {float(th['energy_total_kwh']):.1f} kWh.",
        "",
        "## 4. HDRL",
        "- Обучается в `training/train_hdrl.py` как два PPO-эксперта: winter и summer.",
        "- Evaluation в `evaluation/yearly_validation_hdrl.py` использует seasonal gate и emergency heating rule.",
        f"- Текущий итог: RMSE22 = {float(hd['rmse22_mean']):.3f} C, violation = {float(hd['viol_21_25_mean']):.2f}%, m_s = {float(hd['ms_fixed_mean']):.3f}, total energy = {float(hd['energy_total_kwh']):.1f} kWh.",
        "- Интерпретация: HDRL уже выглядит как structured trade-off controller, но пока не доминирует над thermostatic PPO на annual benchmark.",
        "",
        "## 5. MORL / safety layer",
        "- Entry point MORL: `main.py`.",
        "- Yearly BOPTEST validation для MORL: `evaluation/yearly_validation_morl.py`.",
        "- Surrogate-based action filtering: `layers/safety/action_filter.py`.",
        f"- Report milestone Phase 3: best m_s = {REPORT_MILESTONES['phase3_best_ms_mean']:.3f} ± {REPORT_MILESTONES['phase3_best_ms_std']:.3f}.",
        f"- Current raw Pareto balanced point: m_s = {float(morl_progress[1]['value_primary']):.3f}, total energy = {float(morl_progress[1]['value_secondary']):.1f} kWh.",
        f"- Current safe-MORL multi-seed snapshot: ppo_sf_ms_mean = {float(morl_progress[3]['value_primary']):.3f}, acceptance mean = {float(morl_progress[3]['value_secondary']):.1f}%.",
        "- Честный вывод: MORL нужно показывать как готовую архитектурную следующую фазу, а не как текущий strongest validated result.",
        "",
        "## 6. Что находится в results/",
        "- `figures/paper`: canonical figure из основных отчётов.",
        "- `figures/controllers`: сравнение Standard PI / Thermostatic PPO / HDRL.",
        "- `figures/surrogate`: surrogate rollout, parity и comfort-trace графики.",
        "- `figures/calibration`: raw vs calibrated v3.5 rollout-сравнение.",
        "- `tables/raw`: исходные CSV/JSON/TXT, на которые опирается этот snapshot.",
        "- `tables/controller_highlights.csv`, `tables/surrogate_progress.csv`, `tables/morl_progress.csv`, `tables/code_entrypoints.csv`: компактные презентационные таблицы.",
        "- `manifests`: список скопированных файлов и список удалённых временных графиков.",
        "",
        "## 7. Что было очищено",
        "- Удалены только временные PNG из промежуточных v3.5 rollout-экспериментов (`prior420_rollout_heads`, `prior420_rollout_heads_nonlinear`, `prior420_rollout_heads_mixed`).",
        "- CSV, JSON, модели и остальные исходные результаты сохранены.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    copied_assets = _copy_assets()
    deleted_graphs = _cleanup_temp_graphs()

    controllers_map, controller_rows = _load_controller_metrics()
    surrogate_progress = _build_surrogate_progress()
    morl_progress = _build_morl_progress()
    code_rows = _build_code_entrypoints()

    _write_csv(
        RESULTS_DIR / "tables" / "controller_highlights.csv",
        controller_rows,
        controller_rows[0].keys(),
    )
    _write_csv(
        RESULTS_DIR / "tables" / "surrogate_progress.csv",
        surrogate_progress,
        surrogate_progress[0].keys(),
    )
    _write_csv(
        RESULTS_DIR / "tables" / "morl_progress.csv",
        morl_progress,
        morl_progress[0].keys(),
    )
    _write_csv(
        RESULTS_DIR / "tables" / "code_entrypoints.csv",
        code_rows,
        code_rows[0].keys(),
    )
    _write_csv(
        RESULTS_DIR / "manifests" / "copied_assets.csv",
        copied_assets,
        ["kind", "source", "destination"],
    )
    _write_csv(
        RESULTS_DIR / "manifests" / "deleted_temp_graphs.csv",
        deleted_graphs,
        ["graph_path", "status"],
    )

    key_metrics = {
        "reported_milestones": REPORT_MILESTONES,
        "current_validated_controllers": controllers_map,
        "surrogate_progress": surrogate_progress,
        "morl_progress": morl_progress,
        "deleted_temp_graphs_total_targets": len(deleted_graphs),
        "deleted_temp_graphs_removed_now": sum(1 for row in deleted_graphs if row["status"] == "deleted"),
    }
    (RESULTS_DIR / "manifests" / "key_metrics.json").write_text(
        json.dumps(key_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / "README.md").write_text(
        _build_summary_markdown(controllers_map, surrogate_progress, morl_progress),
        encoding="utf-8",
    )

    print("RESULTS BUNDLE CREATED")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Copied assets: {len(copied_assets)}")
    print(f"Deleted temp graphs: {len(deleted_graphs)}")
    print(f"Summary: {RESULTS_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
