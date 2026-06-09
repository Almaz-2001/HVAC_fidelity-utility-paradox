from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate.direct_tsup_adapter import load_direct_tsup_adapter


REPORTS_DIR = ROOT / "reports"
PREPARED_15MIN_CSV = ROOT / "data" / "block_1_2_surrogate_rmse" / "boptest_block12_15min_prepared.csv"
V3_MODEL_PATH = ROOT / "outputs" / "surrogate_v2" / "rc_node_v3_tsupply.pt"
V35_SUMMARY_JSON = (
    ROOT
    / "outputs"
    / "surrogate_v35_inverse_boptest_15min_power_head_only"
    / "calibration_summary_boptest_v35.json"
)
V35_EXCITATION_JSON = (
    ROOT
    / "outputs"
    / "surrogate_v35_inverse_boptest_15min_power_head_only"
    / "excitation_summary.json"
)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path.relative_to(ROOT)}")


def build_training_hyperparams_table() -> None:
    source = "surrogate/train_surrogate_backbone.py"
    rows = [
        {
            "param": "active_v3_training_entrypoint",
            "value": source,
            "source_file": source,
            "justification": "Canonical comfort-oriented direct-TSup v3 backbone training script currently present in active contour.",
        },
        {
            "param": "legacy_train_surrogate_py",
            "value": "not present in active contour",
            "source_file": "surrogate/train_surrogate.py",
            "justification": "The former generic entrypoint is not part of the current active contour; v3 training is reported through train_surrogate_backbone.py.",
        },
        {
            "param": "epochs",
            "value": "500",
            "source_file": source,
            "justification": "Default CLI value; upper budget constrained by early stopping.",
        },
        {
            "param": "batch_size",
            "value": "256",
            "source_file": source,
            "justification": "Default DataLoader batch size used for stable mini-batch estimates on the BOPTEST corpus.",
        },
        {
            "param": "learning_rate",
            "value": "1e-3",
            "source_file": source,
            "justification": "Default AdamW learning rate for the v3 backbone.",
        },
        {
            "param": "hidden_dim",
            "value": "64",
            "source_file": source,
            "justification": "Default hidden width used by RCNeuralODEv2 heat and power heads.",
        },
        {
            "param": "optimizer",
            "value": "AdamW",
            "source_file": source,
            "justification": "Weight-decayed Adam optimizer stabilizes the black-box residual heads without changing the physical inputs.",
        },
        {
            "param": "weight_decay",
            "value": "1e-4",
            "source_file": source,
            "justification": "Explicit optimizer regularization in the canonical v3 training loop.",
        },
        {
            "param": "scheduler",
            "value": "CosineAnnealingLR(T_max=epochs)",
            "source_file": source,
            "justification": "Smooth learning-rate decay over the configured epoch budget.",
        },
        {
            "param": "early_stopping",
            "value": "patience=30; improvement_threshold=1e-6",
            "source_file": source,
            "justification": "Stops training when the selected validation metric no longer improves.",
        },
        {
            "param": "checkpoint_metric",
            "value": "val_loss default; rollout_rmse optional",
            "source_file": source,
            "justification": "Default keeps one-step supervised fidelity; rollout_rmse was added for rollout-aware selection when explicitly requested.",
        },
        {
            "param": "gradient_clip_norm",
            "value": "1.0",
            "source_file": source,
            "justification": "Limits unstable updates from multi-step rollout loss.",
        },
        {
            "param": "multi_horizons",
            "value": "[2, 4]",
            "source_file": source,
            "justification": "Default short rollout horizons for one-step-plus-local-rollout training.",
        },
        {
            "param": "loss_lambda_temp",
            "value": "1.0",
            "source_file": source,
            "justification": "Primary temperature fidelity term.",
        },
        {
            "param": "loss_lambda_power",
            "value": "0.1",
            "source_file": source,
            "justification": "Power-head term is normalized by P_MAX and weighted below the temperature term.",
        },
        {
            "param": "loss_lambda_multi",
            "value": "0.5",
            "source_file": source,
            "justification": "Balances local rollout consistency against one-step fit.",
        },
        {
            "param": "loss_lambda_physics",
            "value": "0.05",
            "source_file": source,
            "justification": "Penalizes implausibly large one-step temperature jumps.",
        },
    ]
    _write_csv(REPORTS_DIR / "hou_evins_training_hyperparams_table.csv", rows)


def build_scaling_table() -> None:
    rows = [
        {
            "variable": "surrogate_t_zone",
            "context": "v3/v3.5 surrogate input",
            "scaling_method": "affine min-max to [-1, 1]",
            "parameters": "min=15 C; max=35 C",
            "source_file": "surrogate/rc_node_v2.py; surrogate/rc_node_v35.py",
            "justification": "Keeps the learned thermal state within the training support used by the direct-TSup surrogate.",
        },
        {
            "variable": "surrogate_t_amb",
            "context": "v3/v3.5 surrogate input",
            "scaling_method": "affine min-max to [-1, 1]",
            "parameters": "min=-10 C; max=40 C",
            "source_file": "surrogate/rc_node_v2.py; surrogate/rc_node_v35.py",
            "justification": "Normalizes outdoor-air temperature for the heat and power heads.",
        },
        {
            "variable": "hour",
            "context": "v3/v3.5 surrogate input and TSup controller observation",
            "scaling_method": "cyclic sine/cosine",
            "parameters": "sin(2*pi*hour/24); cos(2*pi*hour/24)",
            "source_file": "surrogate/rc_node_v2.py; envs/tsup_features.py",
            "justification": "Avoids discontinuity between hour 23 and hour 0.",
        },
        {
            "variable": "day",
            "context": "v3/v3.5 surrogate input and TSup controller observation",
            "scaling_method": "cyclic sine/cosine",
            "parameters": "sin(2*pi*day/365); cos(2*pi*day/365)",
            "source_file": "surrogate/rc_node_v2.py; envs/tsup_features.py",
            "justification": "Represents seasonal phase without an artificial year-end jump.",
        },
        {
            "variable": "a0_raw",
            "context": "direct supply-temperature action",
            "scaling_method": "native action domain",
            "parameters": "[-1, 1] maps to 18..35 C supply temperature",
            "source_file": "envs/tsup_features.py",
            "justification": "Preserves the PPO action coordinate while keeping physical supply-temperature bounds explicit.",
        },
        {
            "variable": "a1_raw",
            "context": "direct fan action",
            "scaling_method": "native action domain",
            "parameters": "[-1, 1] maps to 0..1 fan command",
            "source_file": "envs/tsup_features.py",
            "justification": "Keeps fan modulation aligned with the controller action range.",
        },
        {
            "variable": "p_total",
            "context": "TSup controller observation, raw mode",
            "scaling_method": "affine min-max to [-1, 1]",
            "parameters": "min=0 W; max=5500 W",
            "source_file": "envs/tsup_features.py",
            "justification": "Raw power normalization for ablation baselines.",
        },
        {
            "variable": "p_total",
            "context": "canonical TSup controller observation",
            "scaling_method": "clipped log transform to [-1, 1]",
            "parameters": "clip at 2500 W; 2*log1p(p)/log1p(2500)-1",
            "source_file": "envs/tsup_features.py",
            "justification": "Compresses high-power tails and improved transfer in the canonical hybrid branch.",
        },
        {
            "variable": "delta_t",
            "context": "TSup controller observation, raw mode",
            "scaling_method": "linear clipping",
            "parameters": "clip(delta_t/5 C, -1, 1)",
            "source_file": "envs/tsup_features.py",
            "justification": "Baseline delta-temperature history encoding.",
        },
        {
            "variable": "delta_t",
            "context": "MORL 17D TSup-style observation",
            "scaling_method": "causal smooth tanh",
            "parameters": "clip to +/-1.5 C; tanh(delta/1.25)",
            "source_file": "envs/tsup_features.py",
            "justification": "Reduces action discontinuities from noisy short-step temperature changes.",
        },
        {
            "variable": "controller_t_zone",
            "context": "TSup controller observation, raw mode",
            "scaling_method": "affine min-max to [-1, 1]",
            "parameters": "min=5 C; max=40 C",
            "source_file": "envs/tsup_features.py",
            "justification": "Wide physical range for controller-side temperature observation.",
        },
        {
            "variable": "controller_t_zone",
            "context": "TSup controller observation, comfort-centered ablation",
            "scaling_method": "comfort-centered clipping",
            "parameters": "clip((T_zone-22.5 C)/3 C, -1, 1)",
            "source_file": "envs/tsup_features.py",
            "justification": "Ablation tested a comfort-local coordinate, but canonical thermostatic hybrid uses raw mode.",
        },
        {
            "variable": "power_head_output",
            "context": "v3 surrogate output",
            "scaling_method": "Softplus multiplied by P_MAX",
            "parameters": "P_MAX=5500 W",
            "source_file": "surrogate/rc_node_v2.py",
            "justification": "Guarantees non-negative predicted HVAC power.",
        },
        {
            "variable": "C_zon",
            "context": "v3.5 physics backbone",
            "scaling_method": "positive reparameterization",
            "parameters": "softplus(log_c_zon)+c_min; c_min=5e4 J/K",
            "source_file": "surrogate/rc_node_v35.py",
            "justification": "Constrains the identified thermal capacitance to physically meaningful positive values.",
        },
    ]
    _write_csv(REPORTS_DIR / "hou_evins_scaling_table.csv", rows)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _normalized_mutual_info(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    total = float(hist.sum())
    if total <= 0.0:
        return float("nan")

    pxy = hist / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0.0
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))
    hx = -float(np.sum(px[px > 0.0] * np.log(px[px > 0.0])))
    hy = -float(np.sum(py[py > 0.0] * np.log(py[py > 0.0])))
    denom = math.sqrt(max(hx * hy, 1e-12))
    return float(mi / denom)


def _interpret_independence(abs_pearson: float, nmi: float) -> str:
    if abs_pearson >= 0.85 or nmi >= 0.70:
        return "high_dependency_review_required"
    if abs_pearson >= 0.60 or nmi >= 0.45:
        return "moderate_dependency_expected_from_physics_or_time"
    return "low_to_moderate_dependency"


def build_input_independence_table() -> None:
    df = pd.read_csv(_require(PREPARED_15MIN_CSV))
    features = pd.DataFrame(
        {
            "t_zone": df["t_zone"].astype(float),
            "t_amb": df["t_amb"].astype(float),
            "hour_sin": np.sin(2.0 * np.pi * df["hour"].astype(float) / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * df["hour"].astype(float) / 24.0),
            "day_sin": np.sin(2.0 * np.pi * df["day"].astype(float) / 365.0),
            "day_cos": np.cos(2.0 * np.pi * df["day"].astype(float) / 365.0),
            "prev_action_a0": df["a0_raw"].astype(float),
            "prev_action_a1": df["a1_raw"].astype(float),
            "power_history_proxy": df["p_total"].astype(float),
            "delta_t_history": df["delta_t"].astype(float),
        }
    )

    rows: list[dict] = []
    names = list(features.columns)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            x = features[left].to_numpy(dtype=float)
            y = features[right].to_numpy(dtype=float)
            pearson = _safe_corr(x, y)
            nmi = _normalized_mutual_info(x, y)
            rows.append(
                {
                    "feature_i": left,
                    "feature_j": right,
                    "pearson_r": pearson,
                    "abs_pearson_r": abs(pearson) if np.isfinite(pearson) else np.nan,
                    "normalized_mutual_info": nmi,
                    "n_samples": len(features),
                    "source_file": str(PREPARED_15MIN_CSV.relative_to(ROOT)).replace("\\", "/"),
                    "interpretation": _interpret_independence(abs(pearson), nmi)
                    if np.isfinite(pearson) and np.isfinite(nmi)
                    else "undefined_constant_or_invalid_feature",
                }
            )
    _write_csv(REPORTS_DIR / "hou_evins_input_independence_table.csv", rows)


def _valid_start_indices(df: pd.DataFrame, horizon_steps: int) -> np.ndarray:
    episode = df["episode_id"].astype(str).to_numpy()
    candidates = []
    n = len(df)
    for idx in range(0, n - horizon_steps):
        if episode[idx] == episode[idx + horizon_steps]:
            candidates.append(idx)
    return np.asarray(candidates, dtype=np.int64)


def _tensor(series: pd.Series, indices: np.ndarray) -> torch.Tensor:
    return torch.tensor(series.to_numpy(dtype=np.float32)[indices], dtype=torch.float32)


def _evaluate_model_horizon(adapter, df: pd.DataFrame, horizon_steps: int) -> dict:
    starts = _valid_start_indices(df, horizon_steps)
    if len(starts) == 0:
        raise ValueError(f"No valid starts for horizon_steps={horizon_steps}")

    with torch.no_grad():
        t_curr = _tensor(df["t_zone"], starts)
        power_pred_steps = []
        power_true_steps = []

        for offset in range(horizon_steps):
            idx = starts + offset
            t_amb = _tensor(df["t_amb"], idx)
            hour = _tensor(df["hour"], idx)
            day = _tensor(df["day"], idx)
            a0 = _tensor(df["a0_raw"], idx)
            a1 = _tensor(df["a1_raw"], idx)
            t_curr, p_pred = adapter(t_curr, t_amb, hour, day, a0, a1)
            power_pred_steps.append(p_pred.detach().cpu().numpy())
            power_true_steps.append(df["p_total"].to_numpy(dtype=np.float32)[idx])

    t_true = df["t_zone"].to_numpy(dtype=np.float32)[starts + horizon_steps]
    t_pred = t_curr.detach().cpu().numpy()
    p_pred_all = np.concatenate(power_pred_steps)
    p_true_all = np.concatenate(power_true_steps)

    t_error = t_pred - t_true
    p_error = p_pred_all - p_true_all
    ss_res = float(np.sum(t_error**2))
    ss_tot = float(np.sum((t_true - float(np.mean(t_true))) ** 2))
    r2_t = 1.0 - ss_res / (ss_tot + 1e-12)
    return {
        "n_windows": int(len(starts)),
        "n_power_steps": int(len(p_pred_all)),
        "RMSE_T": float(np.sqrt(np.mean(t_error**2))),
        "MAE_T": float(np.mean(np.abs(t_error))),
        "R2_T": float(r2_t),
        "RMSE_P": float(np.sqrt(np.mean(p_error**2))),
        "MAE_P": float(np.mean(np.abs(p_error))),
    }


def build_predictive_validity_table() -> None:
    _require(V3_MODEL_PATH)
    _require(V35_SUMMARY_JSON)
    df = pd.read_csv(_require(PREPARED_15MIN_CSV))

    model_specs = [
        {
            "model": "v3",
            "kind": "legacy_v3",
            "lambda_temp_disagree": "",
            "lambda_power_disagree": "",
            "source_note": "direct-TSup v3 primary dynamics",
        },
        {
            "model": "v3.5_calibrated",
            "kind": "v35_calibrated",
            "lambda_temp_disagree": "",
            "lambda_power_disagree": "",
            "source_note": "calibrated v3.5 physical backbone and power head",
        },
        {
            "model": "hybrid_l010",
            "kind": "hybrid_v3_v35",
            "lambda_temp_disagree": "0.10",
            "lambda_power_disagree": "5e-5",
            "source_note": "v3 primary rollout dynamics; calibrated v3.5 disagreement regularizer used during control training",
        },
    ]
    horizons = [
        ("1h", 4),
        ("4h", 16),
        ("8h", 32),
        ("24h", 96),
    ]

    rows: list[dict] = []
    for spec in model_specs:
        adapter = load_direct_tsup_adapter(
            spec["kind"],
            legacy_model_path=V3_MODEL_PATH,
            summary_json=V35_SUMMARY_JSON,
            runtime_step_sec=900,
            legacy_step_sec=3600,
            device="cpu",
        )
        adapter.eval()
        for horizon_label, steps in horizons:
            metrics = _evaluate_model_horizon(adapter, df, steps)
            rows.append(
                {
                    "model": spec["model"],
                    "horizon": horizon_label,
                    "horizon_steps_15min": steps,
                    "RMSE_T": metrics["RMSE_T"],
                    "MAE_T": metrics["MAE_T"],
                    "R2_T": metrics["R2_T"],
                    "RMSE_P": metrics["RMSE_P"],
                    "MAE_P": metrics["MAE_P"],
                    "n_windows": metrics["n_windows"],
                    "n_power_steps": metrics["n_power_steps"],
                    "lambda_temp_disagree": spec["lambda_temp_disagree"],
                    "lambda_power_disagree": spec["lambda_power_disagree"],
                    "source_note": spec["source_note"],
                }
            )
    _write_csv(REPORTS_DIR / "hou_evins_predictive_validity_table.csv", rows)


def build_q1_gap_closure_note() -> None:
    excitation = json.loads(_require(V35_EXCITATION_JSON).read_text(encoding="utf-8"))
    summary = json.loads(_require(V35_SUMMARY_JSON).read_text(encoding="utf-8"))
    text = f"""# Hou and Evins Q1 Gap Closure Notes

Date: 2026-05-11

## Excitation-Window Rationale

The v3.5 inverse task is not trained on uniformly sampled state-action space. It uses scenario-stratified BOPTEST trajectories and then selects the high-information part of the trajectory for Stage B/C calibration. In the canonical 15-minute branch, the excitation selector used mode `{excitation.get("excitation_mode")}`, quantile `{excitation.get("excitation_quantile")}`, threshold `{excitation.get("excitation_threshold")}`, and retained `{excitation.get("rows_excitation")}` excitation rows from `{excitation.get("rows_train_all")}` training rows.

This is deliberate: `C_zon` is identifiable only when the zone temperature is moving enough for the heat-balance residual to carry signal. Low-dynamics comfort-holding periods are useful for controller evaluation, but they are weak evidence for inverse thermal-capacitance estimation. The selected rows had mean excitation score `{excitation.get("score_mean_excitation")}` versus `{excitation.get("score_mean_train")}` over the full training split.

## Scenario Stratification Instead Of LHS

Hou and Evins recommend space-filling designs such as Latin hypercube sampling when the surrogate is meant to approximate a broad static input-output map. This project uses a control-oriented surrogate: the relevant distribution is the closed-loop distribution induced by feasible HVAC policies, not a uniform distribution over all mathematical inputs. Therefore, the current sampling is scenario-stratified by BOPTEST operating windows and controller families. This preserves physically reachable state-action transitions and makes the supervised data closer to the RL deployment distribution.

## Input Independence Check

The independence check is reported in `reports/hou_evins_input_independence_table.csv`. The only pair flagged as `high_dependency_review_required` is `day_sin` versus `day_cos`. This is expected for a restricted seasonal subset: sine and cosine are both deterministic encodings of the same calendar variable, and a narrow day-of-year range can make them strongly correlated. They are retained because the pair is needed to avoid a seasonal discontinuity in the full annual representation.

## Replicative Versus Predictive Validity Boundary

Replicative validity is reported through one-step calibration metrics in `calibration_summary_boptest_v35.json`: calibrated temperature RMSE `{summary.get("calibrated_rmse_c")}` C and calibrated power MAE `{summary.get("calibrated_power_mae_w")}` W.

Predictive validity is now reported separately in `reports/hou_evins_predictive_validity_table.csv` at 1h, 4h, 8h, and 24h horizons for `v3`, `v3.5_calibrated`, and `hybrid_l010`.
"""
    path = REPORTS_DIR / "hou_evins_q1_gap_closure.md"
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    build_training_hyperparams_table()
    build_scaling_table()
    build_input_independence_table()
    build_predictive_validity_table()
    build_q1_gap_closure_note()


if __name__ == "__main__":
    main()
