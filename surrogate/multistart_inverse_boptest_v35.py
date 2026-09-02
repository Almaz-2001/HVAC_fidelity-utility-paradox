from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from surrogate.inverse_problem_boptest_v3 import ArtifactSpec
from surrogate.inverse_problem_boptest_v35 import calibrate_boptest_v35


def _parse_priors(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("At least one C_zon prior is required")
    return values


def _prior_tag(value: float) -> str:
    return f"{value:.0f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-start staged inverse calibration for Surrogate v3.5 over multiple C_zon priors."
    )
    parser.add_argument("--data", default="data/surrogate_v2/boptest_v2_tsupply.csv")
    parser.add_argument("--model", default="outputs/surrogate_v2/rc_node_v3_tsupply.pt")
    parser.add_argument("--base-output-dir", default="outputs/surrogate_v35_inverse_boptest_multistart")
    parser.add_argument("--policy", default="mixed")
    parser.add_argument("--season", default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--no-artifact-injection", action="store_true")
    parser.add_argument("--temp-bias-c", type=float, default=0.5)
    parser.add_argument("--temp-noise-std", type=float, default=0.08)
    parser.add_argument("--temp-latency-steps", type=int, default=2)
    parser.add_argument("--power-scale", type=float, default=1.04)
    parser.add_argument("--power-bias-w", type=float, default=35.0)
    parser.add_argument("--power-noise-rel", type=float, default=0.015)
    parser.add_argument("--c-zon-true", type=float, default=4.2e5)
    parser.add_argument("--surrogate-czon-ref", type=float, default=5.3e5)
    parser.add_argument("--max-latency-search", type=int, default=6)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--target-mode", default="clean", choices=["clean", "preprocessed"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--stage-b-epochs", type=int, default=120)
    parser.add_argument("--stage-b-patience", type=int, default=20)
    parser.add_argument("--stage-c-epochs", type=int, default=8)
    parser.add_argument("--stage-c-patience", type=int, default=3)
    parser.add_argument("--czon-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--calib-lr", type=float, default=1e-3)
    parser.add_argument("--lambda-c-prior-b", type=float, default=0.35)
    parser.add_argument("--lambda-c-prior-c", type=float, default=0.20)
    parser.add_argument("--lambda-power", type=float, default=0.02)
    parser.add_argument("--lambda-q-reg", type=float, default=0.0005)
    parser.add_argument("--lambda-temp-reg", type=float, default=0.02)
    parser.add_argument("--lambda-rollout", type=float, default=0.0)
    parser.add_argument("--rollout-teacher-forced-epochs", type=int, default=10)
    parser.add_argument("--rollout-free-run-final-ratio", type=float, default=0.5)
    parser.add_argument("--excitation-quantile", type=float, default=0.95)
    parser.add_argument("--excitation-mix-ratio", type=float, default=0.0)
    parser.add_argument("--excitation-mode", choices=["hybrid", "dt_only"], default="dt_only")
    parser.add_argument("--c-zon-priors", default="420000,480000,530000")
    parser.add_argument("--c-zon-min", type=float, default=5.0e4)
    parser.add_argument("--q-scale", type=float, default=3000.0)
    parser.add_argument("--stage-c-mode", choices=["joint", "heads_only", "rollout_heads_only"], default="joint")
    parser.add_argument("--rollout-horizons", default="4,8")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    priors = _parse_priors(args.c_zon_priors)
    out_root = Path(args.base_output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    spec = ArtifactSpec(
        temp_bias_c=args.temp_bias_c,
        temp_noise_std=args.temp_noise_std,
        temp_latency_steps=args.temp_latency_steps,
        power_scale=args.power_scale,
        power_bias_w=args.power_bias_w,
        power_noise_rel=args.power_noise_rel,
        c_zon_true_j_per_k=args.c_zon_true,
        surrogate_czon_ref_j_per_k=args.surrogate_czon_ref,
    )

    summary_rows = []
    for idx, prior in enumerate(priors):
        run_dir = out_root / f"prior_{_prior_tag(prior)}"
        print("=" * 72)
        print(f"[MULTISTART_V35] prior={prior:.3e} -> {run_dir}")
        print("=" * 72)

        summary_path = calibrate_boptest_v35(
            data_path=args.data,
            model_path=args.model,
            output_dir=str(run_dir),
            policy=args.policy,
            season=args.season,
            limit_rows=args.limit_rows,
            inject_artifacts=not args.no_artifact_injection,
            artifact_spec=spec,
            max_latency_search=args.max_latency_search,
            smooth_window=args.smooth_window,
            target_mode=args.target_mode,
            stage_b_epochs=args.stage_b_epochs,
            stage_b_patience=args.stage_b_patience,
            stage_c_epochs=args.stage_c_epochs,
            stage_c_patience=args.stage_c_patience,
            batch_size=args.batch_size,
            val_split=args.val_split,
            czon_lr=args.czon_lr,
            backbone_lr=args.backbone_lr,
            calib_lr=args.calib_lr,
            lambda_c_prior_b=args.lambda_c_prior_b,
            lambda_c_prior_c=args.lambda_c_prior_c,
            lambda_power=args.lambda_power,
            lambda_q_reg=args.lambda_q_reg,
            lambda_temp_reg=args.lambda_temp_reg,
            lambda_rollout=args.lambda_rollout,
            rollout_teacher_forced_epochs=args.rollout_teacher_forced_epochs,
            rollout_free_run_final_ratio=args.rollout_free_run_final_ratio,
            excitation_quantile=args.excitation_quantile,
            excitation_mix_ratio=args.excitation_mix_ratio,
            excitation_mode=args.excitation_mode,
            c_zon_prior=prior,
            c_zon_min=args.c_zon_min,
            q_scale=args.q_scale,
            stage_c_mode=args.stage_c_mode,
            rollout_horizons=args.rollout_horizons,
            seed=args.seed + idx,
        )

        row = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        row["run_dir"] = str(run_dir)
        row["c_zon_prior_tested_j_per_k"] = prior
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    sort_cols = ["czon_error_pct", "calibrated_rmse_c"] if "czon_error_pct" in df.columns else ["calibrated_rmse_c"]
    df = df.sort_values(sort_cols, na_position="last")

    csv_path = out_root / "multistart_summary_v35.csv"
    json_path = out_root / "multistart_summary_v35.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    print("=" * 72)
    print("[MULTISTART_V35] COMPLETE")
    print(f"CSV summary:  {csv_path}")
    print(f"JSON summary: {json_path}")
    if len(df) > 0:
        best = df.iloc[0]
        print(
            "[MULTISTART_V35] Best candidate: "
            f"prior={best['c_zon_prior_tested_j_per_k']:.3e}, "
            f"czon_error={best.get('czon_error_pct', float('nan'))}, "
            f"rmse={best['calibrated_rmse_c']:.4f}"
        )


if __name__ == "__main__":
    main()
