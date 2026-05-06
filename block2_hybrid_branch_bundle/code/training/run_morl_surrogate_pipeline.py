from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.loader import load_yaml


@dataclass
class PipelinePaths:
    seed_root: Path
    pretrain_out: Path
    pretrain_model: Path
    eram_out: Path
    eram_model: Path
    eram_weights: Path
    finetune_out: Path
    finetune_model: Path
    yearly_eval_out: Path


def project_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return ROOT / path


def _is_runnable_python(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if os.name != "nt" and path.suffix.lower() == ".exe":
        return False
    return os.name == "nt" or os.access(path, os.X_OK)


def resolve_python_executable() -> str:
    if os.name == "nt":
        candidates = [
            ROOT / ".venv" / "Scripts" / "python.exe",
            ROOT / ".venv" / "bin" / "python",
            Path(sys.executable),
        ]
    else:
        candidates = [
            ROOT / ".venv" / "bin" / "python",
            ROOT / ".venv" / "Scripts" / "python.exe",
            Path(sys.executable),
        ]
    for path in candidates:
        if _is_runnable_python(path):
            return str(path)
    return sys.executable


def build_paths(config_dir: Path, artifact_root: Path, seed: int) -> tuple[dict, PipelinePaths]:
    pipeline_cfg = load_yaml(config_dir / "pipeline.yaml")
    subdirs = pipeline_cfg.get("subdirs", {}) or {}
    seed_root = artifact_root / f"seed{seed}"

    pretrain_out = seed_root / str(subdirs.get("pretrain", "pretrain"))
    eram_out = seed_root / str(subdirs.get("eram_pretrain", "eram_pretrain"))
    finetune_out = seed_root / str(subdirs.get("finetune", "finetune_boptest"))
    yearly_eval_out = seed_root / str(subdirs.get("yearly_eval", "yearly_eval"))

    paths = PipelinePaths(
        seed_root=seed_root,
        pretrain_out=pretrain_out,
        pretrain_model=pretrain_out / "models" / "ppo_model.zip",
        eram_out=eram_out,
        eram_model=eram_out / "models" / "ppo_model.zip",
        eram_weights=eram_out / "final_eram_weights.json",
        finetune_out=finetune_out,
        finetune_model=finetune_out / "models" / "ppo_model.zip",
        yearly_eval_out=yearly_eval_out,
    )
    return pipeline_cfg, paths


def write_manifest(
    manifest_path: Path,
    mode: str,
    config_dir: Path,
    artifact_root: Path,
    seed: int,
    pipeline_cfg: dict,
    paths: PipelinePaths,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": mode,
        "config_dir": str(config_dir),
        "artifact_root": str(artifact_root),
        "seed": seed,
        "pipeline_config": pipeline_cfg,
        "paths": {key: str(value) for key, value in asdict(paths).items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_command(cmd: list[str], label: str) -> None:
    print("=" * 88, flush=True)
    print(label, flush=True)
    print("=" * 88, flush=True)
    print("Command:", flush=True)
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Central MORL/PPO pipeline runner: surrogate pretrain -> optional BOPTEST fine-tune -> yearly validation."
    )
    parser.add_argument(
        "--config-dir",
        default="configs/morl_surrogate_ppo",
        help="Directory containing env.yaml, agent.yaml, train.yaml, pipeline.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["pretrain", "eram_pretrain", "finetune", "eval", "full", "full_eram"],
        default="full",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--artifact-root", default=None, help="Override artifact root from pipeline.yaml")
    parser.add_argument("--model", default=None, help="Override model path for finetune/eval")
    parser.add_argument("--weights-json", default=None, help="Override ERAM weight JSON for finetune")
    parser.add_argument("--pretrain-steps", type=int, default=None)
    parser.add_argument("--eram-iterations", type=int, default=None)
    parser.add_argument("--eram-chunk-steps", type=int, default=None)
    parser.add_argument("--eram-tau-w", type=float, default=None)
    parser.add_argument("--eram-adv-lr", type=float, default=None)
    parser.add_argument("--eram-init-weights", default=None)
    parser.add_argument("--finetune-steps", type=int, default=None)
    parser.add_argument("--finetune-learning-rate", type=float, default=None)
    parser.add_argument("--finetune-jitter-days", type=float, default=None)
    parser.add_argument("--eval-step-sec", type=int, default=None)
    parser.add_argument("--eval-scenario-days", type=int, default=None)
    parser.add_argument("--eval-select-timeout", type=float, default=None)
    parser.add_argument("--eval-advance-timeout", type=float, default=None)
    parser.add_argument("--eval-http-retries", type=int, default=None)
    parser.add_argument("--surrogate-kind", choices=["legacy_v3", "v35_raw", "v35_calibrated", "hybrid_v3_v35"], default=None)
    parser.add_argument("--surrogate-path", default=None)
    parser.add_argument("--surrogate-summary-json", default=None)
    parser.add_argument("--surrogate-checkpoint", default=None)
    parser.add_argument("--surrogate-base-model", default=None)
    parser.add_argument("--lambda-temp-disagree", type=float, default=None)
    parser.add_argument("--lambda-power-disagree", type=float, default=None)
    parser.add_argument("--power-feature-mode", default=None)
    parser.add_argument("--t-zone-feature-mode", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = project_path(args.config_dir)
    pipeline_cfg = load_yaml(config_dir / "pipeline.yaml")
    defaults = pipeline_cfg.get("defaults", {}) or {}

    seed = int(args.seed if args.seed is not None else defaults.get("seed", 42))
    artifact_root = project_path(args.artifact_root or pipeline_cfg.get("artifact_root", "outputs/morl_surrogate_ppo"))
    pipeline_cfg, paths = build_paths(config_dir, artifact_root, seed)

    write_manifest(
        manifest_path=paths.seed_root / "pipeline_manifest.json",
        mode=args.mode,
        config_dir=config_dir,
        artifact_root=artifact_root,
        seed=seed,
        pipeline_cfg=pipeline_cfg,
        paths=paths,
    )

    pretrain_steps = int(args.pretrain_steps if args.pretrain_steps is not None else defaults.get("pretrain_steps", 2_000_000))
    eram_iterations = int(args.eram_iterations if args.eram_iterations is not None else defaults.get("eram_iterations", 20))
    eram_chunk_steps = int(args.eram_chunk_steps if args.eram_chunk_steps is not None else defaults.get("eram_chunk_steps", 100_000))
    eram_tau_w = float(args.eram_tau_w if args.eram_tau_w is not None else defaults.get("eram_tau_w", 0.35))
    eram_adv_lr = float(args.eram_adv_lr if args.eram_adv_lr is not None else defaults.get("eram_adv_lr", 1.0))
    eram_init_weights = str(args.eram_init_weights or defaults.get("eram_init_weights", "0.34,0.33,0.33"))
    finetune_steps = int(args.finetune_steps if args.finetune_steps is not None else defaults.get("finetune_steps", 100_000))
    finetune_learning_rate = float(
        args.finetune_learning_rate if args.finetune_learning_rate is not None else defaults.get("finetune_learning_rate", 1e-4)
    )
    finetune_jitter_days = float(
        args.finetune_jitter_days if args.finetune_jitter_days is not None else defaults.get("finetune_jitter_days", 3.0)
    )
    eval_step_sec = int(args.eval_step_sec if args.eval_step_sec is not None else defaults.get("eval_step_sec", 3600))
    eval_scenario_days = int(
        args.eval_scenario_days if args.eval_scenario_days is not None else defaults.get("eval_scenario_days", 14)
    )
    eval_select_timeout = float(
        args.eval_select_timeout if args.eval_select_timeout is not None else defaults.get("eval_select_timeout", 300.0)
    )
    eval_advance_timeout = float(
        args.eval_advance_timeout if args.eval_advance_timeout is not None else defaults.get("eval_advance_timeout", 60.0)
    )
    eval_http_retries = int(
        args.eval_http_retries if args.eval_http_retries is not None else defaults.get("eval_http_retries", 3)
    )

    python_exe = resolve_python_executable()
    surrogate_overrides: list[str] = []
    if args.surrogate_kind:
        surrogate_overrides.extend(["--surrogate-kind", args.surrogate_kind])
    if args.surrogate_path:
        surrogate_overrides.extend(["--surrogate-path", args.surrogate_path])
    if args.surrogate_summary_json:
        surrogate_overrides.extend(["--surrogate-summary-json", args.surrogate_summary_json])
    if args.surrogate_checkpoint:
        surrogate_overrides.extend(["--surrogate-checkpoint", args.surrogate_checkpoint])
    if args.surrogate_base_model:
        surrogate_overrides.extend(["--surrogate-base-model", args.surrogate_base_model])
    if args.lambda_temp_disagree is not None:
        surrogate_overrides.extend(["--lambda-temp-disagree", str(args.lambda_temp_disagree)])
    if args.lambda_power_disagree is not None:
        surrogate_overrides.extend(["--lambda-power-disagree", str(args.lambda_power_disagree)])
    if args.power_feature_mode:
        surrogate_overrides.extend(["--power-feature-mode", args.power_feature_mode])
    if args.t_zone_feature_mode:
        surrogate_overrides.extend(["--t-zone-feature-mode", args.t_zone_feature_mode])

    pretrain_cmd = [
        python_exe,
        str(ROOT / "training" / "train_morl_surrogate.py"),
        "--config-dir",
        str(config_dir),
        "--seed",
        str(seed),
        "--steps",
        str(pretrain_steps),
        "--out_dir",
        str(paths.pretrain_out),
    ] + surrogate_overrides

    eram_cmd = [
        python_exe,
        str(ROOT / "training" / "train_morl_eram.py"),
        "--config-dir",
        str(config_dir),
        "--seed",
        str(seed),
        "--iterations",
        str(eram_iterations),
        "--chunk_steps",
        str(eram_chunk_steps),
        "--tau_w",
        str(eram_tau_w),
        "--adv_lr",
        str(eram_adv_lr),
        "--init_weights",
        eram_init_weights,
        "--out_dir",
        str(paths.eram_out),
    ] + surrogate_overrides

    def finetune_cmd(model_path: Path, weights_json: Path | None = None) -> list[str]:
        cmd = [
            python_exe,
            str(ROOT / "training" / "finetune_morl_boptest.py"),
            "--config-dir",
            str(config_dir),
            "--seed",
            str(seed),
            "--steps",
            str(finetune_steps),
            "--learning_rate",
            str(finetune_learning_rate),
            "--jitter_days",
            str(finetune_jitter_days),
            "--model",
            str(model_path),
            "--out_dir",
            str(paths.finetune_out),
        ]
        if weights_json is not None:
            cmd.extend(["--weights_json", str(weights_json)])
        return cmd

    def eval_cmd(model_path: Path) -> list[str]:
        return [
            python_exe,
            str(ROOT / "evaluation" / "yearly_validation_morl.py"),
            "--model",
            str(model_path),
            "--output_dir",
            str(paths.yearly_eval_out),
            "--step-sec",
            str(eval_step_sec),
            "--scenario-days",
            str(eval_scenario_days),
            "--select-timeout",
            str(eval_select_timeout),
            "--advance-timeout",
            str(eval_advance_timeout),
            "--http-retries",
            str(eval_http_retries),
        ]

    if args.mode == "pretrain":
        run_command(pretrain_cmd, "MORL SURROGATE PRETRAIN")
        return

    if args.mode == "eram_pretrain":
        run_command(eram_cmd, "MORL ERAM SURROGATE PRETRAIN")
        return

    if args.mode == "finetune":
        model_path = project_path(args.model) if args.model else paths.pretrain_model
        require_file(model_path, "Pretrained MORL model")
        weights_json = project_path(args.weights_json) if args.weights_json else None
        if weights_json is not None:
            require_file(weights_json, "Objective weight JSON")
        run_command(finetune_cmd(model_path, weights_json), "MORL BOPTEST FINE-TUNE")
        return

    if args.mode == "eval":
        model_path = project_path(args.model) if args.model else paths.finetune_model
        require_file(model_path, "MORL model for evaluation")
        run_command(eval_cmd(model_path), "MORL YEARLY VALIDATION")
        return

    if args.mode == "full":
        run_command(pretrain_cmd, "MORL SURROGATE PRETRAIN")
        require_file(paths.pretrain_model, "Pretrained MORL model")
        run_command(finetune_cmd(paths.pretrain_model), "MORL BOPTEST FINE-TUNE")
        require_file(paths.finetune_model, "Fine-tuned MORL model")
        run_command(eval_cmd(paths.finetune_model), "MORL YEARLY VALIDATION")
        return

    if args.mode == "full_eram":
        run_command(eram_cmd, "MORL ERAM SURROGATE PRETRAIN")
        require_file(paths.eram_model, "ERAM-pretrained MORL model")
        require_file(paths.eram_weights, "ERAM final weight JSON")
        run_command(finetune_cmd(paths.eram_model, paths.eram_weights), "MORL BOPTEST FINE-TUNE FROM ERAM")
        require_file(paths.finetune_model, "Fine-tuned MORL model")
        run_command(eval_cmd(paths.finetune_model), "MORL YEARLY VALIDATION")
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
