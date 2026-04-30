from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_1_2_surrogate_rmse.workflow_config import (
    BLOCK_NAME,
    COMFORT_HIGH_C,
    COMFORT_LOW_C,
    DEFAULT_COLLECTED_TRAIN_RUN,
    DEFAULT_COLLECTED_TRAIN_SUBSET_CSV,
    DEFAULT_HYBRID_DATASET_CSV,
    DEFAULT_HYBRID_TRAIN_RUN,
    DEFAULT_PREPARED_DATASET_CSV,
    DEFAULT_PREPARED_ROLLOUT_LONG_TRAIN_RUN,
    DEFAULT_PREPARED_ROLLOUT_TRAIN_RUN,
    DEFAULT_PREPARED_TRAIN_RUN,
    DEFAULT_TRAINING_OUTPUT_ROOT,
)
from surrogate.train_surrogate_backbone import (
    benchmark_backbone_speed,
    train_backbone,
    validate_backbone_safety,
)


DEFAULT_DATA = ROOT / "data" / "surrogate_v2" / "boptest_v2_tsupply.csv"
DEFAULT_OUTPUT_ROOT = DEFAULT_TRAINING_OUTPUT_ROOT


def _parse_horizons(raw: str) -> list[int]:
    items = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        items.append(int(value))
    items = sorted({v for v in items if v > 0})
    if not items:
        raise ValueError("At least one positive horizon is required")
    return items


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    parser_defaults = {
        "data": str(DEFAULT_DATA),
        "output_root": str(DEFAULT_OUTPUT_ROOT),
        "run_name": None,
        "epochs": 500,
        "batch_size": 256,
        "lr": 1e-3,
        "hidden_dim": 64,
        "patience": 30,
        "multi_horizons": "2,4",
        "checkpoint_metric": "val_loss",
        "rollout_val_horizons": None,
        "rollout_val_max_windows": 256,
        "rollout_selection_weights": None,
        "validate_safety": False,
        "benchmark": False,
    }

    def _maybe_set(name: str, value: Any) -> None:
        current = getattr(args, name)
        default = parser_defaults[name]
        if current is None or current == default:
            setattr(args, name, value)

    if args.preset == "hourly_baseline":
        _maybe_set("run_name", "hourly_baseline")
        _maybe_set("epochs", 500)
        _maybe_set("batch_size", 256)
        _maybe_set("lr", 1e-3)
        _maybe_set("hidden_dim", 64)
        _maybe_set("patience", 30)
        _maybe_set("multi_horizons", "2,4")
    elif args.preset == "hourly_long_rollout":
        _maybe_set("run_name", "hourly_long_rollout")
        _maybe_set("epochs", 700)
        _maybe_set("batch_size", 256)
        _maybe_set("lr", 7.5e-4)
        _maybe_set("hidden_dim", 96)
        _maybe_set("patience", 40)
        _maybe_set("multi_horizons", "2,4,8")
    elif args.preset == "hourly_wide":
        _maybe_set("run_name", "hourly_wide")
        _maybe_set("epochs", 800)
        _maybe_set("batch_size", 256)
        _maybe_set("lr", 7.5e-4)
        _maybe_set("hidden_dim", 128)
        _maybe_set("patience", 50)
        _maybe_set("multi_horizons", "2,4,8")
    elif args.preset == "prepared_15min":
        _maybe_set("data", str(DEFAULT_PREPARED_DATASET_CSV))
        _maybe_set("run_name", DEFAULT_PREPARED_TRAIN_RUN)
        _maybe_set("epochs", 700)
        _maybe_set("batch_size", 256)
        _maybe_set("lr", 7.5e-4)
        _maybe_set("hidden_dim", 96)
        _maybe_set("patience", 40)
        _maybe_set("multi_horizons", "4,8,16")
    elif args.preset == "prepared_15min_rollout_select":
        _maybe_set("data", str(DEFAULT_PREPARED_DATASET_CSV))
        _maybe_set("run_name", DEFAULT_PREPARED_ROLLOUT_TRAIN_RUN)
        _maybe_set("epochs", 700)
        _maybe_set("batch_size", 256)
        _maybe_set("lr", 7.5e-4)
        _maybe_set("hidden_dim", 96)
        _maybe_set("patience", 50)
        _maybe_set("multi_horizons", "4,8,16")
        _maybe_set("checkpoint_metric", "rollout_rmse")
        _maybe_set("rollout_val_horizons", "4,8,16")
        _maybe_set("rollout_val_max_windows", 256)
    elif args.preset == "prepared_15min_rollout_long":
        _maybe_set("data", str(DEFAULT_PREPARED_DATASET_CSV))
        _maybe_set("run_name", DEFAULT_PREPARED_ROLLOUT_LONG_TRAIN_RUN)
        _maybe_set("epochs", 700)
        _maybe_set("batch_size", 256)
        _maybe_set("lr", 5e-4)
        _maybe_set("hidden_dim", 96)
        _maybe_set("patience", 60)
        _maybe_set("multi_horizons", "4,8,16")
        _maybe_set("checkpoint_metric", "rollout_rmse")
        _maybe_set("rollout_val_horizons", "4,8,16")
        _maybe_set("rollout_val_max_windows", 384)
        _maybe_set("rollout_selection_weights", "1,2,4")
    elif args.preset == "collected_15min_focus":
        _maybe_set("data", str(DEFAULT_COLLECTED_TRAIN_SUBSET_CSV))
        _maybe_set("run_name", DEFAULT_COLLECTED_TRAIN_RUN)
        _maybe_set("epochs", 700)
        _maybe_set("batch_size", 256)
        _maybe_set("lr", 7.5e-4)
        _maybe_set("hidden_dim", 96)
        _maybe_set("patience", 40)
        _maybe_set("multi_horizons", "4,8,16")
    elif args.preset == "hybrid_15min_anchor":
        _maybe_set("data", str(DEFAULT_HYBRID_DATASET_CSV))
        _maybe_set("run_name", DEFAULT_HYBRID_TRAIN_RUN)
        _maybe_set("epochs", 700)
        _maybe_set("batch_size", 256)
        _maybe_set("lr", 7.5e-4)
        _maybe_set("hidden_dim", 96)
        _maybe_set("patience", 40)
        _maybe_set("multi_horizons", "4,8,16")

    if args.run_name is None:
        args.run_name = args.preset

    return args


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block 1.2 launcher for 15-minute surrogate backbone retraining."
    )
    parser.add_argument(
        "--preset",
        choices=[
            "hourly_baseline",
            "hourly_long_rollout",
            "hourly_wide",
            "prepared_15min",
            "prepared_15min_rollout_select",
            "prepared_15min_rollout_long",
            "collected_15min_focus",
            "hybrid_15min_anchor",
        ],
        default="prepared_15min",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--multi-horizons", default="2,4")
    parser.add_argument("--checkpoint-metric", choices=["val_loss", "rollout_rmse"], default="val_loss")
    parser.add_argument("--rollout-val-horizons", default=None)
    parser.add_argument("--rollout-val-max-windows", type=int, default=256)
    parser.add_argument("--rollout-selection-weights", default=None)
    parser.add_argument("--validate-safety", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--comfort-low", type=float, default=COMFORT_LOW_C)
    parser.add_argument("--comfort-high", type=float, default=COMFORT_HIGH_C)
    args = parser.parse_args()
    args = _apply_preset(args)

    data_path = Path(args.data)
    output_root = Path(args.output_root)
    run_dir = output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    horizons = _parse_horizons(args.multi_horizons)
    rollout_horizons = _parse_horizons(args.rollout_val_horizons) if args.rollout_val_horizons else None
    rollout_weights = (
        [float(item.strip()) for item in args.rollout_selection_weights.split(",") if item.strip()]
        if args.rollout_selection_weights
        else None
    )

    config_snapshot = {
        "block": BLOCK_NAME,
        "goal": "surrogate_rollout_rmse_improvement",
        "preset": args.preset,
        "run_name": args.run_name,
        "data": str(data_path),
        "output_dir": str(run_dir),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "patience": int(args.patience),
        "multi_horizons": horizons,
        "checkpoint_metric": args.checkpoint_metric,
        "rollout_val_horizons": rollout_horizons,
        "rollout_val_max_windows": int(args.rollout_val_max_windows),
        "rollout_selection_weights": rollout_weights,
        "validate_safety": bool(args.validate_safety),
        "benchmark": bool(args.benchmark),
        "comfort_band_c": [float(args.comfort_low), float(args.comfort_high)],
        "notes": [
            "Canonical Block 1 remains frozen on v3.5 heads_only.",
            "This launcher is the active Block 1.2 entrypoint for 15-minute surrogate improvement.",
            "Safety validation uses the configured comfort band instead of a fixed 22 C target.",
        ],
    }
    _write_json(run_dir / "launcher_config_snapshot.json", config_snapshot)

    print("=" * 88)
    print("BLOCK 1.2 SURROGATE BACKBONE RETRAIN")
    print("=" * 88)
    print(f"Preset:          {args.preset}")
    print(f"Run name:        {args.run_name}")
    print(f"Data:            {data_path}")
    print(f"Output dir:      {run_dir}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"LR:              {args.lr}")
    print(f"Hidden dim:      {args.hidden_dim}")
    print(f"Patience:        {args.patience}")
    print(f"Multi-horizons:  {horizons}")
    print(f"Checkpoint met.: {args.checkpoint_metric}")
    print(f"Rollout val hz:  {rollout_horizons}")
    print(f"Rollout windows: {args.rollout_val_max_windows}")
    print(f"Rollout weights: {rollout_weights}")
    print(f"Validate safety: {args.validate_safety}")
    print(f"Benchmark speed: {args.benchmark}")
    print(f"Comfort band:    [{args.comfort_low}, {args.comfort_high}] C")

    model_path = train_backbone(
        data_path=str(data_path),
        output_dir=str(run_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        patience=args.patience,
        multi_horizons=horizons,
        checkpoint_metric=args.checkpoint_metric,
        rollout_val_horizons=rollout_horizons,
        rollout_val_max_windows=args.rollout_val_max_windows,
        rollout_selection_weights=rollout_weights,
    )

    summary: dict[str, Any] = {
        "block": BLOCK_NAME,
        "preset": args.preset,
        "run_name": args.run_name,
        "model_path": str(model_path),
        "output_dir": str(run_dir),
        "data": str(data_path),
        "multi_horizons": horizons,
        "comfort_band_c": [float(args.comfort_low), float(args.comfort_high)],
    }

    if args.validate_safety:
        safety_metrics = validate_backbone_safety(
            model_path=model_path,
            data_path=str(data_path),
            horizons=[1, 2, 4, 6],
            t_low=float(args.comfort_low),
            t_high=float(args.comfort_high),
        )
        summary["safety_metrics"] = safety_metrics

    if args.benchmark:
        benchmark_backbone_speed(model_path)
        summary["benchmark_ran"] = True

    _write_json(run_dir / "launcher_summary.json", summary)

    print("=" * 88)
    print("BLOCK 1.2 RUN COMPLETE")
    print("=" * 88)
    print(f"Model:   {model_path}")
    print(f"Config:  {run_dir / 'launcher_config_snapshot.json'}")
    print(f"Summary: {run_dir / 'launcher_summary.json'}")


if __name__ == "__main__":
    main()
