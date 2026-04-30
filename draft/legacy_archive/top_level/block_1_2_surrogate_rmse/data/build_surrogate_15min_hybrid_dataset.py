from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_1_2_surrogate_rmse.workflow_config import (  # noqa: E402
    DEFAULT_HYBRID_DATASET_CSV,
    DEFAULT_HYBRID_OUTPUT_DIR,
    DEFAULT_PREPARED_DATASET_CSV,
    DEFAULT_COLLECTED_TRAIN_SUBSET_CSV,
)


REQUIRED_COLUMNS = [
    "episode_id",
    "step",
    "step_sec",
    "sim_time_sec",
    "t_zone",
    "t_amb",
    "hour",
    "day",
    "a0_raw",
    "a1_raw",
    "t_supply_cmd_c",
    "fan_cmd_u",
    "t_zone_next",
    "delta_t",
    "p_total",
    "policy",
    "season",
    "controller_source",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df["dataset_source"] = dataset_name
    for optional_col, default_value in {
        "source_group": dataset_name,
        "source_trace": "",
        "testcase": "",
        "collector": "",
    }.items():
        if optional_col not in df.columns:
            df[optional_col] = default_value
    return df


def _sample_filtered_collected(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    allowed_policies = [item.strip() for item in args.collected_policies.split(",") if item.strip()]
    mask = (
        df["policy"].astype(str).isin(allowed_policies)
        & df["t_zone"].between(args.zone_min, args.zone_max)
        & df["t_zone_next"].between(args.zone_min, args.zone_max)
        & df["delta_t"].abs().le(args.max_abs_delta)
        & df["p_total"].le(args.max_power)
    )
    filtered = df.loc[mask].copy()

    max_rows = int(round(args.max_collected_ratio * args.prepared_rows))
    if max_rows <= 0:
        sampled = filtered.iloc[0:0].copy()
    elif len(filtered) > max_rows:
        sampled = filtered.sample(n=max_rows, random_state=args.seed).copy()
    else:
        sampled = filtered.copy()

    sampled["dataset_source"] = "collected_anchor"
    sampled["source_group"] = "collected_anchor"

    stats = {
        "allowed_policies": allowed_policies,
        "zone_min_c": float(args.zone_min),
        "zone_max_c": float(args.zone_max),
        "max_abs_delta_c": float(args.max_abs_delta),
        "max_power_w": float(args.max_power),
        "max_collected_ratio": float(args.max_collected_ratio),
        "filtered_rows": int(len(filtered)),
        "sampled_rows": int(len(sampled)),
        "filtered_policy_counts": filtered["policy"].value_counts().to_dict(),
        "sampled_policy_counts": sampled["policy"].value_counts().to_dict(),
        "sampled_season_counts": sampled["season"].value_counts().to_dict(),
    }
    return sampled, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a hybrid 15-minute surrogate dataset: prepared benchmark traces plus filtered collected anchor transitions."
    )
    parser.add_argument("--prepared-csv", default=str(DEFAULT_PREPARED_DATASET_CSV))
    parser.add_argument("--collected-csv", default=str(DEFAULT_COLLECTED_TRAIN_SUBSET_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_HYBRID_DATASET_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_HYBRID_OUTPUT_DIR))
    parser.add_argument("--collected-policies", default="thermostatic_noise,pulse,cool,mixed")
    parser.add_argument("--zone-min", type=float, default=18.5)
    parser.add_argument("--zone-max", type=float, default=27.0)
    parser.add_argument("--max-abs-delta", type=float, default=1.5)
    parser.add_argument("--max-power", type=float, default=1000.0)
    parser.add_argument("--max-collected-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepared_csv = Path(args.prepared_csv)
    collected_csv = Path(args.collected_csv)
    output_csv = Path(args.output_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    prepared = _load_dataset(prepared_csv, "prepared")
    collected = _load_dataset(collected_csv, "collected_train_subset")
    args.prepared_rows = len(prepared)

    collected_anchor, collected_stats = _sample_filtered_collected(collected, args)
    hybrid = pd.concat([prepared, collected_anchor], ignore_index=True)
    hybrid = hybrid.sort_values(["dataset_source", "episode_id", "step"]).reset_index(drop=True)
    hybrid.to_csv(output_csv, index=False)

    summary = {
        "block": "1.2",
        "dataset_kind": "hybrid_15min_anchor",
        "prepared_csv": str(prepared_csv),
        "collected_csv": str(collected_csv),
        "output_csv": str(output_csv),
        "prepared_rows": int(len(prepared)),
        "collected_rows": int(len(collected)),
        "hybrid_rows": int(len(hybrid)),
        "prepared_episode_count": int(prepared["episode_id"].nunique()),
        "collected_anchor_episode_count": int(collected_anchor["episode_id"].nunique()),
        "hybrid_episode_count": int(hybrid["episode_id"].nunique()),
        "hybrid_t_zone_range_c": [float(hybrid["t_zone"].min()), float(hybrid["t_zone"].max())],
        "hybrid_t_next_range_c": [float(hybrid["t_zone_next"].min()), float(hybrid["t_zone_next"].max())],
        "hybrid_abs_delta_mean_c": float(hybrid["delta_t"].abs().mean()),
        "hybrid_policy_counts": hybrid["policy"].value_counts().to_dict(),
        "hybrid_source_counts": hybrid["dataset_source"].value_counts().to_dict(),
        "collected_anchor_filter": collected_stats,
    }
    _write_json(output_dir / "dataset_summary.json", summary)

    summary_rows = []
    for source_name, source_df in [("prepared", prepared), ("collected_anchor", collected_anchor), ("hybrid", hybrid)]:
        summary_rows.append(
            {
                "source": source_name,
                "rows": int(len(source_df)),
                "episodes": int(source_df["episode_id"].nunique()),
                "t_zone_min_c": float(source_df["t_zone"].min()) if len(source_df) else None,
                "t_zone_max_c": float(source_df["t_zone"].max()) if len(source_df) else None,
                "abs_delta_mean_c": float(source_df["delta_t"].abs().mean()) if len(source_df) else None,
                "mean_power_w": float(source_df["p_total"].mean()) if len(source_df) else None,
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "mix_summary.csv", index=False)

    print("=" * 88)
    print("BLOCK 1.2 BUILD HYBRID 15-MINUTE DATASET")
    print("=" * 88)
    print(f"Prepared rows:          {len(prepared):,}")
    print(f"Collected input rows:   {len(collected):,}")
    print(f"Collected anchor rows:  {len(collected_anchor):,}")
    print(f"Hybrid rows:            {len(hybrid):,}")
    print(f"Hybrid episodes:        {hybrid['episode_id'].nunique()}")
    print(f"Policies kept:          {collected_stats['allowed_policies']}")
    print(f"Output CSV:             {output_csv}")
    print(f"Summary JSON:           {output_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
