from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_TRACE_DIRS = [
    ROOT / "outputs" / "bestest_air_article7_style_15min" / "traces",
    ROOT / "outputs" / "article22_mlp_heat" / "bestest_air_article7_style_15min_ms_guarded" / "traces",
    ROOT / "outputs" / "bestest_air_article7_style_15min_article22_gru" / "traces",
]
DEFAULT_OUTPUT_CSV = ROOT / "data" / "block_1_2_surrogate_rmse" / "boptest_block12_15min_prepared.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "block_1_2_surrogate_rmse" / "prepared_15min_dataset"

REQUIRED_TRACE_COLUMNS = [
    "step",
    "sim_time_sec",
    "t_zone_c",
    "t_amb_c",
    "p_total_w",
    "a0",
    "a1",
    "t_supply_cmd_c",
    "fan_cmd_u",
    "controller_source",
]


@dataclass(frozen=True)
class TraceBuildResult:
    episode_id: str
    controller: str
    scenario: str
    source_group: str
    input_rows: int
    output_rows: int
    step_sec: float
    temp_min_c: float
    temp_max_c: float
    power_mean_w: float


def _parse_trace_dirs(raw: str | None) -> list[Path]:
    if raw is None:
        return list(DEFAULT_TRACE_DIRS)
    items = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        items.append(Path(value))
    return items


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _infer_controller_from_stem(stem: str) -> tuple[str, str]:
    parts = stem.split("_")
    if len(parts) < 2:
        return stem, "unknown"
    controller = parts[-1]
    scenario = "_".join(parts[:-1])
    return scenario, controller


def _validate_trace(df: pd.DataFrame, path: Path) -> None:
    missing = [col for col in REQUIRED_TRACE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required trace columns in {path}: {missing}")
    if len(df) < 2:
        raise ValueError(f"Trace is too short to create transitions: {path}")


def _prepare_trace(path: Path) -> tuple[pd.DataFrame, TraceBuildResult]:
    df = pd.read_csv(path).copy()
    _validate_trace(df, path)
    df = df.sort_values("sim_time_sec").reset_index(drop=True)

    diffs = df["sim_time_sec"].diff().dropna().to_numpy(dtype=float)
    unique_diffs = np.unique(np.round(diffs, 6))
    if len(unique_diffs) == 0:
        raise ValueError(f"Could not infer step size from trace: {path}")
    if len(unique_diffs) > 1:
        raise ValueError(f"Trace has non-uniform timestep in {path}: {unique_diffs.tolist()}")
    step_sec = float(unique_diffs[0])

    scenario, controller = _infer_controller_from_stem(path.stem)
    source_group = path.parents[1].name
    episode_id = f"{source_group}__{path.stem}"

    current = df.iloc[:-1].copy().reset_index(drop=True)
    nxt = df.iloc[1:].copy().reset_index(drop=True)

    prepared = pd.DataFrame(
        {
            "episode_id": episode_id,
            "step": current["step"].astype(int),
            "step_sec": step_sec,
            "sim_time_sec": current["sim_time_sec"].astype(float),
            "t_zone": current["t_zone_c"].astype(float),
            "t_amb": current["t_amb_c"].astype(float),
            "hour": ((current["sim_time_sec"].astype(float) / 3600.0) % 24.0),
            "day": ((current["sim_time_sec"].astype(float) / 86400.0) % 365.0),
            "a0_raw": current["a0"].astype(float),
            "a1_raw": current["a1"].astype(float),
            "t_supply_cmd_c": current["t_supply_cmd_c"].astype(float),
            "fan_cmd_u": current["fan_cmd_u"].astype(float),
            "t_zone_next": nxt["t_zone_c"].astype(float),
            "delta_t": nxt["t_zone_c"].astype(float) - current["t_zone_c"].astype(float),
            "p_total": current["p_total_w"].astype(float),
            "policy": controller,
            "season": scenario,
            "controller_source": current["controller_source"].astype(str),
            "source_group": source_group,
            "source_trace": str(path.relative_to(ROOT)),
        }
    )

    result = TraceBuildResult(
        episode_id=episode_id,
        controller=controller,
        scenario=scenario,
        source_group=source_group,
        input_rows=int(len(df)),
        output_rows=int(len(prepared)),
        step_sec=step_sec,
        temp_min_c=float(prepared["t_zone"].min()),
        temp_max_c=float(prepared["t_zone"].max()),
        power_mean_w=float(prepared["p_total"].mean()),
    )
    return prepared, result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a Block 1.2 15-minute surrogate dataset from existing benchmark trace CSV files."
    )
    parser.add_argument(
        "--trace-dirs",
        default=None,
        help="Comma-separated trace directories. Defaults to known 15-minute benchmark trace folders.",
    )
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    trace_dirs = _parse_trace_dirs(args.trace_dirs)
    output_csv = Path(args.output_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_parent(output_csv)

    trace_files: list[Path] = []
    missing_dirs: list[str] = []
    for trace_dir in trace_dirs:
        if trace_dir.exists():
            trace_files.extend(sorted(trace_dir.glob("*.csv")))
        else:
            missing_dirs.append(str(trace_dir))

    trace_files = [path for path in trace_files if path.name != "summary.csv"]
    if not trace_files:
        raise FileNotFoundError(
            "No 15-minute trace CSV files were found. "
            "Run the 15-minute benchmark path first or switch to a new collection script."
        )

    all_frames: list[pd.DataFrame] = []
    build_rows: list[dict[str, Any]] = []
    for path in trace_files:
        prepared, result = _prepare_trace(path)
        all_frames.append(prepared)
        build_rows.append(
            {
                "episode_id": result.episode_id,
                "controller": result.controller,
                "scenario": result.scenario,
                "source_group": result.source_group,
                "input_rows": result.input_rows,
                "output_rows": result.output_rows,
                "step_sec": result.step_sec,
                "temp_min_c": result.temp_min_c,
                "temp_max_c": result.temp_max_c,
                "power_mean_w": result.power_mean_w,
            }
        )

    dataset = pd.concat(all_frames, ignore_index=True)
    dataset.to_csv(output_csv, index=False)

    trace_summary = pd.DataFrame(build_rows).sort_values(["source_group", "scenario", "controller"]).reset_index(drop=True)
    trace_summary_csv = output_dir / "trace_summary.csv"
    trace_summary.to_csv(trace_summary_csv, index=False)

    dataset_summary = {
        "block": "1.2",
        "dataset_kind": "prepared_from_existing_15min_benchmark_traces",
        "output_csv": str(output_csv),
        "rows": int(len(dataset)),
        "episodes": int(dataset["episode_id"].nunique()),
        "controllers": sorted(dataset["policy"].astype(str).unique().tolist()),
        "scenarios": sorted(dataset["season"].astype(str).unique().tolist()),
        "source_groups": sorted(dataset["source_group"].astype(str).unique().tolist()),
        "step_sec_unique": sorted(dataset["step_sec"].astype(float).unique().tolist()),
        "t_zone_range_c": [float(dataset["t_zone"].min()), float(dataset["t_zone"].max())],
        "t_amb_range_c": [float(dataset["t_amb"].min()), float(dataset["t_amb"].max())],
        "power_range_w": [float(dataset["p_total"].min()), float(dataset["p_total"].max())],
        "limitations": [
            "This dataset is prepared from existing 15-minute closed-loop benchmark traces.",
            "It is heating-window focused and controller-biased, not a broad exploration dataset.",
            "It is suitable as the first Block 1.2 15-minute bootstrap dataset, not yet the final canonical surrogate corpus.",
        ],
        "missing_trace_dirs": missing_dirs,
        "trace_summary_csv": str(trace_summary_csv),
    }
    _write_json(output_dir / "dataset_summary.json", dataset_summary)

    print("=" * 88)
    print("BLOCK 1.2 PREPARE 15-MINUTE DATASET")
    print("=" * 88)
    print(f"Trace files:      {len(trace_files)}")
    print(f"Prepared rows:    {len(dataset):,}")
    print(f"Episodes:         {dataset['episode_id'].nunique()}")
    print(f"Controllers:      {sorted(dataset['policy'].astype(str).unique().tolist())}")
    print(f"Scenarios:        {sorted(dataset['season'].astype(str).unique().tolist())}")
    print(f"Step seconds:     {sorted(dataset['step_sec'].astype(float).unique().tolist())}")
    print(f"Output CSV:       {output_csv}")
    print(f"Trace summary:    {trace_summary_csv}")
    print(f"Dataset summary:  {output_dir / 'dataset_summary.json'}")
    if missing_dirs:
        print(f"Missing dirs:     {missing_dirs}")


if __name__ == "__main__":
    main()
