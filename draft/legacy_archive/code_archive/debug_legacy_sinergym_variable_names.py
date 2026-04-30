from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.loader import load_all_configs
from envs.factory import EnvFactory


DEFAULT_CONFIG_DIR = "configs/legacy_sinergym"
DEFAULT_OUT_DIR = "outputs/legacy_sinergym/debug_variable_names"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect legacy Sinergym wrappers and discover observation/action variable names."
    )
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head", type=int, default=25, help="How many observation values to preview")
    return parser.parse_args()


def _flatten_obs(obs: Any) -> np.ndarray:
    return np.asarray(obs, dtype=np.float64).reshape(-1)


def _iter_env_layers(env: Any) -> list[tuple[int, Any]]:
    layers: list[tuple[int, Any]] = []
    cursor = env
    visited: set[int] = set()
    depth = 0
    while cursor is not None and id(cursor) not in visited:
        visited.add(id(cursor))
        layers.append((depth, cursor))
        cursor = getattr(cursor, "env", None)
        depth += 1
    return layers


def _is_string_sequence(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if not value:
        return False
    return all(isinstance(item, str) for item in value)


def _short_repr(value: Any, limit: int = 180) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        text = repr(value)
    text = text.replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"


def _candidate_rows_from_attr(
    *,
    depth: int,
    layer: Any,
    attr_name: str,
    value: Any,
    obs_dim: int,
    action_dim: int,
    obs_values: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    nested_rows: list[dict[str, Any]] = []

    layer_class = layer.__class__.__name__
    layer_module = layer.__class__.__module__

    def add_summary(kind: str, length: int | None, preview: str) -> None:
        summary_rows.append(
            {
                "depth": depth,
                "layer_class": layer_class,
                "layer_module": layer_module,
                "attr_name": attr_name,
                "kind": kind,
                "length": length,
                "preview": preview,
            }
        )

    if _is_string_sequence(value):
        seq = list(value)
        add_summary("string_sequence", len(seq), _short_repr(seq[:8]))
        if len(seq) == obs_dim:
            for idx, name in enumerate(seq):
                mapping_rows.append(
                    {
                        "depth": depth,
                        "layer_class": layer_class,
                        "attr_name": attr_name,
                        "mapping_kind": "observation",
                        "index": idx,
                        "name": name,
                        "current_obs_value": float(obs_values[idx]),
                    }
                )
        if len(seq) == action_dim:
            for idx, name in enumerate(seq):
                mapping_rows.append(
                    {
                        "depth": depth,
                        "layer_class": layer_class,
                        "attr_name": attr_name,
                        "mapping_kind": "action",
                        "index": idx,
                        "name": name,
                        "current_obs_value": np.nan,
                    }
                )
        return summary_rows, mapping_rows, nested_rows

    if isinstance(value, dict):
        add_summary("dict", len(value), _short_repr(list(value.keys())[:12]))
        for key, nested_value in value.items():
            nested_attr = f"{attr_name}.{key}"
            if _is_string_sequence(nested_value):
                seq = list(nested_value)
                nested_rows.append(
                    {
                        "depth": depth,
                        "layer_class": layer_class,
                        "attr_name": nested_attr,
                        "kind": "string_sequence",
                        "length": len(seq),
                        "preview": _short_repr(seq[:8]),
                    }
                )
                if len(seq) == obs_dim:
                    for idx, name in enumerate(seq):
                        mapping_rows.append(
                            {
                                "depth": depth,
                                "layer_class": layer_class,
                                "attr_name": nested_attr,
                                "mapping_kind": "observation",
                                "index": idx,
                                "name": name,
                                "current_obs_value": float(obs_values[idx]),
                            }
                        )
                if len(seq) == action_dim:
                    for idx, name in enumerate(seq):
                        mapping_rows.append(
                            {
                                "depth": depth,
                                "layer_class": layer_class,
                                "attr_name": nested_attr,
                                "mapping_kind": "action",
                                "index": idx,
                                "name": name,
                                "current_obs_value": np.nan,
                            }
                        )
        return summary_rows, mapping_rows, nested_rows

    if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S", "O"}:
        add_summary("ndarray_strings", int(value.size), _short_repr(value.reshape(-1)[:8].tolist()))
        seq = value.reshape(-1).tolist()
        if seq and all(isinstance(item, str) for item in seq):
            if len(seq) == obs_dim:
                for idx, name in enumerate(seq):
                    mapping_rows.append(
                        {
                            "depth": depth,
                            "layer_class": layer_class,
                            "attr_name": attr_name,
                            "mapping_kind": "observation",
                            "index": idx,
                            "name": name,
                            "current_obs_value": float(obs_values[idx]),
                        }
                    )
            if len(seq) == action_dim:
                for idx, name in enumerate(seq):
                    mapping_rows.append(
                        {
                            "depth": depth,
                            "layer_class": layer_class,
                            "attr_name": attr_name,
                            "mapping_kind": "action",
                            "index": idx,
                            "name": name,
                            "current_obs_value": np.nan,
                        }
                    )
        return summary_rows, mapping_rows, nested_rows

    return summary_rows, mapping_rows, nested_rows


def _safe_getattr(obj: Any, attr: str) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return None


def _collect_candidates(layers: list[tuple[int, Any]], obs_values: np.ndarray, action_dim: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    obs_dim = int(obs_values.size)

    priority_attrs = [
        "observation_variables",
        "observation_var_names",
        "observation_names",
        "variables",
        "obs_variables",
        "obs_var_names",
        "action_variables",
        "action_var_names",
    ]

    for depth, layer in layers:
        seen_attrs: set[str] = set()
        attrs = priority_attrs + sorted(a for a in dir(layer) if not a.startswith("__"))
        for attr_name in attrs:
            if attr_name in seen_attrs:
                continue
            seen_attrs.add(attr_name)
            if attr_name.startswith("_") and attr_name not in priority_attrs:
                continue
            value = _safe_getattr(layer, attr_name)
            if value is None or callable(value):
                continue
            rows, mappings, nested_rows = _candidate_rows_from_attr(
                depth=depth,
                layer=layer,
                attr_name=attr_name,
                value=value,
                obs_dim=obs_dim,
                action_dim=action_dim,
                obs_values=obs_values,
            )
            summary_rows.extend(rows)
            summary_rows.extend(nested_rows)
            mapping_rows.extend(mappings)

    summary_df = pd.DataFrame(summary_rows).drop_duplicates().reset_index(drop=True)
    mapping_df = pd.DataFrame(mapping_rows).drop_duplicates().reset_index(drop=True)
    return summary_df, mapping_df


def _layer_rows(layers: list[tuple[int, Any]]) -> pd.DataFrame:
    rows = []
    for depth, layer in layers:
        rows.append(
            {
                "depth": depth,
                "layer_class": layer.__class__.__name__,
                "layer_module": layer.__class__.__module__,
                "has_env_attr": hasattr(layer, "env"),
            }
        )
    return pd.DataFrame(rows)


def _obs_preview_rows(obs_values: np.ndarray, head: int) -> pd.DataFrame:
    limit = min(int(head), int(obs_values.size))
    rows = []
    for idx in range(limit):
        rows.append({"obs_index": idx, "current_value": float(obs_values[idx])})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_all_configs(args.config_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = EnvFactory.create(dict(cfg["env"]))

    try:
        obs, info = env.reset(seed=args.seed)
        obs_values = _flatten_obs(obs)
        action_dim = int(np.prod(env.action_space.shape))
        layers = _iter_env_layers(env)

        layer_df = _layer_rows(layers)
        layer_path = out_dir / "wrapper_layers.csv"
        layer_df.to_csv(layer_path, index=False)

        preview_df = _obs_preview_rows(obs_values, args.head)
        preview_path = out_dir / "observation_preview.csv"
        preview_df.to_csv(preview_path, index=False)

        summary_df, mapping_df = _collect_candidates(layers, obs_values, action_dim)
        summary_path = out_dir / "attribute_candidate_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        mapping_path = out_dir / "observation_name_mapping_candidates.csv"
        if mapping_df.empty:
            pd.DataFrame(
                columns=["depth", "layer_class", "attr_name", "mapping_kind", "index", "name", "current_obs_value"]
            ).to_csv(mapping_path, index=False)
        else:
            mapping_df.sort_values(["mapping_kind", "depth", "attr_name", "index"]).to_csv(mapping_path, index=False)

            for candidate in mapping_df[["depth", "layer_class", "attr_name", "mapping_kind"]].drop_duplicates().itertuples(index=False):
                subset = mapping_df[
                    (mapping_df["depth"] == candidate.depth)
                    & (mapping_df["layer_class"] == candidate.layer_class)
                    & (mapping_df["attr_name"] == candidate.attr_name)
                    & (mapping_df["mapping_kind"] == candidate.mapping_kind)
                ].sort_values("index")
                filename = (
                    f"{candidate.mapping_kind}_mapping_depth{candidate.depth}_"
                    f"{_sanitize(candidate.layer_class)}_{_sanitize(candidate.attr_name)}.csv"
                )
                subset.to_csv(out_dir / filename, index=False)

        report_lines = [
            "LEGACY SINERGYM VARIABLE NAME DEBUG",
            "===================================",
            f"Config dir: {args.config_dir}",
            f"Seed: {args.seed}",
            f"Observation dim: {obs_values.size}",
            f"Action dim: {action_dim}",
            "",
            "Wrapper chain:",
        ]
        for row in layer_df.itertuples(index=False):
            report_lines.append(
                f"  depth={row.depth} class={row.layer_class} module={row.layer_module}"
            )

        report_lines.extend(
            [
                "",
                f"Saved layer list: {layer_path}",
                f"Saved observation preview: {preview_path}",
                f"Saved attribute candidates: {summary_path}",
                f"Saved mapping candidates: {mapping_path}",
                "",
            ]
        )

        if mapping_df.empty:
            report_lines.append("No exact observation/action name mapping candidates found.")
        else:
            report_lines.append("Exact-length mapping candidates:")
            for candidate in mapping_df[["depth", "layer_class", "attr_name", "mapping_kind"]].drop_duplicates().itertuples(index=False):
                subset = mapping_df[
                    (mapping_df["depth"] == candidate.depth)
                    & (mapping_df["layer_class"] == candidate.layer_class)
                    & (mapping_df["attr_name"] == candidate.attr_name)
                    & (mapping_df["mapping_kind"] == candidate.mapping_kind)
                ].sort_values("index")
                names_preview = ", ".join(
                    f"{int(row.index)}:{row.name}" for row in subset.head(8).itertuples(index=False)
                )
                report_lines.append(
                    f"  - {candidate.mapping_kind} via depth={candidate.depth} "
                    f"{candidate.layer_class}.{candidate.attr_name}: {names_preview}"
                )

        report_path = out_dir / "variable_name_report.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print("\n".join(report_lines))
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
