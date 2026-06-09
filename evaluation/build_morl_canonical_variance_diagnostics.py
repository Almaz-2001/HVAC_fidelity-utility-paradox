from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDFIX_ROOT = ROOT / "outputs" / "morl_pareto_hybrid_power_only_seedfix"
REPORTS = ROOT / "reports"

METRICS = ["rmse", "mae", "within_1c_pct", "within_05c_pct", "viol_pct", "energy_kwh", "ms"]


def parse_seed(seed_dir: Path) -> int:
    text = seed_dir.name.replace("seed", "")
    return int(text) if text.isdigit() else -1


def collect_rows(point_root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed_dir in sorted(point_root.glob("seed*"), key=parse_seed):
        seed = parse_seed(seed_dir)
        if seed < 0:
            continue
        summary = seed_dir / "yearly_eval" / "morl_yearly_summary.csv"
        if not summary.exists():
            continue
        df = pd.read_csv(summary)
        for _, row in df.iterrows():
            out = {
                "canonical": point_root.name,
                "seed": seed,
                "scenario": str(row["name"]),
                "source": str(summary.relative_to(ROOT)),
            }
            for metric in METRICS:
                out[metric] = float(row[metric])
            rows.append(out)
    return pd.DataFrame(rows)


def summarize_monthly(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["canonical", "scenario"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        rmse_min=("rmse", "min"),
        rmse_max=("rmse", "max"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        mae_min=("mae", "min"),
        mae_max=("mae", "max"),
        within_1c_pct_mean=("within_1c_pct", "mean"),
        within_1c_pct_std=("within_1c_pct", "std"),
        within_1c_pct_min=("within_1c_pct", "min"),
        within_1c_pct_max=("within_1c_pct", "max"),
        viol_pct_mean=("viol_pct", "mean"),
        viol_pct_std=("viol_pct", "std"),
        viol_pct_min=("viol_pct", "min"),
        viol_pct_max=("viol_pct", "max"),
        energy_kwh_mean=("energy_kwh", "mean"),
        energy_kwh_std=("energy_kwh", "std"),
        energy_kwh_min=("energy_kwh", "min"),
        energy_kwh_max=("energy_kwh", "max"),
        ms_mean=("ms", "mean"),
        ms_std=("ms", "std"),
        ms_min=("ms", "min"),
        ms_max=("ms", "max"),
    )
    grouped["ms_cv"] = grouped["ms_std"] / grouped["ms_mean"].replace(0.0, pd.NA)
    grouped["ms_range"] = grouped["ms_max"] - grouped["ms_min"]
    grouped["trigger_like_month"] = (grouped["ms_cv"] > 0.30) & (grouped["ms_std"] > 0.01)
    return grouped.sort_values(["canonical", "ms_std"], ascending=[True, False])


def summarize_yearly(df: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        df.groupby(["canonical", "seed"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            within_1c_pct_mean=("within_1c_pct", "mean"),
            within_05c_pct_mean=("within_05c_pct", "mean"),
            violation_pct_mean=("viol_pct", "mean"),
            energy_kwh_sum=("energy_kwh", "sum"),
            ms_mean=("ms", "mean"),
        )
        .sort_values(["canonical", "seed"])
    )
    summary = (
        per_seed.groupby("canonical", as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            rmse_mean=("rmse_mean", "mean"),
            rmse_std=("rmse_mean", "std"),
            mae_mean=("mae_mean", "mean"),
            mae_std=("mae_mean", "std"),
            within_1c_pct_mean=("within_1c_pct_mean", "mean"),
            within_1c_pct_std=("within_1c_pct_mean", "std"),
            violation_pct_mean=("violation_pct_mean", "mean"),
            violation_pct_std=("violation_pct_mean", "std"),
            energy_kwh_sum_mean=("energy_kwh_sum", "mean"),
            energy_kwh_sum_std=("energy_kwh_sum", "std"),
            ms_mean=("ms_mean", "mean"),
            ms_std=("ms_mean", "std"),
            ms_min=("ms_mean", "min"),
            ms_max=("ms_mean", "max"),
        )
        .sort_values("canonical")
    )
    summary["ms_cv"] = summary["ms_std"] / summary["ms_mean"].replace(0.0, pd.NA)
    return per_seed, summary


def output_path(label: str) -> Path:
    if label == "comfort_050_energy_050":
        return REPORTS / "morl_neutral_canonical_monthly_variance_diagnostic.csv"
    if label == "comfort_075_energy_025":
        return REPORTS / "morl_practical_canonical_monthly_variance_diagnostic.csv"
    return REPORTS / f"morl_{label}_monthly_variance_diagnostic.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MORL canonical monthly/yearly variance diagnostics.")
    parser.add_argument("--seedfix-root", type=Path, default=DEFAULT_SEEDFIX_ROOT)
    parser.add_argument("--labels", nargs="+", default=["comfort_050_energy_050", "comfort_075_energy_025"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    for label in args.labels:
        point_root = args.seedfix_root / label
        df = collect_rows(point_root)
        if df.empty:
            raise FileNotFoundError(f"No complete yearly summaries found under {point_root}")
        monthly = summarize_monthly(df)
        monthly.to_csv(output_path(label), index=False)
        all_rows.append(df)
        print(f"Saved monthly diagnostic for {label}: {output_path(label)}")

    combined = pd.concat(all_rows, ignore_index=True)
    per_seed, yearly = summarize_yearly(combined)
    per_seed.to_csv(REPORTS / "morl_canonical_seedfix_yearly_per_seed.csv", index=False)
    yearly.to_csv(REPORTS / "morl_canonical_seedfix_yearly_summary.csv", index=False)
    print(f"Saved yearly per-seed: {REPORTS / 'morl_canonical_seedfix_yearly_per_seed.csv'}")
    print(f"Saved yearly summary: {REPORTS / 'morl_canonical_seedfix_yearly_summary.csv'}")
    print(yearly.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
