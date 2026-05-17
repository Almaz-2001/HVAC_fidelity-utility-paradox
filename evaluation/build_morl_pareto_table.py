from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARETO_ROOT = REPO_ROOT / "outputs" / "morl_pareto_hybrid_power_only"
DEFAULT_SEEDFIX_ROOT = REPO_ROOT / "outputs" / "morl_pareto_hybrid_power_only_seedfix"
DEFAULT_PI_SUMMARY = REPO_ROOT / "outputs" / "pi_baseline_15min_yearly" / "pi_yearly_summary.csv"
DEFAULT_CANONICAL_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "morl_hybrid_v3_v35_power_only_17d"
    / "seed42"
    / "yearly_eval"
    / "morl_yearly_summary.csv"
)
DEFAULT_TABLE = REPO_ROOT / "reports" / "morl_pareto_front_table.csv"
DEFAULT_LONG_TABLE = REPO_ROOT / "reports" / "morl_pareto_front_long_table.csv"
DEFAULT_FIGURE = REPO_ROOT / "reports" / "figures" / "morl_pareto_front.png"
DEFAULT_FIGURE_PDF = REPO_ROOT / "reports" / "figures" / "morl_pareto_front.pdf"
METRICS = ["rmse", "mae", "within_1c_pct", "within_05c_pct", "viol_pct", "energy_kwh", "ms"]


def canonical_designation(label: str) -> str:
    if label == "comfort_050_energy_050":
        return "pre_registered"
    if label == "comfort_075_energy_025":
        return "practical_deployment"
    if label == "legacy_canonical_080_020":
        return "legacy_reference"
    if label == "pi_yearly_builtin":
        return "baseline"
    return "pareto_point"


def summarize_yearly(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {
        "rmse_mean": float(df["rmse"].mean()),
        "mae_mean": float(df["mae"].mean()),
        "within_1c_pct_mean": float(df["within_1c_pct"].mean()),
        "within_05c_pct_mean": float(df["within_05c_pct"].mean()),
        "violation_pct_mean": float(df["viol_pct"].mean()),
        "energy_kwh_mean": float(df["energy_kwh"].mean()),
        "ms_mean": float(df["ms"].mean()),
    }


def load_weights(seed_root: Path) -> dict[str, float]:
    weights_path = seed_root / "fixed_objective_weights.json"
    if not weights_path.exists():
        return {"w_comfort": float("nan"), "w_energy": float("nan"), "w_safety": float("nan")}
    data = json.loads(weights_path.read_text(encoding="utf-8"))
    return {
        "w_comfort": float(data.get("w_comfort", float("nan"))),
        "w_energy": float(data.get("w_energy", float("nan"))),
        "w_safety": float(data.get("w_safety", float("nan"))),
    }


def parse_seed(seed_root: Path) -> int:
    text = seed_root.name.replace("seed", "")
    return int(text) if text.isdigit() else -1


def point_roots(pareto_root: Path, seedfix_root: Path) -> list[Path]:
    roots = {point_root.name: point_root for point_root in sorted(pareto_root.glob("comfort_*_energy_*"))}
    for label in ["comfort_050_energy_050", "comfort_075_energy_025"]:
        seedfix_point = seedfix_root / label
        if seedfix_point.exists():
            roots[label] = seedfix_point
    return [roots[label] for label in sorted(roots)]


def build_table(pareto_root: Path, seedfix_root: Path, pi_summary: Path, canonical_summary: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []

    for point_root in point_roots(pareto_root, seedfix_root):
        for seed_root in sorted(point_root.glob("seed*")):
            seed = parse_seed(seed_root)
            if seed < 0:
                continue
            summary_path = seed_root / "yearly_eval" / "morl_yearly_summary.csv"
            row: dict[str, float | str | bool] = {
                "kind": "morl_pareto",
                "label": point_root.name,
                "canonical_designation": canonical_designation(point_root.name),
                "seed": seed,
                "complete": summary_path.exists(),
                "source": str(summary_path.relative_to(REPO_ROOT)) if summary_path.exists() else "",
                **load_weights(seed_root),
            }
            if summary_path.exists():
                row.update(summarize_yearly(summary_path))
            rows.append(row)

    if canonical_summary.exists():
        rows.append(
            {
                "kind": "morl_reference",
                "label": "legacy_canonical_080_020",
                "canonical_designation": canonical_designation("legacy_canonical_080_020"),
                "seed": 42,
                "complete": True,
                "source": str(canonical_summary.relative_to(REPO_ROOT)),
                "w_comfort": 0.80,
                "w_energy": 0.20,
                "w_safety": 0.00,
                **summarize_yearly(canonical_summary),
            }
        )

    if pi_summary.exists():
        rows.append(
            {
                "kind": "baseline",
                "label": "pi_yearly_builtin",
                "canonical_designation": canonical_designation("pi_yearly_builtin"),
                "seed": 0,
                "complete": True,
                "source": str(pi_summary.relative_to(REPO_ROOT)),
                "w_comfort": float("nan"),
                "w_energy": float("nan"),
                "w_safety": float("nan"),
                **summarize_yearly(pi_summary),
            }
        )

    return pd.DataFrame(rows)


def add_long_rows(
    rows: list[dict[str, float | str | int]],
    *,
    summary_path: Path,
    kind: str,
    label: str,
    designation: str,
    seed: int,
    weights: dict[str, float],
) -> None:
    df = pd.read_csv(summary_path)
    for _, scenario_row in df.iterrows():
        scenario = str(scenario_row["name"])
        for metric in METRICS:
            rows.append(
                {
                    "kind": kind,
                    "label": label,
                    "seed": seed,
                    "scenario": scenario,
                    "canonical_designation": designation,
                    "preference_w_comfort": weights["w_comfort"],
                    "preference_w_energy": weights["w_energy"],
                    "preference_w_safety": weights["w_safety"],
                    "metric": metric,
                    "value": float(scenario_row[metric]),
                    "source": str(summary_path.relative_to(REPO_ROOT)),
                }
            )


def build_long_table(pareto_root: Path, seedfix_root: Path, pi_summary: Path, canonical_summary: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for point_root in point_roots(pareto_root, seedfix_root):
        for seed_root in sorted(point_root.glob("seed*")):
            seed = parse_seed(seed_root)
            if seed < 0:
                continue
            summary_path = seed_root / "yearly_eval" / "morl_yearly_summary.csv"
            if not summary_path.exists():
                continue
            add_long_rows(
                rows,
                summary_path=summary_path,
                kind="morl_pareto",
                label=point_root.name,
                designation=canonical_designation(point_root.name),
                seed=seed,
                weights=load_weights(seed_root),
            )

    if canonical_summary.exists():
        add_long_rows(
            rows,
            summary_path=canonical_summary,
            kind="morl_reference",
            label="legacy_canonical_080_020",
            designation=canonical_designation("legacy_canonical_080_020"),
            seed=42,
            weights={"w_comfort": 0.80, "w_energy": 0.20, "w_safety": 0.00},
        )

    if pi_summary.exists():
        add_long_rows(
            rows,
            summary_path=pi_summary,
            kind="baseline",
            label="pi_yearly_builtin",
            designation=canonical_designation("pi_yearly_builtin"),
            seed=0,
            weights={"w_comfort": float("nan"), "w_energy": float("nan"), "w_safety": float("nan")},
        )

    return pd.DataFrame(rows)


def plot_front(table: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    pareto = table[(table["kind"] == "morl_pareto") & (table["complete"] == True)].copy()
    pareto = (
        pareto.groupby(["label", "canonical_designation", "w_comfort", "w_energy", "w_safety"], as_index=False)
        .agg(
            energy_kwh_mean=("energy_kwh_mean", "mean"),
            energy_kwh_std=("energy_kwh_mean", "std"),
            violation_pct_mean=("violation_pct_mean", "mean"),
            violation_pct_std=("violation_pct_mean", "std"),
            ms_mean=("ms_mean", "mean"),
            seed_count=("seed", "count"),
        )
        .sort_values("w_energy")
    )
    ax.plot(
        pareto["energy_kwh_mean"],
        pareto["violation_pct_mean"],
        color="#1f77b4",
        linewidth=1.5,
        alpha=0.75,
        zorder=1,
    )
    scatter = ax.scatter(
        pareto["energy_kwh_mean"],
        pareto["violation_pct_mean"],
        c=pareto["w_energy"],
        cmap="viridis",
        s=95,
        edgecolor="black",
        linewidth=0.7,
        label="MORL preference points",
        zorder=3,
    )
    for _, row in pareto.iterrows():
        label = f"{row['w_comfort']:.2f}/{row['w_energy']:.2f}"
        ax.annotate(label, (row["energy_kwh_mean"], row["violation_pct_mean"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
        if int(row["seed_count"]) > 1:
            ax.errorbar(
                row["energy_kwh_mean"],
                row["violation_pct_mean"],
                xerr=0.0 if pd.isna(row["energy_kwh_std"]) else row["energy_kwh_std"],
                yerr=0.0 if pd.isna(row["violation_pct_std"]) else row["violation_pct_std"],
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=3,
                zorder=2,
            )

    refs = table[table["kind"].isin(["baseline", "morl_reference"])]
    for _, row in refs.iterrows():
        marker = "X" if row["kind"] == "baseline" else "D"
        color = "#d62728" if row["kind"] == "baseline" else "#ff7f0e"
        ax.scatter(row["energy_kwh_mean"], row["violation_pct_mean"], marker=marker, s=105, color=color, edgecolor="black", label=row["label"], zorder=4)

    ax.axhline(5.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8, label="5% violation threshold")
    ax.set_xlabel("Mean yearly energy (kWh)")
    ax.set_ylabel("Mean comfort violation (%)")
    ax.set_title("MORL comfort-energy Pareto sweep on hybrid backend")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Energy preference weight")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-facing MORL Pareto table and figure.")
    parser.add_argument("--pareto-root", type=Path, default=DEFAULT_PARETO_ROOT)
    parser.add_argument("--seedfix-root", type=Path, default=DEFAULT_SEEDFIX_ROOT)
    parser.add_argument("--pi-summary", type=Path, default=DEFAULT_PI_SUMMARY)
    parser.add_argument("--canonical-summary", type=Path, default=DEFAULT_CANONICAL_SUMMARY)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output-long-table", type=Path, default=DEFAULT_LONG_TABLE)
    parser.add_argument("--output-figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--output-figure-pdf", type=Path, default=DEFAULT_FIGURE_PDF)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = build_table(args.pareto_root, args.seedfix_root, args.pi_summary, args.canonical_summary)
    long_table = build_long_table(args.pareto_root, args.seedfix_root, args.pi_summary, args.canonical_summary)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.output_long_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_table, index=False)
    long_table.to_csv(args.output_long_table, index=False)
    plot_front(table, args.output_figure)
    if args.output_figure_pdf != args.output_figure.with_suffix(".pdf"):
        plot_front(table, args.output_figure_pdf)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved table: {args.output_table}")
    print(f"Saved long table: {args.output_long_table}")
    print(f"Saved figure: {args.output_figure}")
    print(f"Saved figure PDF: {args.output_figure_pdf}")


if __name__ == "__main__":
    main()
