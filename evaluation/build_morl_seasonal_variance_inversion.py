from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures" / "article_real"

PRACTICAL_CSV = REPORTS / "morl_practical_canonical_monthly_variance_diagnostic.csv"
NEUTRAL_CSV = REPORTS / "morl_neutral_canonical_monthly_variance_diagnostic.csv"
OUT_CSV = REPORTS / "morl_seasonal_variance_inversion_table.csv"
OUT_PNG = FIGURES / "block2_morl_seasonal_variance_inversion.png"
OUT_PDF = FIGURES / "block2_morl_seasonal_variance_inversion.pdf"

MONTH_ORDER = [
    "Jan_Winter",
    "Feb_Winter",
    "Mar_Spring",
    "Apr_Spring",
    "May_Spring",
    "Jun_Summer",
    "Jul_Summer",
    "Aug_Summer",
    "Sep_Autumn",
    "Oct_Autumn",
    "Nov_Autumn",
    "Dec_Winter",
]


def _load_monthly(path: Path, row_label: str, n_seeds: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"scenario", "ms_mean", "ms_std"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    out = df.loc[:, ["scenario", "ms_mean", "ms_std"]].copy()
    out["display_label"] = row_label
    out["n_seeds"] = n_seeds
    return out


def build_table() -> pd.DataFrame:
    practical = _load_monthly(
        PRACTICAL_CSV,
        "Practical w=(0.75, 0.25), N=3",
        3,
    )
    neutral = _load_monthly(
        NEUTRAL_CSV,
        "Neutral w=(0.50, 0.50), N=5",
        5,
    )
    combined = pd.concat([practical, neutral], ignore_index=True)
    combined["month_index"] = combined["scenario"].map(
        {scenario: idx for idx, scenario in enumerate(MONTH_ORDER)}
    )
    if combined["month_index"].isna().any():
        unknown = sorted(combined.loc[combined["month_index"].isna(), "scenario"].unique())
        raise ValueError(f"Unexpected scenarios in monthly diagnostics: {unknown}")
    combined = combined.sort_values(["display_label", "month_index"]).reset_index(drop=True)

    pivot = combined.pivot(index="scenario", columns="display_label", values="ms_std")
    if {
        "Neutral w=(0.50, 0.50), N=5",
        "Practical w=(0.75, 0.25), N=3",
    }.issubset(pivot.columns):
        ratio = (
            pivot["Neutral w=(0.50, 0.50), N=5"]
            / pivot["Practical w=(0.75, 0.25), N=3"].replace(0.0, np.nan)
        )
        combined["neutral_over_practical_sigma_ratio"] = combined["scenario"].map(ratio)
    return combined


def plot_heatmap(table: pd.DataFrame) -> None:
    row_order = [
        "Practical w=(0.75, 0.25), N=3",
        "Neutral w=(0.50, 0.50), N=5",
    ]
    heat = (
        table.pivot(index="display_label", columns="scenario", values="ms_std")
        .loc[row_order, MONTH_ORDER]
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.5, 3.7), constrained_layout=True)
    vmax = max(0.16, float(np.nanmax(heat.to_numpy())))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)

    ax.set_xticks(np.arange(len(MONTH_ORDER)))
    ax.set_xticklabels([name.split("_")[0] for name in MONTH_ORDER], fontsize=10)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=10)
    ax.set_title(
        "MORL seasonal variance inversion across preference weights",
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("Yearly validation scenario")

    for y in range(heat.shape[0]):
        for x in range(heat.shape[1]):
            value = float(heat.iloc[y, x])
            color = "white" if value > 0.09 else "#222222"
            ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=8.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Across-seed std of monthly $m_s$", rotation=90)

    subtitle = (
        "Cells are computed from monthly yearly-evaluation CSVs. "
        "Practical N=3 will be refreshed after seed45/seed46."
    )
    fig.text(0.5, -0.06, subtitle, ha="center", fontsize=9)

    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    table = build_table()
    table.to_csv(OUT_CSV, index=False)
    plot_heatmap(table)
    print(f"Saved table: {OUT_CSV}")
    print(f"Saved figure: {OUT_PNG}")
    print(f"Saved figure: {OUT_PDF}")


if __name__ == "__main__":
    main()
