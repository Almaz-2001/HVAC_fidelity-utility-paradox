from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = REPO_ROOT / "reports" / "block2_morl_comparison_summary.csv"
DEFAULT_RERUN = (
    REPO_ROOT
    / "outputs"
    / "morl_5d_legacy_rerun"
    / "comfort_075_energy_025"
    / "seed42"
    / "yearly_eval"
    / "morl_yearly_summary.csv"
)
DEFAULT_OUT = REPO_ROOT / "reports" / "block2_morl_5d_reconstructed_comparison.csv"


def summarize_yearly(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {
        "rmse_c": float(df["rmse"].mean()),
        "mae_c": float(df["mae"].mean()),
        "within_1c_pct": float(df["within_1c_pct"].mean()),
        "within_05c_pct": float(df["within_05c_pct"].mean()),
        "violation_pct": float(df["viol_pct"].mean()),
        "energy_kwh": float(df["energy_kwh"].mean()),
        "m_s": float(df["ms"].mean()),
    }


def build_table(frozen_csv: Path, rerun_summary: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool]] = []

    frozen = pd.read_csv(frozen_csv)
    for _, row in frozen.iterrows():
        rows.append(
            {
                "evidence_layer": "historical_frozen",
                "variant": str(row["variant"]),
                "obs_path": str(row["obs_path"]),
                "backend": str(row["backend"]),
                "seed": "historical",
                "reproducibility_status": "frozen_csv_preserved",
                "source": str(frozen_csv.relative_to(REPO_ROOT)),
                "complete": True,
                "rmse_c": float(row["rmse_c"]),
                "mae_c": float(row["mae_c"]),
                "within_1c_pct": float(row["within_1c_pct"]),
                "within_05c_pct": float(row["within_05c_pct"]),
                "violation_pct": float(row["violation_pct"]),
                "energy_kwh": float(row["energy_kwh"]),
                "m_s": float(row["m_s"]),
            }
        )

    if rerun_summary.exists():
        rows.append(
            {
                "evidence_layer": "current_reconstructed_rerun",
                "variant": "MORL_5D_basic_reconstructed",
                "obs_path": "basic_5d",
                "backend": "hybrid_v3_v35",
                "seed": "42",
                "reproducibility_status": "current_code_rerun",
                "source": str(rerun_summary.relative_to(REPO_ROOT)),
                "complete": True,
                **summarize_yearly(rerun_summary),
            }
        )
    else:
        rows.append(
            {
                "evidence_layer": "current_reconstructed_rerun",
                "variant": "MORL_5D_basic_reconstructed",
                "obs_path": "basic_5d",
                "backend": "hybrid_v3_v35",
                "seed": "42",
                "reproducibility_status": "pending_run",
                "source": str(rerun_summary.relative_to(REPO_ROOT)),
                "complete": False,
                "rmse_c": float("nan"),
                "mae_c": float("nan"),
                "within_1c_pct": float("nan"),
                "within_05c_pct": float("nan"),
                "violation_pct": float("nan"),
                "energy_kwh": float("nan"),
                "m_s": float("nan"),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the frozen historical MORL 5D artifact with the current reconstructed 5D rerun."
    )
    parser.add_argument("--frozen-csv", default=str(DEFAULT_FROZEN))
    parser.add_argument("--rerun-summary", default=str(DEFAULT_RERUN))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    frozen_csv = Path(args.frozen_csv)
    rerun_summary = Path(args.rerun_summary)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    table = build_table(frozen_csv, rerun_summary)
    table.to_csv(output, index=False)

    print(f"Saved comparison: {output}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
