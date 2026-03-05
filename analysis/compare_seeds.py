from __future__ import annotations

import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


REQUIRED_COLS = [
    "step",
    "reward_scalar",
    "comfort",
    "energy",
    "zone_temp",
    "hvac_power",
]

def read_csv_safe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize col names (just in case)
    df.columns = [c.strip() for c in df.columns]
    return df

def ensure_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Have: {list(df.columns)}")

def p95_abs_comfort(comfort: pd.Series) -> float:
    # comfort is negative penalty already; p95 of |comfort| is OK too
    return float(np.quantile(comfort.astype(float).values, 0.95))

def summarize_run(df: pd.DataFrame, name: str) -> dict:
    # drop NaNs in numeric columns
    dfn = df.copy()
    for c in REQUIRED_COLS:
        if c in dfn.columns:
            dfn[c] = pd.to_numeric(dfn[c], errors="coerce")
    dfn = dfn.dropna(subset=["comfort", "energy", "zone_temp", "hvac_power"])

    out = {
        "name": name,
        "steps": int(dfn.shape[0]),
        "reward_mean": float(np.nanmean(pd.to_numeric(dfn["reward_scalar"], errors="coerce"))),
        "comfort_mean": float(np.nanmean(dfn["comfort"])),
        "comfort_p95": p95_abs_comfort(dfn["comfort"]),
        "energy_mean": float(np.nanmean(dfn["energy"])),
        "hvac_power_mean": float(np.nanmean(dfn["hvac_power"])),
        "zone_temp_mean": float(np.nanmean(dfn["zone_temp"])),
        "zone_temp_min": float(np.nanmin(dfn["zone_temp"])),
        "zone_temp_max": float(np.nanmax(dfn["zone_temp"])),
    }
    return out

def save_pareto(df_sum: pd.DataFrame, out_png: str) -> None:
    # Pareto-like scatter: x = energy_mean (positive = worse) , y = comfort_mean (more negative = worse)
    plt.figure()
    plt.scatter(df_sum["energy_mean"], df_sum["comfort_mean"])
    for _, r in df_sum.iterrows():
        plt.annotate(r["name"], (r["energy_mean"], r["comfort_mean"]))
    plt.xlabel("energy_mean (scaled)")
    plt.ylabel("comfort_mean (penalty)")
    plt.title("Pareto cloud (lower is better in both axes)")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    outputs_root = Path("/app/outputs")
    seed_dirs = sorted([p for p in outputs_root.glob("seed*") if p.is_dir()])

    if not seed_dirs:
        print("No seed folders found in /app/outputs. Expected /app/outputs/seed42, seed43, ...")
        return

    all_rows = []

    for sd in seed_dirs:
        seed_name = sd.name  # seed42
        morl_path = sd / "morl_log.csv"
        if not morl_path.exists():
            print(f"[SKIP] {seed_name}: morl_log.csv not found")
            continue

        df_morl = read_csv_safe(str(morl_path))
        ensure_cols(df_morl, REQUIRED_COLS, f"{seed_name}/morl_log.csv")
        all_rows.append(summarize_run(df_morl, f"{seed_name}_morl"))

        # baselines
        base_dir = sd / "baselines"
        if base_dir.exists():
            for bname in ["random", "zero_hold"]:
                f = base_dir / f"{bname}.csv"
                if f.exists():
                    dfb = read_csv_safe(str(f))
                    # baseline CSV might have more cols; but must include required ones
                    ensure_cols(dfb, REQUIRED_COLS, f"{seed_name}/baselines/{bname}.csv")
                    all_rows.append(summarize_run(dfb, f"{seed_name}_{bname}"))
                else:
                    print(f"[WARN] {seed_name}: baseline file missing: {f}")
        else:
            print(f"[WARN] {seed_name}: baselines folder not found")

    df_sum = pd.DataFrame(all_rows)
    out_dir = outputs_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "all_seeds_summary.csv"
    df_sum.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # Overall pareto cloud
    out_pareto = out_dir / "pareto_all_seeds.png"
    save_pareto(df_sum, str(out_pareto))
    print("Saved:", out_pareto)

    print("\nDone. Next: run plot_seeds.py for boxplots and mean±std.")
    

if __name__ == "__main__":
    main()
