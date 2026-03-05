import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _extract_seed(path: str) -> int:
    # .../outputs/seed42/eval/ppo_eval.csv -> 42
    parts = Path(path).parts
    for p in parts:
        if p.startswith("seed") and p[4:].isdigit():
            return int(p[4:])
    return -1


def _safe_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(x.values))


def _safe_sum(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nansum(x.values))


def _pct_in_band(temp: pd.Series, low: float, high: float) -> float:
    t = pd.to_numeric(temp, errors="coerce")
    ok = (t >= low) & (t <= high)
    if ok.count() == 0:
        return float("nan")
    return float(ok.mean() * 100.0)


def load_eval(eval_csv: str) -> pd.DataFrame:
    df = pd.read_csv(eval_csv)
    # normalize column names if needed
    df.columns = [c.strip() for c in df.columns]
    return df


def summarize_seed_eval(seed: int, df: pd.DataFrame, temp_low: float, temp_high: float) -> dict:
    out = {
        "seed": seed,
        "n_steps": int(len(df)),
        "reward_mean": _safe_mean(df.get("reward_scalar", pd.Series(dtype=float))),
        "reward_sum": _safe_sum(df.get("reward_scalar", pd.Series(dtype=float))),
        "comfort_mean": _safe_mean(df.get("comfort", pd.Series(dtype=float))),
        "comfort_sum": _safe_sum(df.get("comfort", pd.Series(dtype=float))),
        "energy_mean": _safe_mean(df.get("energy", pd.Series(dtype=float))),
        "energy_sum": _safe_sum(df.get("energy", pd.Series(dtype=float))),
        "temp_mean": _safe_mean(df.get("zone_temp", pd.Series(dtype=float))),
        "temp_min": float(pd.to_numeric(df.get("zone_temp", pd.Series(dtype=float)), errors="coerce").min()),
        "temp_max": float(pd.to_numeric(df.get("zone_temp", pd.Series(dtype=float)), errors="coerce").max()),
        "hvac_power_mean": _safe_mean(df.get("hvac_power", pd.Series(dtype=float))),
        "hvac_power_max": float(pd.to_numeric(df.get("hvac_power", pd.Series(dtype=float)), errors="coerce").max()),
        "pct_in_comfort_band": _pct_in_band(df.get("zone_temp", pd.Series(dtype=float)), temp_low, temp_high),
    }
    return out


def plot_mean_curve(dfs: list[pd.DataFrame], col: str, title: str, out_path: Path) -> None:
    # align by step index
    series = []
    min_len = min(len(d) for d in dfs)
    for d in dfs:
        s = pd.to_numeric(d[col].iloc[:min_len], errors="coerce")
        series.append(s.values)

    arr = np.vstack(series)  # [n_seed, T]
    m = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)

    x = np.arange(min_len)
    plt.figure()
    plt.plot(x, m)
    plt.fill_between(x, m - sd, m + sd, alpha=0.2)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs", help="Path to outputs folder")
    ap.add_argument("--temp_low", type=float, default=21.0, help="Comfort band low (°C)")
    ap.add_argument("--temp_high", type=float, default=25.0, help="Comfort band high (°C)")
    ap.add_argument("--include_baselines", action="store_true", help="Also summarize baselines if present")
    args = ap.parse_args()

    outputs = Path(args.outputs)
    eval_paths = sorted(glob.glob(str(outputs / "seed*" / "eval" / "ppo_eval.csv")))

    if not eval_paths:
        raise SystemExit(f"No eval files found at: {outputs}/seed*/eval/ppo_eval.csv")

    rows = []
    dfs_for_plots = []

    for p in eval_paths:
        seed = _extract_seed(p)
        df = load_eval(p)
        dfs_for_plots.append(df)
        rows.append(summarize_seed_eval(seed, df, args.temp_low, args.temp_high))

    per_seed = pd.DataFrame(rows).sort_values("seed")
    out_dir = outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_path = out_dir / "summary_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    # mean ± std across seeds (numeric columns only)
    numeric = per_seed.select_dtypes(include=[np.number])
    mean_row = numeric.mean(numeric_only=True)
    std_row = numeric.std(numeric_only=True)

    summary = pd.DataFrame({
        "metric": mean_row.index,
        "mean": mean_row.values,
        "std": std_row.reindex(mean_row.index).values,
    })
    summary_path = out_dir / "summary_mean_std.csv"
    summary.to_csv(summary_path, index=False)

    # plots
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if all("reward_scalar" in d.columns for d in dfs_for_plots):
        plot_mean_curve(dfs_for_plots, "reward_scalar", "PPO eval: reward_scalar (mean ± std across seeds)", plots_dir / "ppo_reward_mean_std.png")
    if all("comfort" in d.columns for d in dfs_for_plots):
        plot_mean_curve(dfs_for_plots, "comfort", "PPO eval: comfort (mean ± std across seeds)", plots_dir / "ppo_comfort_mean_std.png")
    if all("energy" in d.columns for d in dfs_for_plots):
        plot_mean_curve(dfs_for_plots, "energy", "PPO eval: energy (mean ± std across seeds)", plots_dir / "ppo_energy_mean_std.png")

    # optional: baselines
    if args.include_baselines:
        base_paths = sorted(glob.glob(str(outputs / "seed*" / "baselines" / "*.csv")))
        base_rows = []
        for bp in base_paths:
            seed = _extract_seed(bp)
            name = Path(bp).stem  # random / zero_hold
            bdf = pd.read_csv(bp)
            # expect columns like: step,reward_scalar,comfort,energy,zone_temp,hvac_power,...
            base_rows.append({
                "seed": seed,
                "baseline": name,
                "n_steps": int(len(bdf)),
                "reward_mean": _safe_mean(bdf.get("reward_scalar", pd.Series(dtype=float))),
                "reward_sum": _safe_sum(bdf.get("reward_scalar", pd.Series(dtype=float))),
                "comfort_mean": _safe_mean(bdf.get("comfort", pd.Series(dtype=float))),
                "energy_mean": _safe_mean(bdf.get("energy", pd.Series(dtype=float))),
                "temp_mean": _safe_mean(bdf.get("zone_temp", pd.Series(dtype=float))),
                "pct_in_comfort_band": _pct_in_band(bdf.get("zone_temp", pd.Series(dtype=float)), args.temp_low, args.temp_high),
            })
        if base_rows:
            baselines_df = pd.DataFrame(base_rows).sort_values(["baseline", "seed"])
            baselines_df.to_csv(out_dir / "summary_baselines_per_seed.csv", index=False)

    print("Saved:", per_seed_path)
    print("Saved:", summary_path)
    print("Saved plots in:", plots_dir)


if __name__ == "__main__":
    main()