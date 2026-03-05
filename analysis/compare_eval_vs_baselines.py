from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


REQ_EVAL = ["step", "reward_scalar", "comfort", "energy", "zone_temp", "hvac_power"]
REQ_BASE = ["step", "reward_scalar", "comfort", "energy", "zone_temp", "hvac_power"]

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def ensure_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Have: {list(df.columns)}")

def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def summarize(df: pd.DataFrame, name: str) -> dict:
    d = to_numeric(df, ["reward_scalar","comfort","energy","zone_temp","hvac_power"]).dropna()
    return {
        "name": name,
        "steps": int(len(d)),
        "reward_mean": float(d["reward_scalar"].mean()),
        "reward_std": float(d["reward_scalar"].std()),
        "comfort_mean": float(d["comfort"].mean()),
        "comfort_p95": float(np.quantile(d["comfort"].values, 0.95)),
        "energy_mean": float(d["energy"].mean()),
        "hvac_power_mean": float(d["hvac_power"].mean()),
        "zone_temp_mean": float(d["zone_temp"].mean()),
        "zone_temp_min": float(d["zone_temp"].min()),
        "zone_temp_max": float(d["zone_temp"].max()),
    }

def plot_timeseries(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    d = to_numeric(df, ["step","zone_temp","hvac_power"]).dropna()

    # Temperature
    plt.figure()
    plt.plot(d["step"].values, d["zone_temp"].values)
    plt.xlabel("step")
    plt.ylabel("zone_temp (C)")
    plt.title(f"Zone Temperature — {name}")
    plt.savefig(out_dir / f"ts_temp_{name}.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Power
    plt.figure()
    plt.plot(d["step"].values, d["hvac_power"].values)
    plt.xlabel("step")
    plt.ylabel("hvac_power (W)")
    plt.title(f"HVAC Power — {name}")
    plt.savefig(out_dir / f"ts_power_{name}.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_pareto(sum_df: pd.DataFrame, out_png: Path) -> None:
    # x: energy_mean (closer to 0 is better because it's negative penalty), y: comfort_mean (closer to 0 is better)
    plt.figure()
    plt.scatter(sum_df["energy_mean"], sum_df["comfort_mean"])
    for _, r in sum_df.iterrows():
        plt.annotate(r["name"], (r["energy_mean"], r["comfort_mean"]))
    plt.xlabel("energy_mean (penalty; closer to 0 is better)")
    plt.ylabel("comfort_mean (penalty; closer to 0 is better)")
    plt.title("Pareto cloud: Comfort vs Energy (Evaluation)")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    outputs = Path("/app/outputs")
    analysis_dir = outputs / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Required inputs
    eval_path = analysis_dir / "eval_morl.csv"

    # baselines could be either:
    # 1) /app/outputs/baselines/*.csv  (old single-run)
    # 2) /app/outputs/seed42/baselines/*.csv (multi-seed)
    # We'll try global first, else pick the first seed folder.
    global_base = outputs / "baselines"
    seed_dirs = sorted([p for p in outputs.glob("seed*") if p.is_dir()])
    if global_base.exists():
        base_dir = global_base
    elif seed_dirs and (seed_dirs[0] / "baselines").exists():
        base_dir = seed_dirs[0] / "baselines"
    else:
        base_dir = None

    if base_dir is None:
        raise FileNotFoundError("Baselines not found. Expected /app/outputs/baselines or /app/outputs/seedXX/baselines")

    rand_path = base_dir / "random.csv"
    zero_path = base_dir / "zero_hold.csv"

    df_eval = load_csv(eval_path); ensure_cols(df_eval, REQ_EVAL, "eval_morl.csv")
    df_rand = load_csv(rand_path); ensure_cols(df_rand, REQ_BASE, f"{rand_path}")
    df_zero = load_csv(zero_path); ensure_cols(df_zero, REQ_BASE, f"{zero_path}")

    # Summaries
    rows = [
        summarize(df_rand, "baseline_random"),
        summarize(df_zero, "baseline_zero_hold"),
        summarize(df_eval, "morl_eval"),
    ]
    sum_df = pd.DataFrame(rows)
    out_sum = analysis_dir / "eval_vs_baselines_summary.csv"
    sum_df.to_csv(out_sum, index=False)

    # Plots
    plot_pareto(sum_df, analysis_dir / "pareto_eval_vs_baselines.png")
    plot_timeseries(df_eval, "morl_eval", analysis_dir)
    plot_timeseries(df_rand, "baseline_random", analysis_dir)
    plot_timeseries(df_zero, "baseline_zero_hold", analysis_dir)

    print("Saved:", out_sum)
    print("Saved:", analysis_dir / "pareto_eval_vs_baselines.png")
    print("Saved time-series: ts_temp_*.png and ts_power_*.png in", analysis_dir)

if __name__ == "__main__":
    main()
