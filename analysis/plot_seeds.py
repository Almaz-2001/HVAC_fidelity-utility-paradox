from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # numeric
    for c in ["step","reward_scalar","comfort","energy","zone_temp","hvac_power"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def list_seed_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.glob("seed*") if p.is_dir()])

def plot_boxplot_reward(means: pd.DataFrame, out_png: Path) -> None:
    # boxplot reward_mean by controller type across seeds
    # expects columns: seed, kind, reward_mean
    plt.figure()

    kinds = sorted(means["kind"].unique())
    data = [means.loc[means["kind"] == k, "reward_mean"].values for k in kinds]

    plt.boxplot(data, labels=kinds, showmeans=True)
    plt.ylabel("reward_mean")
    plt.title("Reward mean distribution across seeds (boxplot)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def plot_pareto_by_seed(summary: pd.DataFrame, out_png: Path) -> None:
    # color by seed; markers by kind
    plt.figure()

    markers = {"morl":"o", "random":"x", "zero_hold":"s"}
    seeds = sorted(summary["seed"].unique())

    for sd in seeds:
        s = summary[summary["seed"] == sd]
        for _, r in s.iterrows():
            plt.scatter(r["energy_mean"], r["comfort_mean"], marker=markers.get(r["kind"], "o"))
            plt.annotate(f"{r['seed']}-{r['kind']}", (r["energy_mean"], r["comfort_mean"]))

    plt.xlabel("energy_mean (scaled)")
    plt.ylabel("comfort_mean (penalty)")
    plt.title("Pareto cloud by seed (marker = controller)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def align_and_stack(series_list: list[pd.Series], max_len: int) -> np.ndarray:
    arr = np.full((len(series_list), max_len), np.nan, dtype=float)
    for i, s in enumerate(series_list):
        v = s.values.astype(float)
        L = min(len(v), max_len)
        arr[i, :L] = v[:L]
    return arr

def plot_mean_std_trajectory(seed_dirs: list[Path], rel_csv: str, y_col: str, title: str, out_png: Path, max_len: int = 2000) -> None:
    # mean ± std across seeds for a given CSV (morl or baseline)
    series_list = []
    for sd in seed_dirs:
        p = sd / rel_csv
        if not p.exists():
            continue
        df = read_csv(p)
        if y_col not in df.columns:
            continue
        series_list.append(df[y_col].reset_index(drop=True))

    if not series_list:
        print(f"[WARN] No series for {rel_csv} col={y_col}")
        return

    arr = align_and_stack(series_list, max_len=max_len)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)

    x = np.arange(len(mean))
    plt.figure()
    plt.plot(x, mean)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("step")
    plt.ylabel(y_col)
    plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    root = Path("/app/outputs")
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = list_seed_dirs(root)
    if not seed_dirs:
        print("No seed dirs found in /app/outputs.")
        return

    # Build summary per seed/kind
    rows = []
    for sd in seed_dirs:
        seed = sd.name
        # morl
        morl = sd / "morl_log.csv"
        if morl.exists():
            df = read_csv(morl)
            rows.append({
                "seed": seed,
                "kind": "morl",
                "reward_mean": float(df["reward_scalar"].mean()),
                "comfort_mean": float(df["comfort"].mean()),
                "energy_mean": float(df["energy"].mean()),
            })
        # baselines
        for kind in ["random", "zero_hold"]:
            f = sd / "baselines" / f"{kind}.csv"
            if f.exists():
                df = read_csv(f)
                rows.append({
                    "seed": seed,
                    "kind": kind,
                    "reward_mean": float(df["reward_scalar"].mean()),
                    "comfort_mean": float(df["comfort"].mean()),
                    "energy_mean": float(df["energy"].mean()),
                })

    summary = pd.DataFrame(rows)
    if summary.empty:
        print("Summary empty. Check files in /app/outputs/seedXX/")
        return

    summary.to_csv(out_dir / "summary_by_seed.csv", index=False)
    print("Saved:", out_dir / "summary_by_seed.csv")

    # 1) Boxplot reward_mean by kind
    plot_boxplot_reward(summary, out_dir / "boxplot_reward_mean.png")
    print("Saved:", out_dir / "boxplot_reward_mean.png")

    # 2) Pareto cloud
    plot_pareto_by_seed(summary, out_dir / "pareto_by_seed.png")
    print("Saved:", out_dir / "pareto_by_seed.png")

    # 3) Mean±std trajectories for MORL temp and power
    plot_mean_std_trajectory(seed_dirs, "morl_log.csv", "zone_temp", "MORL zone_temp mean±std across seeds", out_dir / "morl_zone_temp_mean_std.png")
    print("Saved:", out_dir / "morl_zone_temp_mean_std.png")

    plot_mean_std_trajectory(seed_dirs, "morl_log.csv", "hvac_power", "MORL hvac_power mean±std across seeds", out_dir / "morl_hvac_power_mean_std.png")
    print("Saved:", out_dir / "morl_hvac_power_mean_std.png")

    # (опционально) baseline mean±std
    plot_mean_std_trajectory(seed_dirs, "baselines/random.csv", "zone_temp", "Random baseline zone_temp mean±std across seeds", out_dir / "random_zone_temp_mean_std.png")
    plot_mean_std_trajectory(seed_dirs, "baselines/zero_hold.csv", "zone_temp", "Zero-hold baseline zone_temp mean±std across seeds", out_dir / "zero_zone_temp_mean_std.png")

    print("\nDone. Open /outputs/analysis/*.png")

if __name__ == "__main__":
    main()
