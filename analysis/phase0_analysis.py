from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


PARETO_ORDER = [
    "comfort_only", "comfort_dominant", "balanced",
    "energy_dominant", "energy_only",
]

WANG_REF = {
    "PI":       {"m_s": 0.096, "discomfort": 8.381, "cost": 0.909},
    "MPC":      {"m_s": 0.016, "discomfort": 1.655, "cost": 0.976},
    "DDQN":     {"m_s": 0.015, "discomfort": 0.532, "cost": 1.007},
    "Safe DRL": {"m_s": 0.000, "discomfort": 0.000, "cost": 0.953},
}


def _std(series) -> float:
    """std с ddof=0 — никогда не даёт NaN при одном элементе."""
    return float(series.std(ddof=0)) if len(series) > 0 else 0.0


def _safe_err(val: float) -> float:
    """Возвращает 0 если val NaN или не конечный."""
    return float(val) if (val is not None and np.isfinite(val) and val > 0) else 0.0


def load(input_dir: str) -> pd.DataFrame:
    path = os.path.join(input_dir, "pareto_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"pareto_results.csv не найден в {input_dir}\n"
            "Сначала запусти: MODE=pareto python main.py"
        )
    return pd.read_csv(path)


# -----------------------------------------------------------------------
# Диагностика scale dominance
# -----------------------------------------------------------------------

def check_scale_dominance(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("ДИАГНОСТИКА: Scale Dominance")
    print("="*60)
    print(f"  {'Run':22s} | {'|comfort|':>10} | {'|energy|':>10} | {'ratio':>8} | status")
    print("  " + "-"*65)
    for name in PARETO_ORDER:
        sub = df[df["run_name"] == name]
        if sub.empty:
            continue
        c = sub["mean_comfort"].abs().mean()
        e = sub["mean_energy"].abs().mean()
        ratio = c / e if e > 1e-12 else float("inf")
        ok = "✓ OK" if ratio < 100 else "⚠ DOMINANCE"
        print(f"  {name:22s} | {c:10.4f} | {e:10.6f} | {ratio:8.1f} | {ok}")
    print("\n  Норма: ratio < 100")
    print("  До фикса (energy_scale=1e-7):  ratio ~ 100 000 → energy игнорировался")
    print("  После фикса (energy_scale=2e-4): ratio ~ 1-10 → реальный баланс")


# -----------------------------------------------------------------------
# Сводная таблица
# -----------------------------------------------------------------------

def print_table(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ PARETO SWEEP")
    n_seeds = df["seed"].nunique()
    print(f"  seeds: {sorted(df['seed'].unique().tolist())}  (n={n_seeds})")
    print("="*60)
    hdr = f"  {'Run':22s} | {'|Comfort|':>13} | {'Energy kWh':>13} | {'m_s':>12} | {'Viol%':>6}"
    print(hdr)
    print("  " + "-"*(len(hdr)-2))
    for name in PARETO_ORDER:
        sub = df[df["run_name"] == name]
        if sub.empty:
            continue
        c_m  = sub["mean_comfort"].abs().mean()
        c_s  = _std(sub["mean_comfort"].abs())
        e_m  = sub["total_energy_kwh"].mean()
        e_s  = _std(sub["total_energy_kwh"])
        ms_m = sub["m_s"].mean()
        ms_s = _std(sub["m_s"])
        vp   = sub["violation_pct"].mean()
        # показываем ± только если несколько seeds
        if n_seeds > 1:
            print(f"  {name:22s} | {c_m:.3f} ± {c_s:.3f}  | "
                  f"{e_m:.4f} ± {e_s:.4f} | {ms_m:.4f} ± {ms_s:.4f} | {vp:.1f}%")
        else:
            print(f"  {name:22s} | {c_m:.3f}           | "
                  f"{e_m:.4f}          | {ms_m:.4f}          | {vp:.1f}%")

    print("\n  --- Wang et al. (2024) для сравнения ---")
    for algo, v in WANG_REF.items():
        print(f"  {algo:22s} | discomfort={v['discomfort']:>6.3f}        |"
              f"               | m_s={v['m_s']:.3f}       |")


# -----------------------------------------------------------------------
# Лучшая конфигурация
# -----------------------------------------------------------------------

def find_best(df: pd.DataFrame) -> str:
    agg = df.groupby("run_name").agg(
        c=("mean_comfort",     lambda x: x.abs().mean()),
        e=("total_energy_kwh", "mean"),
        ms=("m_s",             "mean"),
    )
    c_rng = agg["c"].max() - agg["c"].min()
    e_rng = agg["e"].max() - agg["e"].min()
    agg["score"] = (
        (agg["c"] - agg["c"].min()) / (c_rng if c_rng > 1e-9 else 1.0) +
        (agg["e"] - agg["e"].min()) / (e_rng if e_rng > 1e-9 else 1.0)
    )
    best = agg["score"].idxmin()
    wc   = df[df["run_name"] == best]["w_comfort"].iloc[0]
    we   = df[df["run_name"] == best]["w_energy"].iloc[0]

    print("\n" + "="*60)
    print("РЕКОМЕНДАЦИЯ для MARL (Фаза 4)")
    print("="*60)
    print(f"  Лучший баланс: '{best}'")
    print(f"  w_comfort={wc}, w_energy={we}")
    print(f"  → Используй эти веса как дефолт в env.yaml при переходе в MARL")
    print("\n  Все конфигурации по balance_score:")
    print(agg[["c", "e", "ms", "score"]].sort_values("score").to_string())
    return best


# -----------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------

def plot_dashboard(df: pd.DataFrame, output_dir: str) -> None:
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(PARETO_ORDER)))
    valid  = [n for n in PARETO_ORDER if not df[df["run_name"] == n].empty]
    vc     = [colors[i] for i, n in enumerate(PARETO_ORDER) if n in valid]

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.33)

    labels = {
        "comfort_only":     "wc=1.0",
        "comfort_dominant": "wc=0.8",
        "balanced":         "wc=0.6",
        "energy_dominant":  "wc=0.4",
        "energy_only":      "wc=0.0",
    }

    # ---- 1: Pareto-фронт ----
    ax1 = fig.add_subplot(gs[0, 0])
    xs, ys = [], []
    for i, name in enumerate(valid):
        sub = df[df["run_name"] == name]
        x_m = sub["mean_comfort"].abs().mean()
        y_m = sub["total_energy_kwh"].mean()
        x_e = _safe_err(_std(sub["mean_comfort"].abs()))
        y_e = _safe_err(_std(sub["total_energy_kwh"]))

        kwargs = dict(fmt="o", color=vc[i], markersize=11, capsize=4,
                      label=labels.get(name, name))
        if x_e > 0:
            kwargs["xerr"] = x_e
        if y_e > 0:
            kwargs["yerr"] = y_e
        ax1.errorbar(x_m, y_m, **kwargs)
        xs.append(x_m)
        ys.append(y_m)

    ax1.plot(xs, ys, "k--", alpha=0.3, lw=1)
    ax1.set_xlabel("|Discomfort Penalty| (↓ better)", fontsize=10)
    ax1.set_ylabel("Total Energy kWh (↓ better)", fontsize=10)
    ax1.set_title("Pareto Front", fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- 2: Safety metric + Wang et al. ----
    ax2 = fig.add_subplot(gs[0, 1])
    ms_m     = [df[df["run_name"] == n]["m_s"].mean() for n in valid]
    ms_e_raw = [_safe_err(_std(df[df["run_name"] == n]["m_s"])) for n in valid]
    use_yerr = any(v > 0 for v in ms_e_raw)

    bars = ax2.bar(
        [labels.get(n, n) for n in valid], ms_m,
        yerr=ms_e_raw if use_yerr else None,
        color=vc, capsize=5, alpha=0.85,
    )
    ax2.axhline(0.000, color="green", ls="--", lw=1.5, label="Safe DRL")
    ax2.axhline(0.016, color="blue",  ls="--", lw=1.5, label="MPC")
    ax2.axhline(0.096, color="red",   ls="--", lw=1.5, label="PI")
    for bar, val in zip(bars, ms_m):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("m_s (↓ safer)", fontsize=10)
    ax2.set_title("Safety Metric vs Wang et al.", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.tick_params(axis="x", rotation=20)

    # ---- 3: r_time и r_sev ----
    ax3   = fig.add_subplot(gs[1, 0])
    x_pos = np.arange(len(valid))
    w     = 0.35
    rt    = [df[df["run_name"] == n]["r_time"].mean() for n in valid]
    rs    = [df[df["run_name"] == n]["r_sev"].mean()  for n in valid]
    ax3.bar(x_pos - w/2, rt, w, label="r_time (violation ratio)",
            color="coral",     alpha=0.85)
    ax3.bar(x_pos + w/2, rs, w, label="r_sev  (severity)",
            color="steelblue", alpha=0.85)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([labels.get(n, n) for n in valid], rotation=20, fontsize=8)
    ax3.set_ylabel("Value")
    ax3.set_title("Компоненты m_s", fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    # ---- 4: Температурное распределение ----
    ax4 = fig.add_subplot(gs[1, 1])
    t_m = [df[df["run_name"] == n]["mean_temp"].mean() for n in valid]
    t_e = [_safe_err(_std(df[df["run_name"] == n]["mean_temp"])) for n in valid]

    # errorbar только если есть реальный std
    if any(v > 0 for v in t_e):
        ax4.errorbar(range(len(valid)), t_m, yerr=t_e, fmt="s-",
                     color="darkorange", markersize=10, capsize=5, lw=2)
    else:
        ax4.plot(range(len(valid)), t_m, "s-",
                 color="darkorange", markersize=10, lw=2)

    ax4.axhspan(21, 25, alpha=0.12, color="green")
    ax4.axhline(21, color="green", ls="--", alpha=0.5, label="Comfort [21-25°C]")
    ax4.axhline(25, color="green", ls="--", alpha=0.5)
    ax4.set_xticks(range(len(valid)))
    ax4.set_xticklabels([labels.get(n, n) for n in valid], rotation=20, fontsize=8)
    ax4.set_ylabel("Mean Zone Temp (°C)", fontsize=10)
    ax4.set_title("Средняя температура зоны", fontsize=12)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(
        "Фаза 0: MORL-PPO Pareto Sweep\n"
        "Фикс: energy_scale 1e-7 → 2e-4  |  obs нормализован в [-1, 1]",
        fontsize=12, fontweight="bold", y=1.01,
    )

    path = os.path.join(output_dir, "phase0_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ANALYSIS] Dashboard: {path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/app/outputs/pareto")
    args = parser.parse_args()

    df = load(args.input)

    check_scale_dominance(df)
    print_table(df)
    find_best(df)
    plot_dashboard(df, args.input)

    print(f"\n[ANALYSIS] Готово → {args.input}/phase0_dashboard.png")


if __name__ == "__main__":
    main()