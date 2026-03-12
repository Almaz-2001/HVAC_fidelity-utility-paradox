"""

Запускает 5 конфигураций w_comfort/w_energy * N seeds.
Результат: pareto_results.csv + графики.

Вызывается из main.py при MODE=pareto.
"""

from __future__ import annotations

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO

from envs.factory import EnvFactory
from training.train_ppo import build_ppo, train_ppo, maybe_save_model




PARETO_RUNS = [
    {"name": "comfort_only",     "w_comfort": 1.0, "w_energy": 0.0},
    {"name": "comfort_dominant", "w_comfort": 0.8, "w_energy": 0.2},
    {"name": "balanced",         "w_comfort": 0.6, "w_energy": 0.4},
    {"name": "energy_dominant",  "w_comfort": 0.4, "w_energy": 0.6},
    {"name": "energy_only",      "w_comfort": 0.0, "w_energy": 1.0},
]




def _make_env_cfg(base_cfg: dict, w_comfort: float, w_energy: float,
                  output_dir: str) -> dict:
    """Копирует env_cfg и подставляет нужные веса."""
    cfg = copy.deepcopy(base_cfg["env"])
    cfg["morl"]["w_comfort"]  = w_comfort
    cfg["morl"]["w_energy"]   = w_energy
    cfg["output_dir"]         = output_dir
    cfg["morl_csv_name"]      = "morl_log.csv"
    return cfg


def _make_train_cfg(base_cfg: dict, seed: int,
                    total_timesteps: int, output_dir: str) -> dict:
    cfg = copy.deepcopy(base_cfg["train"])
    cfg["seed"]            = seed
    cfg["output_dir"]      = output_dir
    cfg["total_timesteps"] = total_timesteps
    cfg["save_model"]      = True
    cfg["save_path"]       = os.path.join(output_dir, "models")
    cfg["morl_csv_name"]   = "morl_log.csv"
    os.makedirs(cfg["save_path"], exist_ok=True)
    return cfg




def _train_one(run_cfg: dict, seed: int, base_cfg: dict,
               total_timesteps: int, base_output: str) -> str:
    name    = run_cfg["name"]
    out_dir = os.path.join(base_output, name, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"[PARETO] {name} | seed={seed} | steps={total_timesteps}")
    print(f"  w_comfort={run_cfg['w_comfort']}, w_energy={run_cfg['w_energy']}")
    print(f"{'='*55}")

    env_cfg   = _make_env_cfg(base_cfg, run_cfg["w_comfort"],
                              run_cfg["w_energy"], out_dir)
    train_cfg = _make_train_cfg(base_cfg, seed, total_timesteps, out_dir)

    env   = EnvFactory.create(env_cfg)
    model = build_ppo(env, base_cfg.get("agent", {}))
    train_ppo(model, train_cfg)
    maybe_save_model(model, train_cfg)

    try:
        env.close()
    except Exception:
        pass

    model_path = os.path.join(out_dir, "models", "ppo_model")
    print(f"[PARETO] Saved: {model_path}.zip")
    return model_path




def _eval_one(run_cfg: dict, seed: int, base_cfg: dict,
              model_path: str, eval_steps: int,
              base_output: str) -> dict:
    name    = run_cfg["name"]
    out_dir = os.path.join(base_output, name, f"seed{seed}", "eval")
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = _make_env_cfg(base_cfg, run_cfg["w_comfort"],
                            run_cfg["w_energy"], out_dir)
    env     = EnvFactory.create(env_cfg)

    zip_path = model_path + ".zip" if not model_path.endswith(".zip") else model_path
    model    = PPO.load(zip_path, device="cpu")

    obs, _ = env.reset(seed=seed)

    comforts, energies, temps, powers, rewards = [], [], [], [], []

    for step in range(eval_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        rv = info.get("reward_vector", {}) if isinstance(info, dict) else {}
        comforts.append(float(rv.get("comfort",    0.0)))
        energies.append(float(rv.get("energy",     0.0)))
        temps.append(   float(rv.get("zone_temp",  0.0)))
        powers.append(  float(rv.get("hvac_power", 0.0)))
        rewards.append( float(reward))

        if terminated or truncated:
            obs, _ = env.reset(seed=seed)

    
    try:
        safety = env.get_safety_metric()
    except AttributeError:
        safety = {"r_time": None, "r_sev": None, "m_s": None}

    try:
        env.close()
    except Exception:
        pass

    
    eval_csv = os.path.join(out_dir, "ppo_eval.csv")
    pd.DataFrame({
        "step": range(eval_steps),
        "reward": rewards, "comfort": comforts,
        "energy": energies, "zone_temp": temps, "hvac_power": powers,
    }).to_csv(eval_csv, index=False)

    result = {
        "run_name":         name,
        "seed":             seed,
        "w_comfort":        run_cfg["w_comfort"],
        "w_energy":         run_cfg["w_energy"],
        "mean_reward":      float(np.mean(rewards)),
        "mean_comfort":     float(np.mean(comforts)),
        "mean_energy":      float(np.mean(energies)),
        "mean_temp":        float(np.mean(temps)),
        "mean_power_w":     float(np.mean(powers)),
        
        "total_energy_kwh": float(np.sum(powers)) * 3600 / 1e6,
        "r_time":           safety.get("r_time"),
        "r_sev":            safety.get("r_sev"),
        "m_s":              safety.get("m_s"),
        "violation_pct":    (safety["r_time"] * 100
                             if safety.get("r_time") is not None else None),
    }

    print(f"[EVAL] {name} seed={seed}: "
          f"comfort={result['mean_comfort']:.3f}, "
          f"energy={result['mean_energy']:.5f}, "
          f"m_s={result['m_s']:.4f}")
    return result




def _plot_pareto_front(df: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(PARETO_RUNS)))

    # --- Pareto-фронт ---
    ax = axes[0]
    for i, run in enumerate(PARETO_RUNS):
        sub = df[df["run_name"] == run["name"]]
        if sub.empty:
            continue
        x_m = sub["mean_comfort"].abs().mean()
        y_m = sub["total_energy_kwh"].mean()
        # ddof=0 чтобы не было NaN при одном seed
        x_e = sub["mean_comfort"].abs().std(ddof=0)
        y_e = sub["total_energy_kwh"].std(ddof=0)

        # убираем yerr/xerr если они NaN или ноль (один seed)
        kwargs = dict(fmt="o", color=colors[i], markersize=11, capsize=4,
                      label=f"{run['name']} (wc={run['w_comfort']})")
        if np.isfinite(x_e) and x_e > 0:
            kwargs["xerr"] = x_e
        if np.isfinite(y_e) and y_e > 0:
            kwargs["yerr"] = y_e
        ax.errorbar(x_m, y_m, **kwargs)

    
    xs = [df[df["run_name"] == r["name"]]["mean_comfort"].abs().mean()
          for r in PARETO_RUNS if not df[df["run_name"] == r["name"]].empty]
    ys = [df[df["run_name"] == r["name"]]["total_energy_kwh"].mean()
          for r in PARETO_RUNS if not df[df["run_name"] == r["name"]].empty]
    ax.plot(xs, ys, "k--", alpha=0.35, linewidth=1)

    ax.set_xlabel("|Discomfort Penalty| (↓ better)", fontsize=11)
    ax.set_ylabel("Total Energy kWh (↓ better)", fontsize=11)
    ax.set_title("Pareto Front: Comfort vs Energy", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    
    ax2 = axes[1]
    names   = [r["name"] for r in PARETO_RUNS
               if not df[df["run_name"] == r["name"]].empty]
    ms_m    = [df[df["run_name"] == n]["m_s"].mean() for n in names]
    
    ms_e    = [df[df["run_name"] == n]["m_s"].std(ddof=0) for n in names]
    valid_c = [colors[i] for i, r in enumerate(PARETO_RUNS)
               if not df[df["run_name"] == r["name"]].empty]

    # убираем yerr если все NaN или нули (один seed)
    ms_e_clean = [v if (np.isfinite(v) and v > 0) else 0.0 for v in ms_e]
    use_yerr   = any(v > 0 for v in ms_e_clean)

    bars = ax2.bar(names, ms_m,
                   yerr=ms_e_clean if use_yerr else None,
                   color=valid_c, capsize=5, alpha=0.85)

    # референсные линии Wang et al. (2024)
    ax2.axhline(0.000, color="green", ls="--", lw=1.5, label="Safe DRL (Wang 2024)")
    ax2.axhline(0.016, color="blue",  ls="--", lw=1.5, label="MPC (Wang 2024)")
    ax2.axhline(0.096, color="red",   ls="--", lw=1.5, label="PI (Wang 2024)")

    for bar, val in zip(bars, ms_m):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax2.set_ylabel("Safety Metric m_s (↓ safer)", fontsize=11)
    ax2.set_title("Safety Metric\nvs Wang et al. (2024)", fontsize=13)
    ax2.legend(fontsize=8)
    ax2.tick_params(axis="x", rotation=25)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Фаза 0: Pareto Sweep  |  energy_scale fixed (1e-7 → 2e-4)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "pareto_front.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PARETO] Plot saved: {path}")


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "="*65)
    print("PARETO SWEEP RESULTS  (mean ± std across seeds)")
    print("="*65)
    hdr = f"{'Run':22s} | {'|Comfort|':>12} | {'Energy kWh':>12} | {'m_s':>10} | {'Viol%':>7}"
    print(hdr)
    print("-" * len(hdr))
    for run in PARETO_RUNS:
        sub = df[df["run_name"] == run["name"]]
        if sub.empty:
            continue
        c_m  = sub["mean_comfort"].abs().mean();  c_s  = sub["mean_comfort"].abs().std()
        e_m  = sub["total_energy_kwh"].mean();    e_s  = sub["total_energy_kwh"].std()
        ms_m = sub["m_s"].mean();                 ms_s = sub["m_s"].std()
        vp   = sub["violation_pct"].mean()
        print(f"  {run['name']:20s} | {c_m:.3f}±{c_s:.3f}   | "
              f"{e_m:.4f}±{e_s:.4f} | {ms_m:.4f}±{ms_s:.4f} | {vp:.1f}%")

    print("\n  --- Wang et al. (2024) Peak Heat Day для сравнения ---")
    ref = [("PI", 8.381, "—", 0.096), ("MPC", 1.655, "—", 0.016),
           ("DDQN", 0.532, "—", 0.015), ("Safe DRL", 0.000, "—", 0.000)]
    for name, td, en, ms in ref:
        print(f"  {name:20s} | discomfort={td:>6.3f}        |              "
              f"| m_s={ms:.3f}    |")


def _find_best(df: pd.DataFrame) -> str:
    agg = df.groupby("run_name").agg(
        c=("mean_comfort", lambda x: x.abs().mean()),
        e=("total_energy_kwh", "mean"),
    )
    c_rng = agg["c"].max() - agg["c"].min()
    e_rng = agg["e"].max() - agg["e"].min()
    agg["score"] = (
        (agg["c"] - agg["c"].min()) / (c_rng if c_rng > 0 else 1) +
        (agg["e"] - agg["e"].min()) / (e_rng if e_rng > 0 else 1)
    )
    best = agg["score"].idxmin()
    print(f"\n[PARETO] Лучший баланс: '{best}'")
    print(f"  → Используй эти веса как дефолт в MARL (Фаза 4)")
    return best




def run_pareto_sweep(cfg: dict, seeds: list,
                     total_timesteps: int, eval_steps: int) -> None:
    base_output = "/app/outputs/pareto"
    os.makedirs(base_output, exist_ok=True)

    all_results = []

    for run_cfg in PARETO_RUNS:
        for seed in seeds:
            model_path = os.path.join(
                base_output, run_cfg["name"], f"seed{seed}", "models", "ppo_model"
            )

            
            if not os.path.exists(model_path + ".zip"):
                model_path = _train_one(
                    run_cfg, seed, cfg, total_timesteps, base_output
                )
            else:
                print(f"[PARETO] Skip training {run_cfg['name']} seed={seed} "
                      f"(model exists)")

            
            result = _eval_one(
                run_cfg, seed, cfg, model_path, eval_steps, base_output
            )
            all_results.append(result)

    df = pd.DataFrame(all_results)

    
    csv_path = os.path.join(base_output, "pareto_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[PARETO] Results saved: {csv_path}")

    
    agg_path = os.path.join(base_output, "pareto_summary.csv")
    df.groupby("run_name").agg({
        "mean_comfort":     ["mean", "std"],
        "mean_energy":      ["mean", "std"],
        "total_energy_kwh": ["mean", "std"],
        "m_s":              ["mean", "std"],
        "violation_pct":    ["mean", "std"],
    }).round(5).to_csv(agg_path)

    _print_summary(df)
    _find_best(df)
    _plot_pareto_front(df, base_output)

    print(f"\n[PARETO] Все результаты в: {base_output}")