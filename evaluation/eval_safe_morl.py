

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from stable_baselines3 import PPO
from configs.loader import load_all_configs
from envs.factory import EnvFactory


def eval_safe_morl(
    model_path: str,
    config_dir: str = "configs",
    surrogate_path: Optional[str] = None,
    surrogate_kind: str = "legacy_v3",
    surrogate_summary_json: Optional[str] = None,
    surrogate_checkpoint: Optional[str] = None,
    surrogate_base_model: Optional[str] = None,
    out_dir: str = "/app/outputs/eval_safe_morl",
    n_steps: int = 5000,
    seed: int = 42,
    use_safety: bool = True,
    horizon: int = 2,
    margin: float = 0.82,
    t_low: float = 21.0,
    t_high: float = 25.0,
    warmup_steps: int = 300,
) -> Dict[str, float]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cfg = load_all_configs(config_dir)
    env_cfg = dict(cfg["env"])
    if surrogate_kind:
        env_cfg["surrogate_kind"] = surrogate_kind
    if surrogate_path:
        env_cfg["surrogate_path"] = surrogate_path
    if surrogate_summary_json:
        env_cfg["surrogate_summary_json"] = surrogate_summary_json
    if surrogate_checkpoint:
        env_cfg["surrogate_checkpoint"] = surrogate_checkpoint
    if surrogate_base_model:
        env_cfg["surrogate_base_model"] = surrogate_base_model
    env = EnvFactory.create(env_cfg)

    model = PPO.load(model_path, device="cpu")
    print(f"[EVAL] PPO model: {model_path}")
    if model.observation_space.shape != env.observation_space.shape:
        raise RuntimeError(
            f"Model obs shape {model.observation_space.shape} does not match env "
            f"obs shape {env.observation_space.shape}. Retrain or pick a model "
            "for the direct-TSup pipeline."
        )

    safety_filter = None
    if use_safety and surrogate_path:
        from layers.safety.action_filter import SurrogateSafetyFilter
        safety_filter = SurrogateSafetyFilter(
            model_path=surrogate_path,
            surrogate_kind=surrogate_kind,
            surrogate_summary_json=surrogate_summary_json,
            surrogate_checkpoint=surrogate_checkpoint,
            surrogate_base_model=surrogate_base_model,
            horizon=horizon,
            t_low=t_low,
            t_high=t_high,
            margin=margin,
        )
        print(f"[EVAL] Safety filter: ON (horizon={horizon}, margin={margin})")
    else:
        print(f"[EVAL] Safety filter: OFF")

    print(f"[EVAL] Warmup exclusion: {warmup_steps} steps")

    obs, info = env.reset(seed=seed)
    rows = []

    
    total_steps = 0
    # Post-warmup counters 
    pw_violation_steps = 0
    pw_max_overshoot = 0.0
    pw_max_undershoot = 0.0
    # Full counters 
    full_violation_steps = 0
    full_max_overshoot = 0.0
    full_max_undershoot = 0.0

    total_energy = 0.0
    total_comfort_penalty = 0.0
    ppo_accepted = 0
    ppo_rejected = 0

    print(f"[EVAL] Running {n_steps} steps...")

    for step in range(n_steps):
        action_ppo, _ = model.predict(obs, deterministic=True)

        if safety_filter is not None:
            rv = info.get("reward_vector", {}) if isinstance(info, dict) else {}
            state = {
                't_zone': rv.get('zone_temp', 22.0),
                't_amb': info.get('t_amb', 10.0) if isinstance(info, dict) else 10.0,
                'hour': info.get('hour', 12.0) if isinstance(info, dict) else 12.0,
                'day': info.get('day', 180.0) if isinstance(info, dict) else 180.0,
            }
            action, sf_info = safety_filter.filter(action_ppo, state)
            if sf_info['safe']:
                ppo_accepted += 1
            else:
                ppo_rejected += 1
        else:
            action = action_ppo
            sf_info = {'safe': True, 'source': 'ppo'}

        obs, reward, terminated, truncated, info = env.step(action)

        rv = info.get("reward_vector", {}) if isinstance(info, dict) else {}
        t_zone = rv.get("zone_temp", 22.0)
        p_total = rv.get("hvac_power", 0.0)
        comfort = rv.get("comfort", 0.0)
        t_amb = info.get("t_amb", 10.0) if isinstance(info, dict) else 10.0

        total_steps += 1
        total_energy += p_total

        # Full tracking 
        is_violation = (t_zone > t_high) or (t_zone < t_low)
        if is_violation:
            full_violation_steps += 1
        if t_zone > t_high:
            full_max_overshoot = max(full_max_overshoot, (t_zone - t_high) / t_high)
        elif t_zone < t_low:
            full_max_undershoot = max(full_max_undershoot, (t_low - t_zone) / t_low)

        # Post-warmup tracking 
        if step >= warmup_steps:
            total_comfort_penalty += abs(comfort)
            if t_zone > t_high:
                pw_violation_steps += 1
                pw_max_overshoot = max(pw_max_overshoot, (t_zone - t_high) / t_high)
            elif t_zone < t_low:
                pw_violation_steps += 1
                pw_max_undershoot = max(pw_max_undershoot, (t_low - t_zone) / t_low)

        in_warmup = "WU" if step < warmup_steps else ""
        rows.append({
            "step": step,
            "t_zone": round(t_zone, 3),
            "t_amb": round(t_amb, 2),
            "p_total": round(p_total, 2),
            "comfort": round(comfort, 4),
            "reward": round(float(reward), 5),
            "a0": round(float(action[0]), 4),
            "a1": round(float(action[1]), 4) if len(action) > 1 else 0.0,
            "a0_ppo": round(float(action_ppo[0]), 4),
            "a1_ppo": round(float(action_ppo[1]), 4) if len(action_ppo) > 1 else 0.0,
            "safe": sf_info['safe'],
            "source": sf_info.get('source', 'ppo'),
            "warmup": step < warmup_steps,
        })

        if step % 500 == 0:
            eff = max(step - warmup_steps + 1, 1) if step >= warmup_steps else 1
            pw_rt = pw_violation_steps / eff if step >= warmup_steps else 0
            pw_rs = max(pw_max_overshoot, pw_max_undershoot)
            pw_ms = pw_rt + pw_rs
            pw_viol = pw_violation_steps / eff * 100 if step >= warmup_steps else 0
            accept_pct = ppo_accepted / max(ppo_accepted + ppo_rejected, 1) * 100
            print(f"  step={step}/{n_steps} T={t_zone:.1f}C "
                  f"m_s={pw_ms:.3f} viol={pw_viol:.1f}% "
                  f"accept={accept_pct:.0f}% {in_warmup}")

        if terminated or truncated:
            obs, info = env.reset(seed=seed + step)

    try:
        env.close()
    except Exception:
        pass

    #Final metricsPOST-WARMUP 
    effective_steps = total_steps - warmup_steps
    pw_r_time = pw_violation_steps / max(effective_steps, 1)
    pw_r_sev = max(pw_max_overshoot, pw_max_undershoot)
    pw_m_s = pw_r_time + pw_r_sev
    pw_viol_pct = pw_violation_steps / max(effective_steps, 1) * 100

    # Full metrics
    full_r_time = full_violation_steps / total_steps
    full_r_sev = max(full_max_overshoot, full_max_undershoot)
    full_m_s = full_r_time + full_r_sev
    full_viol_pct = full_violation_steps / total_steps * 100

    energy_kwh = total_energy / 1000.0
    accept_rate = ppo_accepted / max(ppo_accepted + ppo_rejected, 1) * 100

    results = {
        "m_s": pw_m_s,
        "r_time": pw_r_time,
        "r_sev": pw_r_sev,
        "violation_pct": pw_viol_pct,
        "violation_steps": pw_violation_steps,
        "effective_steps": effective_steps,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "energy_kwh": energy_kwh,
        "mean_comfort_penalty": total_comfort_penalty / max(effective_steps, 1),
        "ppo_accepted": ppo_accepted,
        "ppo_rejected": ppo_rejected,
        "acceptance_rate": accept_rate,
        # Full metrics for reference
        "full_m_s": full_m_s,
        "full_violation_pct": full_viol_pct,
    }

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "eval_safe_morl.csv")
    df.to_csv(csv_path, index=False)
    pd.DataFrame([results]).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Safety filter:     {'ON' if use_safety else 'OFF'}")
    print(f"  Total steps:       {total_steps}")
    print(f"  Warmup excluded:   {warmup_steps} steps")
    print(f"  Effective steps:   {effective_steps}")
    print(f"")
    print(f"  SAFETY METRIC (post-warmup, Wang et al.):")
    print(f"    r_time:          {pw_r_time:.4f}")
    print(f"    r_sev:           {pw_r_sev:.4f}")
    print(f"    m_s:             {pw_m_s:.4f}")
    print(f"    Violation %:     {pw_viol_pct:.1f}%")
    print(f"")
    print(f"  FULL METRICS (incl. warmup, for reference):")
    print(f"    m_s (full):      {full_m_s:.4f}")
    print(f"    Violation (full):{full_viol_pct:.1f}%")
    print(f"")
    print(f"  ENERGY:")
    print(f"    Total energy:    {energy_kwh:.1f} kWh")
    print(f"")
    if use_safety:
        print(f"  SAFETY FILTER STATS:")
        print(f"    PPO accepted:    {ppo_accepted} ({accept_rate:.1f}%)")
        print(f"    Fallback used:   {ppo_rejected} ({100-accept_rate:.1f}%)")
    print(f"")
    print(f"  COMPARISON:")
    print(f"    ┌──────────────────────┬─────────┬──────────┐")
    print(f"    │ Controller           │  m_s    │ Viol. %  │")
    print(f"    ├──────────────────────┼─────────┼──────────┤")
    print(f"    │ Wang MPC             │  0.016  │   1.2%   │")
    print(f"    │ Wang Safe DRL        │  0.000  │   0.0%   │")
    print(f"    │ OURS (this run)      │  {pw_m_s:.3f}  │  {pw_viol_pct:.1f}%  │")
    print(f"    └──────────────────────┴─────────┴──────────┘")
    print(f"  Saved: {csv_path}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO + Surrogate Safety Filter"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--surrogate",
                        default="/app/outputs/surrogate_v2/rc_node_v3_tsupply.pt")
    parser.add_argument("--surrogate-kind", choices=["legacy_v3", "v35_raw", "v35_calibrated"], default="legacy_v3")
    parser.add_argument("--surrogate-summary-json", default=None)
    parser.add_argument("--surrogate-checkpoint", default=None)
    parser.add_argument("--surrogate-base-model", default=None)
    parser.add_argument("--out_dir",
                        default="/app/outputs/eval_safe_morl")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--margin", type=float, default=0.82)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--no_safety", action="store_true")
    args = parser.parse_args()

    eval_safe_morl(
        model_path=args.model,
        config_dir=args.config_dir,
        surrogate_path=args.surrogate,
        surrogate_kind=args.surrogate_kind,
        surrogate_summary_json=args.surrogate_summary_json,
        surrogate_checkpoint=args.surrogate_checkpoint,
        surrogate_base_model=args.surrogate_base_model,
        out_dir=args.out_dir,
        n_steps=args.steps,
        seed=args.seed,
        use_safety=not args.no_safety,
        horizon=args.horizon,
        margin=args.margin,
        warmup_steps=args.warmup,
    )


if __name__ == "__main__":
    main()
