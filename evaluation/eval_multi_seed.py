

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from evaluation.eval_safe_morl import eval_safe_morl


def run_multi_seed(
    model_path: str,
    surrogate_path: str,
    out_dir: str = "/app/outputs/eval_multi_seed",
    n_steps: int = 5000,
    seeds: list = None,
    horizon: int = 2,
    margin: float = 0.82,
) -> None:
    if seeds is None:
        seeds = [42, 43, 44]

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    all_results = []

    total_runs = len(seeds) * 2  # with and without safety
    run_idx = 0

    for seed in seeds:
        for use_safety in [False, True]:
            run_idx += 1
            mode = "WITH_SF" if use_safety else "NO_SF"
            timestamp = datetime.now().strftime("%H:%M:%S")

            print(f"\n{'#'*60}")
            print(f"# RUN {run_idx}/{total_runs}: seed={seed}, safety={mode}")
            print(f"# Started at {timestamp}")
            print(f"{'#'*60}")

            run_out_dir = os.path.join(out_dir, f"seed{seed}_{mode.lower()}")

            results = eval_safe_morl(
                model_path=model_path,
                surrogate_path=surrogate_path if use_safety else None,
                out_dir=run_out_dir,
                n_steps=n_steps,
                seed=seed,
                use_safety=use_safety,
                horizon=horizon,
                margin=margin,
            )

            results['seed'] = seed
            results['mode'] = mode
            all_results.append(results)

    # Aggregate
    df = pd.DataFrame(all_results)

    # Save raw results
    raw_path = os.path.join(out_dir, "all_results_raw.csv")
    df.to_csv(raw_path, index=False)

    # Print summary
    print(f"\n\n{'='*70}")
    print(f"MULTI-SEED EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Model:     {model_path}")
    print(f"  Surrogate: {surrogate_path}")
    print(f"  Seeds:     {seeds}")
    print(f"  Steps:     {n_steps}")
    print(f"")

    metrics = ['m_s', 'r_time', 'r_sev', 'violation_pct', 'energy_kwh', 'acceptance_rate']

    for mode in ['NO_SF', 'WITH_SF']:
        subset = df[df['mode'] == mode]
        label = "PPO alone" if mode == "NO_SF" else "PPO + Safety Filter"
        print(f"  {label}:")
        for m in metrics:
            if m in subset.columns:
                vals = subset[m].values
                mean = np.mean(vals)
                std = np.std(vals)
                print(f"    {m:20s}: {mean:.4f} +/- {std:.4f}  ({vals})")
        print()

    # Formatted comparison table
    no_sf = df[df['mode'] == 'NO_SF']
    with_sf = df[df['mode'] == 'WITH_SF']

    print(f"  COMPARISON TABLE (mean +/- std, n={len(seeds)}):")
    print(f"  {'':25s} {'PPO alone':>20s} {'PPO + SF':>20s} {'Delta':>12s}")
    print(f"  {'-'*80}")

    for m in ['m_s', 'violation_pct', 'energy_kwh']:
        v1 = no_sf[m].values
        v2 = with_sf[m].values
        m1, s1 = np.mean(v1), np.std(v1)
        m2, s2 = np.mean(v2), np.std(v2)
        delta = m2 - m1
        sign = '+' if delta > 0 else ''
        print(f"  {m:25s} {m1:8.3f} +/- {s1:.3f}   {m2:8.3f} +/- {s2:.3f}   {sign}{delta:.3f}")

    if 'acceptance_rate' in with_sf.columns:
        ar = with_sf['acceptance_rate'].values
        print(f"  {'acceptance_rate':25s} {'---':>20s} {np.mean(ar):8.1f} +/- {np.std(ar):.1f}%")

    # Wang et al comparison
    print(f"\n  COMPARISON WITH WANG ET AL.:")
    print(f"  ┌────────────────────────────┬──────────────────┬──────────────┐")
    print(f"  │ Controller                 │  m_s             │ Viol. %      │")
    print(f"  ├────────────────────────────┼──────────────────┼──────────────┤")
    print(f"  │ PI Controller (Wang)       │  0.096           │   9.3%       │")
    print(f"  │ MPC (Wang)                 │  0.016           │   1.2%       │")
    print(f"  │ Safe DRL (Wang)            │  0.000           │   0.0%       │")

    m_no = np.mean(no_sf['m_s'].values)
    s_no = np.std(no_sf['m_s'].values)
    v_no = np.mean(no_sf['violation_pct'].values)
    print(f"  │ Our PPO alone              │  {m_no:.3f} +/- {s_no:.3f}  │  {v_no:.1f}%      │")

    m_sf = np.mean(with_sf['m_s'].values)
    s_sf = np.std(with_sf['m_s'].values)
    v_sf = np.mean(with_sf['violation_pct'].values)
    print(f"  │ Our PPO + SF               │  {m_sf:.3f} +/- {s_sf:.3f}  │  {v_sf:.1f}%      │")
    print(f"  └────────────────────────────┴──────────────────┴──────────────┘")

    print(f"\n  Raw results: {raw_path}")
    print(f"{'='*70}")

    # Save summary
    summary = {
        'ppo_alone_ms_mean': np.mean(no_sf['m_s'].values),
        'ppo_alone_ms_std': np.std(no_sf['m_s'].values),
        'ppo_sf_ms_mean': np.mean(with_sf['m_s'].values),
        'ppo_sf_ms_std': np.std(with_sf['m_s'].values),
        'ppo_alone_viol_mean': np.mean(no_sf['violation_pct'].values),
        'ppo_alone_viol_std': np.std(no_sf['violation_pct'].values),
        'ppo_sf_viol_mean': np.mean(with_sf['violation_pct'].values),
        'ppo_sf_viol_std': np.std(with_sf['violation_pct'].values),
        'ppo_sf_accept_mean': np.mean(with_sf['acceptance_rate'].values),
        'ppo_sf_accept_std': np.std(with_sf['acceptance_rate'].values),
        'n_seeds': len(seeds),
        'n_steps': n_steps,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed evaluation of PPO +/- Safety Filter"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--surrogate",
                        default="/app/outputs/surrogate_v2/rc_node_v2_best.pt")
    parser.add_argument("--out_dir",
                        default="/app/outputs/eval_multi_seed")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--margin", type=float, default=0.82)
    args = parser.parse_args()

    run_multi_seed(
        model_path=args.model,
        surrogate_path=args.surrogate,
        out_dir=args.out_dir,
        n_steps=args.steps,
        seeds=args.seeds,
        horizon=args.horizon,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()