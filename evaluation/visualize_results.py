

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from pathlib import Path





PARETO = {
    'comfort_only':     {'energy_kwh': 192.3, 'ms': 1.284, 'viol': 78.7},
    'comfort_dominant': {'energy_kwh': 190.2, 'ms': 1.289, 'viol': 78.7},
    'balanced':         {'energy_kwh': 187.1, 'ms': 1.294, 'viol': 78.7},
    'energy_dominant':  {'energy_kwh': 179.6, 'ms': 1.310, 'viol': 78.7},
    'energy_only':      {'energy_kwh': 138.7, 'ms': 1.389, 'viol': 79.7},
}


SURROGATE = {
    'Phase 1 (v1)': {'rmse': 1.128, 'r2': 0.884, 'data_k': 10, 'params': 4800, 'inputs': 3},
    'Phase 3 (v2)': {'rmse': 0.163, 'r2': 0.991, 'data_k': 51.2, 'params': 8482, 'inputs': 8},
}


ROLLOUT = {
    1: {'rmse': 0.303, 'bias': 0.030, 'false_safe': 1.7, 'margin_95': 0.63},
    2: {'rmse': 0.413, 'bias': 0.052, 'false_safe': 2.2, 'margin_95': 0.82},
    4: {'rmse': 0.578, 'bias': 0.085, 'false_safe': 2.6, 'margin_95': 1.11},
    6: {'rmse': 0.710, 'bias': 0.109, 'false_safe': 3.2, 'margin_95': 1.31},
}


MULTI_SEED = {
    'no_sf':   {'viol': [81.8, 75.8, 77.2], 'energy': [336, 198, 207]},
    'with_sf': {'viol': [66.9, 67.2, 67.9], 'energy': [264, 269, 267]},
}


CALIBRATION = {
    'Before (uncalibrated)': 9.12,
    'Linear calibration':    8.65,
    'Finetune calibration':  1.91,
}


MS_COMPARISON = {
    'Phase 0\n(PPO 5k)':       1.284,
    'Phase 1\n(PPO 100k\nsurrogate)': 0.510,
    'Phase 3\n(PPO+SF\nBOPTEST)': 1.474,
    'Wang\nMPC':                0.016,
    'Wang\nSafe DRL':           0.000,
}


OUT_DIR = os.environ.get("FIGURE_OUTPUT_DIR", "/app/outputs/figures")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
})

PURPLE = '#534AB7'
TEAL = '#1D9E75'
CORAL = '#D85A30'
GRAY = '#73726c'
RED = '#E24B4A'
BLUE = '#3266ad'



def plot_ms_comparison():
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = list(MS_COMPARISON.keys())
    values = list(MS_COMPARISON.values())
    colors = [PURPLE, PURPLE, TEAL, CORAL, CORAL]

    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor='white', linewidth=0.5)

    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    
    ax.axhline(y=0.1, color=RED, linestyle='--', linewidth=1.2, alpha=0.7, label='Target m_s < 0.1')

    ax.set_ylabel('Safety metric $m_s$ (lower is better)')
    ax.set_title('Figure 1: Safety metric $m_s$ across all phases')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.7)

    
    handles = [
        mpatches.Patch(color=PURPLE, label='Our PPO'),
        mpatches.Patch(color=TEAL, label='Our PPO + Safety Filter'),
        mpatches.Patch(color=CORAL, label='Wang et al. (2025)'),
    ]
    ax.legend(handles=handles, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_ms_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')



def plot_surrogate_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    
    ax = axes[0]
    ax.bar(['Phase 1\n(3 inputs)', 'Phase 3\n(8 inputs)'],
           [1.128, 0.163], color=[GRAY, PURPLE], width=0.5)
    ax.set_ylabel('RMSE (°C)')
    ax.set_title('1-step RMSE')
    ax.annotate('6.9× better', xy=(1, 0.163), fontsize=10, color=TEAL,
                ha='center', va='bottom', xytext=(1, 0.35),
                arrowprops=dict(arrowstyle='->', color=TEAL))

    
    ax = axes[1]
    ax.bar(['Phase 1', 'Phase 3'], [0.884, 0.991], color=[GRAY, PURPLE], width=0.5)
    ax.set_ylabel('R²')
    ax.set_title('Coefficient of determination')
    ax.set_ylim(0.8, 1.0)
    ax.axhline(y=0.90, color=RED, linestyle='--', alpha=0.5, label='Target > 0.90')
    ax.legend(fontsize=9)

    # Training data
    ax = axes[2]
    ax.bar(['Phase 1', 'Phase 3'], [10, 51.2], color=[GRAY, PURPLE], width=0.5)
    ax.set_ylabel('Training data (k steps)')
    ax.set_title('Data collection')
    ax.bar_label(ax.containers[0], fmt='%.0fk', fontsize=10)

    fig.suptitle('Figure 2: Surrogate model improvement (Phase 1 → Phase 3)', fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_surrogate_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')



def plot_rollout_validation():
    fig, ax1 = plt.subplots(figsize=(8, 5))

    horizons = list(ROLLOUT.keys())
    rmse = [ROLLOUT[h]['rmse'] for h in horizons]
    false_safe = [ROLLOUT[h]['false_safe'] for h in horizons]
    margins = [ROLLOUT[h]['margin_95'] for h in horizons]

    
    ln1 = ax1.plot(horizons, rmse, 'o-', color=PURPLE, linewidth=2, markersize=8,
                   label='Rollout RMSE (°C)')
    ax1.fill_between(horizons, rmse, alpha=0.1, color=PURPLE)
    ax1.set_xlabel('Prediction horizon (hours)')
    ax1.set_ylabel('Rollout RMSE (°C)', color=PURPLE)
    ax1.tick_params(axis='y', labelcolor=PURPLE)
    ax1.set_ylim(0, 1.0)

    
    ln2 = ax1.plot(horizons, margins, 's--', color=BLUE, linewidth=1.5, markersize=7,
                   label='Safety margin 95% (°C)')

    
    ax2 = ax1.twinx()
    ln3 = ax2.plot(horizons, false_safe, '^-', color=RED, linewidth=2, markersize=8,
                   label='False-safe rate (%)')
    ax2.set_ylabel('False-safe rate (%)', color=RED)
    ax2.tick_params(axis='y', labelcolor=RED)
    ax2.set_ylim(0, 5)

    
    ax1.axvspan(1.7, 2.3, alpha=0.15, color=TEAL, label='Selected horizon')
    ax1.annotate('Chosen:\nhorizon=2\nmargin=0.82°C', xy=(2, 0.413), fontsize=9,
                 xytext=(3.5, 0.2), arrowprops=dict(arrowstyle='->', color=TEAL),
                 color=TEAL, fontweight='bold')

    
    lines = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=9)

    ax1.set_title('Figure 3: Safety filter validation — rollout accuracy vs horizon')
    ax1.set_xticks(horizons)
    ax1.set_xticklabels([f'{h}h' for h in horizons])
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_rollout_validation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')



def plot_multi_seed():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    seeds = ['seed 42', 'seed 43', 'seed 44']
    x = np.arange(len(seeds))
    w = 0.35

    
    ax = axes[0]
    b1 = ax.bar(x - w/2, MULTI_SEED['no_sf']['viol'], w, color=GRAY, label='PPO alone')
    b2 = ax.bar(x + w/2, MULTI_SEED['with_sf']['viol'], w, color=PURPLE, label='PPO + SF')
    ax.set_ylabel('Violation %')
    ax.set_title('Violation % (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.set_ylim(0, 100)

    
    no_mean = np.mean(MULTI_SEED['no_sf']['viol'])
    sf_mean = np.mean(MULTI_SEED['with_sf']['viol'])
    ax.axhline(no_mean, color=GRAY, linestyle=':', alpha=0.6)
    ax.axhline(sf_mean, color=PURPLE, linestyle=':', alpha=0.6)
    ax.annotate(f'mean={no_mean:.1f}%', xy=(2.3, no_mean), fontsize=9, color=GRAY)
    ax.annotate(f'mean={sf_mean:.1f}%', xy=(2.3, sf_mean), fontsize=9, color=PURPLE)

    
    ax = axes[1]
    b1 = ax.bar(x - w/2, MULTI_SEED['no_sf']['energy'], w, color=GRAY, label='PPO alone')
    b2 = ax.bar(x + w/2, MULTI_SEED['with_sf']['energy'], w, color=PURPLE, label='PPO + SF')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Energy consumption')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend()

    
    no_std = np.std(MULTI_SEED['no_sf']['energy'])
    sf_std = np.std(MULTI_SEED['with_sf']['energy'])
    ax.annotate(f'std={no_std:.0f}', xy=(2.3, np.mean(MULTI_SEED['no_sf']['energy'])),
                fontsize=9, color=GRAY)
    ax.annotate(f'std={sf_std:.0f}', xy=(2.3, np.mean(MULTI_SEED['with_sf']['energy'])),
                fontsize=9, color=PURPLE)

    fig.suptitle('Figure 4: Multi-seed evaluation on BOPTEST (n=3 PPO models)', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_multi_seed.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')



def plot_calibration():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    labels = list(CALIBRATION.keys())
    values = list(CALIBRATION.values())
    colors = [CORAL, GRAY, TEAL]

    bars = ax.bar(labels, values, color=colors, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.2f}°C', ha='center', va='bottom', fontsize=11, fontweight='bold')

    
    ax.annotate('79% improvement', xy=(2, 1.91), xytext=(1, 5),
                fontsize=11, color=TEAL, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=2))

    ax.set_ylabel('RMSE (°C)')
    ax.set_title('Figure 5: Phase 2 — Inverse problem calibration results')
    ax.set_ylim(0, 11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig5_calibration.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')



def plot_pareto():
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(PARETO.keys())
    energies = [PARETO[n]['energy_kwh'] for n in names]
    ms_vals = [PARETO[n]['ms'] for n in names]

    
    ax.plot(energies, ms_vals, 'o-', color=PURPLE, linewidth=2, markersize=10, zorder=3)

    
    for i, name in enumerate(names):
        short = name.replace('_', '\n')
        offset = (8, 8) if i % 2 == 0 else (-8, -15)
        ax.annotate(short, xy=(energies[i], ms_vals[i]),
                    xytext=offset, textcoords='offset points',
                    fontsize=8, ha='left', color=PURPLE)

    
    ax.annotate('', xy=(135, 1.26), xytext=(195, 1.41),
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=2.5))
    ax.text(155, 1.35, '28% energy\nreduction', fontsize=10, color=TEAL,
            ha='center', fontweight='bold')

    ax.set_xlabel('Energy consumption (kWh)')
    ax.set_ylabel('Safety metric $m_s$')
    ax.set_title('Figure 6: Phase 0 — Pareto front (5 MORL weight configurations)')
    ax.invert_xaxis()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig6_pareto_front.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')



def main():
    print(f"Generating figures → {OUT_DIR}/\n")

    plot_ms_comparison()
    plot_surrogate_comparison()
    plot_rollout_validation()
    plot_multi_seed()
    plot_calibration()
    plot_pareto()

    print(f"\nAll 6 figures saved to {OUT_DIR}/")
    print("Open in VS Code: click any PNG in the file explorer")


if __name__ == "__main__":
    main()
