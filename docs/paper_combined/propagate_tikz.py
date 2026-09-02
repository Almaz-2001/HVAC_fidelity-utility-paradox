"""Copy every compiled fig_*_tikz.pdf into figures/ under its manuscript basename.

All manuscript figures are now native TikZ/pgfplots standalones in this directory;
this mirrors them into figures/ (which main_paper.tex / supplementary.tex include).
Run after compiling the standalones: python docs/paper_combined/propagate_tikz.py
"""
from __future__ import annotations
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# fig_*_tikz.pdf  ->  manuscript basename (without .pdf)
MAP = {
    "fig_system_overview": "block2_system_overview",
    "fig_hybrid_mechanism": "final17_fig07_hybrid_reward_shaping_mechanism",
    "fig_rie03_stageabc": "rie_fig03_stage_abc_diagnostics",
    "fig_rie04_predictive": "rie_fig04_predictive_validity",
    "fig_rie05_waterfall": "rie_fig05_matched_corpus_attribution",
    "fig_fidelity_scatter": "block2_fidelity_utility_scatter",
    "fig_ms_decomp": "fig_block2_ms_decomposition",
    "fig_surface_mechanism": "block2_surface_response_curves",
    "fig_deployment_quadrants": "block3_q1_polish_deployment_quadrants",
    "fig_final17_fig15": "final17_fig15_full_stage_transfer_rmse_improvement",
    "fig_final_eng_fig12_czon": "final_eng_fig12_czon_hypothesis_interval",
    "fig_rie01_artifact": "rie_fig01_block1_artifact_chain",
    "fig_rie02_surrogate": "rie_fig02_surrogate_design",
    "fig_rie06_runtime": "rie_fig06_runtime_feasibility",
    "fig_rie07_episodes": "rie_fig07_episode_replicability",
    "fig_rie08_learning": "rie_fig08_v3_learning_curve",
    "fig_rie09_physics": "rie_fig09_physics_consistency",
    "fig_conceptual": "block2_conceptual_overview",
    "fig_block2_pipeline": "fig_block2_pipeline",
    "fig_block2_reward_shaping": "fig_block2_reward_shaping",
    "fig_block2_hdrl": "fig_block2_hdrl_architecture",
    "fig_block2_morl": "fig_block2_morl_pipeline",
    "fig_block3_protocol": "fig_block3_protocol",
    "fig_block3_adapter": "fig_block3_adapter",
    "fig_block3_topology": "fig_block3_topology",
    "fig_block3_controller_bar": "fig_block3_controller_bar",
    "fig_regime_progression": "fig_block3_regime_progression",
    "fig_seed_band": "block2_seed_band",
    "fig_hdrl_sweep": "block2_hdrl_lambda_sweep_sensitivity",
    "fig_lambda_specificity": "block2_lambda_specificity",
    "fig_morl_5d17d": "final17_fig09_morl_5d_failure_17d_success",
    "fig_morl_17d_heatmap": "block2_morl_17d_seasonal_heatmap",
    "fig_morl_inversion": "block2_morl_seasonal_variance_inversion",
    "fig_warmstart": "block2_warmstart_negative_eval_kpis",
    "fig_block1_fig10": "block1_q1_fig10_transfer_gap_diagnostics",
    "fig_final17_fig06": "final17_fig06_live_boptest_controller_comparison",
    "fig_fig13_verdict": "final17_fig13_block3_controller_transfer_heatmap",
    "fig_fig17_closure": "final17_fig17_hypothesis_closure_matrix",
    "fig_rte_control_loop": "rte_control_loop",
    "fig_runtime_fidelity": "block2_runtime_fidelity",
    "fig_closed_loop": "block2_q1_polish_closed_loop_disturbance",
    "fig_phase_density": "block2_q1_polish_phase_density",
}

copied, missing = [], []
for stem, dest in MAP.items():
    src = BASE / f"{stem}_tikz.pdf"
    if src.exists():
        shutil.copy2(src, FIG / f"{dest}.pdf")
        copied.append(dest)
    else:
        missing.append(stem)

print(f"copied {len(copied)} TikZ figures into {FIG}")
if missing:
    print(f"!! MISSING compiled PDFs (compile the standalone first): {missing}")
