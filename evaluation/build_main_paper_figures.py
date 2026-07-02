from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "article_real"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, stem: str) -> None:
    fig.patch.set_facecolor("white")
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def box(ax, xy, w, h, text, fc="#eef4f7", ec="#2f4858", fontsize=9):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=1.3,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return patch


def arrow(ax, start, end, text=None, color="#3d5a80"):
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color=color)
    ax.add_patch(arr)
    if text:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.04,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )


def figure_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    ax.axis("off")
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(0, 1)
    ax.set_title("Hybrid backend: v3 dynamics with frozen v3.5 physical regularizer", fontsize=12, weight="bold", pad=10)

    box(ax, (0.03, 0.56), 0.16, 0.18, "State s_t\npolicy obs", fc="#f7f7f7")
    box(ax, (0.27, 0.60), 0.19, 0.17, "PPO policy\npi_theta(a|s)", fc="#e9f5db")
    box(ax, (0.55, 0.63), 0.19, 0.16, "v3 surrogate\ntrain-time dynamics", fc="#dceefb")
    box(ax, (0.55, 0.31), 0.19, 0.16, "calibrated v3.5\nfrozen physical twin", fc="#fde2e4")
    box(ax, (0.29, 0.20), 0.20, 0.15, "PPO objective\n+ disagreement loss", fc="#fff3bf")
    box(ax, (0.83, 0.58), 0.14, 0.20, "next state\nreward", fc="#f7f7f7")

    arrow(ax, (0.19, 0.65), (0.27, 0.68), "s_t")
    arrow(ax, (0.46, 0.69), (0.55, 0.71), "a_t")
    arrow(ax, (0.74, 0.71), (0.83, 0.68))
    arrow(ax, (0.46, 0.62), (0.55, 0.39), "same (s,a)", color="#b23a48")
    arrow(ax, (0.55, 0.39), (0.49, 0.28), color="#b23a48")
    arrow(ax, (0.55, 0.67), (0.49, 0.31), color="#3d5a80")
    arrow(ax, (0.38, 0.35), (0.36, 0.60), "L_total", color="#a67c00")
    ax.text(
        0.50,
        0.08,
        r"$L_{total}=L_{PPO}+\lambda_{temp}\|T_{v3}-T_{v3.5}\|^2+\lambda_{power}\|P_{v3}-P_{v3.5}\|^2$",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.86, bottom=0.12)
    save(fig, "main_fig1_pipeline_schematic")


def figure_fidelity_gap() -> None:
    arch = pd.read_csv(ROOT / "reports" / "hou_evins_architecture_justification_table.csv")
    pred = pd.read_csv(ROOT / "reports" / "hou_evins_predictive_validity_table.csv")
    rows = []
    for model, label in [("v3", "v3"), ("v35_calibrated", "v3.5 calibrated"), ("hybrid_l010", "hybrid_l010")]:
        if model == "v35_calibrated":
            pred_val = float(pred[(pred["model"] == "v3.5_calibrated") & (pred["horizon"] == "24h")]["RMSE_T"].iloc[0])
        else:
            pred_val = float(pred[(pred["model"] == model) & (pred["horizon"] == "24h")]["RMSE_T"].iloc[0])
        transfer = arch[arch["variant"].eq(model)]
        live = float(np.nanmean([transfer["peak_transfer_temp_rmse_c"].iloc[0], transfer["typical_transfer_temp_rmse_c"].iloc[0]]))
        rows.append((label, pred_val, live))
    df = pd.DataFrame(rows, columns=["model", "predictive_24h_rmse", "live_transfer_rmse"])

    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, df["predictive_24h_rmse"], width, label="Predictive 24h RMSE", color="#4c78a8")
    ax.bar(x + width / 2, df["live_transfer_rmse"], width, label="Live BOPTEST transfer RMSE", color="#f58518")
    ax.set_xticks(x, df["model"])
    ax.set_ylabel("Temperature RMSE (C)")
    ax.set_title("Fidelity-to-control gap", weight="bold")
    ax.set_ylim(0, float(df[["predictive_24h_rmse", "live_transfer_rmse"]].to_numpy().max()) * 1.22)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    for i, (pred_val, live_val) in enumerate(zip(df["predictive_24h_rmse"], df["live_transfer_rmse"])):
        ax.text(i - width / 2, pred_val + 0.07, f"{pred_val:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, live_val + 0.07, f"{live_val:.2f}", ha="center", fontsize=8)
        if pred_val > 0:
            ax.text(i, max(pred_val, live_val) + 0.25, f"{live_val / pred_val:.1f}x", ha="center", fontsize=9, color="#7a3e00")
    save(fig, "main_fig3_fidelity_to_rl_gap")


def figure_stage_calibration() -> None:
    stage_b = pd.read_csv(ROOT / "outputs" / "surrogate_v35_inverse_boptest_15min_episodeaware" / "stage_b_history_v35.csv")
    metrics = ["1-step RMSE_T", "24h RMSE_T", "Power MAE (kW)"]
    raw = np.array([0.3839, 1.4665, 0.8103])
    cal = np.array([0.2348, 0.6441, 0.4820])
    x = np.arange(len(metrics))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].bar(x - 0.18, raw, 0.36, label="raw v3.5", color="#b25f2c")
    axes[0].bar(x + 0.18, cal, 0.36, label="calibrated v3.5", color="#21867a")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    for i, (b, a) in enumerate(zip(raw, cal)):
        axes[0].text(i, max(b, a) * 1.06, f"-{100*(1-a/b):.0f}%", ha="center", fontsize=8, weight="bold")
    style(axes[0], "Stage A/B/C predictive-fidelity gain", ylabel="metric value")
    axes[0].legend(frameon=False)

    axes[1].plot(stage_b["epoch"], stage_b["c_zon_j_per_k"] / 1e5, color="#21867a", linewidth=2.4)
    axes[1].axhline(4.2, color="#5c6470", linestyle="--", linewidth=1.1, label="prior")
    axes[1].axhline(stage_b["c_zon_j_per_k"].iloc[-1] / 1e5, color="#b25f2c", linestyle=":", linewidth=1.6, label="identified")
    style(axes[1], "Stage B inverse identification of C_zon", "epoch", "C_zon (1e5 J/K)")
    axes[1].legend(frameon=False)
    fig.suptitle("Figure 2. Stage A/B/C calibration and physical parameter identification", fontsize=13, weight="bold")
    save(fig, "main_fig2_stage_abc_czon")


def figure_matched_decomposition() -> None:
    matched = pd.read_csv(ROOT / "reports" / "block1_corpus_matched_comparison.csv")
    variants = ["v3_hourly", "v3_15min_matched", "v35_raw", "v35_calibrated"]
    labels = ["v3 hourly", "v3 15-min", "raw v3.5", "calibrated v3.5"]
    vals = [float(matched[matched["variant"].eq(v)]["rmse_24h_c"].iloc[0]) for v in variants]
    start, mid, end = 1.5572, 0.8761, 0.6441
    corpus, calib = start - mid, mid - end

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].bar(labels, vals, color=["#2f5d8c", "#5b8cc0", "#b25f2c", "#21867a"])
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.04, f"{v:.3f}", ha="center", fontsize=8)
    axes[0].tick_params(axis="x", rotation=15)
    style(axes[0], "Corpus-controlled 24h RMSE", ylabel="RMSE_T (C)")

    axes[1].bar([0], [start], color="#2f5d8c", width=0.55)
    axes[1].bar([1], [-corpus], bottom=[start], color="#76a7d8", width=0.55)
    axes[1].bar([2], [-calib], bottom=[mid], color="#21867a", width=0.55)
    axes[1].bar([3], [end], color="#6f4e7c", width=0.55)
    axes[1].set_xticks([0, 1, 2, 3])
    axes[1].set_xticklabels(["start\n1.557", "corpus\n-0.681\n74.6%", "calibration\n-0.232\n25.4%", "final\n0.644"])
    style(axes[1], "Attribution waterfall", ylabel="RMSE_T (C)")
    fig.suptitle("Figure 3. Matched-corpus decomposition of predictive-fidelity gain", fontsize=13, weight="bold")
    save(fig, "main_fig3_matched_corpus_decomposition")


def figure_fidelity_control_combined() -> None:
    arch = pd.read_csv(ROOT / "reports" / "hou_evins_architecture_justification_table.csv")
    points = [
        ("v3", 1.5572, float(arch[arch["variant"].eq("v3")]["typical_control_m_s"].iloc[0]), "#2f5d8c"),
        ("v3.5", 0.6441, float(arch[arch["variant"].eq("v35_calibrated")]["typical_control_m_s"].iloc[0]), "#b25f2c"),
        ("hybrid", 0.6441, float(arch[arch["variant"].eq("hybrid_l010")]["typical_control_m_s"].iloc[0]), "#6f4e7c"),
    ]
    labels = ["v3", "v3.5", "hybrid"]
    variants = ["v3", "v35_calibrated", "hybrid_l010"]
    colors = ["#2f5d8c", "#b25f2c", "#6f4e7c"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for label, x, y, color in points:
        axes[0].scatter([x], [y], s=150, color=color, edgecolor="#222222")
        axes[0].text(x + 0.03, y + 0.025, label, fontsize=9, weight="bold")
    style(axes[0], "Predictive fidelity vs live utility", "24h RMSE_T (C)", "typical live m_s")
    axes[0].set_ylim(0, 1.25)

    x = np.arange(len(labels))
    w = 0.34
    peak = [float(arch[arch["variant"].eq(v)]["peak_control_m_s"].iloc[0]) for v in variants]
    typ = [float(arch[arch["variant"].eq(v)]["typical_control_m_s"].iloc[0]) for v in variants]
    axes[1].bar(x - w / 2, peak, w, color=colors, alpha=0.85, label="peak")
    axes[1].bar(x + w / 2, typ, w, color=colors, alpha=0.45, label="typical")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    style(axes[1], "Live BOPTEST controller performance", ylabel="m_s")
    axes[1].legend(frameon=False)
    fig.suptitle("Figure 4. Predictive fidelity does not imply RL training utility", fontsize=13, weight="bold")
    save(fig, "main_fig4_fidelity_control")


def figure_morl_summary() -> None:
    pareto = pd.read_csv(ROOT / "reports" / "morl_pareto_front_table.csv")
    neutral = pd.read_csv(ROOT / "reports" / "morl_neutral_canonical_monthly_variance_diagnostic.csv")
    practical = pd.read_csv(ROOT / "reports" / "morl_practical_canonical_monthly_variance_diagnostic.csv")
    seeds = pd.read_csv(ROOT / "reports" / "morl_canonical_seedfix_yearly_per_seed.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1.25, 1.15, 0.9]})
    axes[0].scatter(pareto["energy_kwh_mean"], pareto["ms_mean"], c="#6f4e7c", s=52)
    for _, r in pareto.iterrows():
        if r.get("kind", "") == "canonical_n5":
            axes[0].errorbar(r["energy_kwh_mean"], r["ms_mean"], xerr=r.get("energy_kwh_ci95", 0), yerr=r.get("ms_ci95", 0), fmt="none", ecolor="#222222", capsize=3)
    style(axes[0], "MORL Pareto sweep", "energy (kWh)", "m_s")

    month_col = "month" if "month" in neutral.columns else "scenario"
    ms_std_col = "m_s_std" if "m_s_std" in neutral.columns else "ms_std"
    months = sorted(set(neutral[month_col]).intersection(set(practical[month_col])))
    nstd = neutral.set_index(month_col).reindex(months)[ms_std_col].to_numpy()
    pstd = practical.set_index(month_col).reindex(months)[ms_std_col].to_numpy()
    data = np.vstack([nstd, pstd])
    im = axes[1].imshow(data, aspect="auto", cmap="viridis")
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["neutral", "practical"])
    axes[1].set_xticks(range(len(months))); axes[1].set_xticklabels([m[:3] for m in months], rotation=45, ha="right", fontsize=7)
    axes[1].set_title("Monthly seed variance", loc="left", fontsize=11, weight="bold")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="std(m_s)")

    seed_ms_col = "m_s" if "m_s" in seeds.columns else "ms_mean"
    grouped = [seeds[seeds["canonical"].str.contains("neutral|0.50|050", case=False, na=False)][seed_ms_col].to_numpy(),
               seeds[seeds["canonical"].str.contains("practical|0.75|075", case=False, na=False)][seed_ms_col].to_numpy()]
    if len(grouped[0]) == 0 or len(grouped[1]) == 0:
        grouped = [seeds.iloc[:5][seed_ms_col].to_numpy(), seeds.iloc[5:][seed_ms_col].to_numpy()]
    axes[2].boxplot(grouped, labels=["neutral", "practical"], patch_artist=True)
    style(axes[2], "N=5 canonical seed spread", ylabel="yearly m_s")
    fig.suptitle("Figure 5. MORL Pareto structure and seed-variance diagnostics", fontsize=13, weight="bold")
    save(fig, "main_fig5_morl_pareto_variance")


def figure_audit_timeline() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    events = [
        (0.08, "Block 2\npre-reg\n93df9b3"),
        (0.28, "MORL N=5\nfalsification\n62dc859"),
        (0.50, "Block 3\nmanifest\npre-run"),
        (0.72, "Adapter specs\nbefore\ncontrol runs"),
        (0.91, "Block 3\nclosed\nN=3"),
    ]
    ax.plot([0.08, 0.91], [0.5, 0.5], color="#5c6470", linewidth=2)
    for x, label in events:
        ax.scatter([x], [0.5], s=180, color="#2f5d8c", edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(x, 0.66, label, ha="center", va="bottom", fontsize=9, weight="bold")
    ax.text(0.03, 0.15, "Figure 8. Audit/pre-registration timeline. Predictions and adapter mappings were committed before the corresponding BOPTEST runs; results were appended afterward.", fontsize=11, weight="bold")
    save(fig, "main_fig8_audit_timeline")


def figure_transfer_heatmap() -> None:
    matrix = pd.read_csv(ROOT / "reports" / "block3_transfer_matrix.csv")
    regimes = ["none", "partial", "full"]
    short_tests = ["heat pump", "hydronic", "commercial"]
    vals = np.full((len(short_tests), len(regimes)), np.nan)
    labels = [["" for _ in regimes] for _ in short_tests]
    for i, row in matrix.iterrows():
        vals[i, 0] = 1 if row["none_controller_verdict"] == "PASS" else -1
        labels[i][0] = row["none_controller_verdict"]
        # Partial live KPI is unchanged under frozen-controller scope; unrun cells are structural, not missing evidence.
        vals[i, 1] = vals[i, 0]
        labels[i][1] = "same as\nnone"
        vals[i, 2] = 1 if row["full_controller_verdict"] == "PASS" else -1
        labels[i][2] = row["full_controller_verdict"]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    cmap = plt.matplotlib.colors.ListedColormap(["#d73027", "#1a9850"])
    norm = plt.matplotlib.colors.BoundaryNorm([-1.5, 0, 1.5], cmap.N)
    ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(regimes)), regimes)
    ax.set_yticks(np.arange(len(short_tests)), short_tests)
    ax.set_title("Block 3 controller verdict heatmap", weight="bold")
    for i in range(len(short_tests)):
        for j in range(len(regimes)):
            ax.text(j, i, labels[i][j], ha="center", va="center", color="white", fontsize=9, weight="bold")
    ax.set_xlabel("Recalibration regime")
    ax.set_ylabel("Target testcase")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, "main_fig5_block3_transfer_verdict_heatmap")


def figure_czon_consistency() -> None:
    matrix = pd.read_csv(ROOT / "reports" / "block3_transfer_matrix.csv")
    labels = ["heat pump", "hydronic", "commercial"]
    ratios = matrix["c_zon_ratio_vs_bestest_air"].to_numpy(dtype=float)
    mean = ratios.mean()
    std = ratios.std(ddof=0)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bars = ax.bar(labels, ratios, color=["#4c78a8", "#72b7b2", "#f58518"], width=0.58)
    ax.axhline(mean, color="#333333", linestyle="--", linewidth=1.2, label=f"mean={mean:.2f}x")
    ax.fill_between([-0.5, 2.5], mean - std, mean + std, color="#999999", alpha=0.12, label=f"+/-1 sigma={std:.2f}x")
    ax.set_ylim(0, max(2.3, ratios.max() + 0.25))
    ax.set_ylabel("C_zon ratio vs bestest_air")
    ax.set_title("Hydronic-family C_zon consistency", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.04, f"{val:.2f}x", ha="center", fontsize=9)
    save(fig, "main_fig6_block3_czon_consistency")


def main() -> None:
    figure_pipeline()
    figure_stage_calibration()
    figure_matched_decomposition()
    figure_fidelity_control_combined()
    figure_fidelity_gap()
    figure_morl_summary()
    figure_transfer_heatmap()
    figure_czon_consistency()
    figure_audit_timeline()


if __name__ == "__main__":
    main()
