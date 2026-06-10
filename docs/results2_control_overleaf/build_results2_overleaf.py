"""Build the data-driven Overleaf package for Results II / Block 2.

The section follows Block 2 of ``roadmap.md`` (Sections 4-11): pure v3
thermostatic PPO baseline, direct-v3.5 negative control, thermostatic hybrid,
warm-start negative control, transfer diagnostics, HDRL sweep, MORL 5D->17D
observation ablation, MORL Pareto + N=5 canonical seed analysis, seasonal
falsification, and PI reference.

Design: every table and inline KPI is read from versioned project artifacts in
``reports/`` and ``outputs/`` (provenance map: roadmap Section 11.1). Figures are
referenced from ``figures/`` (already produced by the Block 2 evaluation
scripts); this builder does not regenerate them. It writes ``main.tex`` only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"

NAVY = "#1f4e79"; TEAL = "#008080"; AMBER = "#c9822b"
GREEN = "#3b7d3a"; SLATE = "#5d6875"; PURPLE = "#6b5b95"; BURGUNDY = "#9b3d3d"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 9.5, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "figure.dpi": 130,
})


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def _save(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _box(ax, x, y, w, h, text, color, fc="#ffffff", fs=8.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.2, edgecolor=color, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#1f2933")


def _arrow(ax, start, end, color=SLATE, text=None):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=color))
    if text:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.03, text, ha="center", va="bottom", fontsize=7.6, color=color)


def fig_reward_shaping(ctx: dict) -> None:
    """Clean, non-overlapping reward-shaping schematic with real lambda values and
    measured disagreement. Replaces the legacy figure whose green box overlapped
    the title."""
    fig, ax = plt.subplots(figsize=(11.0, 4.9))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # Title rows (kept above the diagram band so nothing overlaps).
    ax.text(0.5, 0.96, "Hybrid backend: per-step reward shaping", ha="center",
            fontsize=13, weight="bold", color="#1f2933")
    ax.text(0.5, 0.885, "v3.5 is a frozen reward-shaping censor --- NOT a policy-loss term and NOT the rollout model",
            ha="center", fontsize=9.2, style="italic", color=SLATE)
    # Diagram band y in [0.30, 0.78].
    _box(ax, 0.02, 0.46, 0.17, 0.20, "state $s_t$,\naction $a_t$", NAVY, "#eef5fb")
    _box(ax, 0.27, 0.60, 0.22, 0.16, "v3 rollout dynamics\n$T_{v3},\\,P_{v3}$", TEAL, "#edf8f7")
    _box(ax, 0.27, 0.34, 0.22, 0.16, "frozen v3.5 censor\n$T_{v3.5},\\,P_{v3.5}$", GREEN, "#eef8ee")
    _box(ax, 0.55, 0.46, 0.18, 0.20, "disagreement\n$|\\Delta T|,\\,|\\Delta P|$", AMBER, "#fff6ea")
    _box(ax, 0.785, 0.42, 0.195, 0.28,
         "reward\n$r=r_{c}+r_{s}+r_{e}$\n$-\\lambda_T|\\Delta T|-\\lambda_P|\\Delta P|$", PURPLE, "#f4f1fa", fs=8.0)
    _arrow(ax, (0.19, 0.56), (0.27, 0.66), TEAL)
    _arrow(ax, (0.19, 0.56), (0.27, 0.44), GREEN)
    _arrow(ax, (0.49, 0.66), (0.55, 0.58), TEAL)
    _arrow(ax, (0.49, 0.42), (0.55, 0.52), GREEN)
    _arrow(ax, (0.73, 0.56), (0.785, 0.56), AMBER)
    # Bottom annotation row, well below the diagram band (no overlap).
    ax.text(0.5, 0.12,
            f"canonical thermostatic: $\\lambda_T={ctx['lam_T']}$, $\\lambda_P={ctx['lam_P']}$"
            f"   |   measured disagreement: mean $|\\Delta T|={ctx['dis_temp_mean']}\\,^\\circ$C, "
            f"mean $|\\Delta P|={ctx['dis_pow_mean']}$ W",
            ha="center", fontsize=8.6, color="#374151")
    ax.text(0.5, 0.045, "PPO rolls out only on v3; v3.5 is evaluated in parallel and enters the scalar reward, "
            "so the advantage $A_t$ uses the shaped reward unchanged.",
            ha="center", fontsize=8.2, style="italic", color=SLATE)
    _save(fig, "fig_block2_reward_shaping")


def fig_ms_decomposition(rows: list) -> None:
    """Data-driven decomposition m_s = r_time + r_sev for the three thermostatic
    backends on both windows (r_time = violation/100; r_sev = m_s - r_time)."""
    labels = [r[0] for r in rows]
    r_time = [r[1] for r in rows]
    r_sev = [r[2] for r in rows]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.bar(x, r_time, color=TEAL, label="$r_{\\mathrm{time}}$ (violation fraction)", edgecolor="#111827", linewidth=0.4)
    ax.bar(x, r_sev, bottom=r_time, color=BURGUNDY, label="$r_{\\mathrm{sev}}$ (worst rel. severity)", edgecolor="#111827", linewidth=0.4)
    for i, (a, b) in enumerate(zip(r_time, r_sev)):
        ax.text(i, a + b + 0.02, f"{a+b:.3f}", ha="center", fontsize=8.2, weight="bold")
    ax.axhline(0.10, color=SLATE, linestyle="--", linewidth=1.0)
    ax.text(len(labels) - 0.5, 0.11, "$m_s=0.10$", color=SLATE, fontsize=8, ha="right")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("$m_s = r_{\\mathrm{time}} + r_{\\mathrm{sev}}$")
    ax.set_title("Maintenance-score decomposition on the live BOPTEST windows", loc="left", weight="bold")
    ax.grid(True, axis="y", color="#e6e8eb", linewidth=0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_block2_ms_decomposition")


def tex_escape(value: object) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        text = text.replace(a, b)
    return text


def f(value: float, nd: int = 3) -> str:
    return f"{float(value):.{nd}f}"


# ---------------------------------------------------------------------------
# Data accessors
# ---------------------------------------------------------------------------

def _scen_row(df: pd.DataFrame, scenario: str, **conds) -> pd.Series:
    sub = df[df["scenario"] == scenario]
    for k, v in conds.items():
        sub = sub[sub[k] == v]
    return sub.iloc[0]


def load_block2():
    return {
        "pure": read_csv("outputs/bestest_air_article7_style_15min/summary.csv"),
        "hybrid": read_csv("outputs/block2_thermostatic_hybrid_v3_v35_l010/summary.csv"),
        "warm": read_csv("outputs/block2_thermostatic_warmstart_utility/comparison_summary.csv"),
        "arch": read_csv("reports/hou_evins_architecture_justification_table.csv"),
        "transfer": read_csv("reports/hybrid_transfer_comparison.csv"),
        "hdrl": read_csv("reports/block2_hdrl_lambda_sweep_summary.csv"),
        "morl_recon": read_csv("reports/block2_morl_5d_reconstructed_comparison.csv"),
        "morl_cmp": read_csv("reports/block2_morl_comparison_summary.csv"),
        "pareto": read_csv("reports/morl_pareto_front_table.csv"),
        "seed_sum": read_csv("reports/morl_canonical_seedfix_yearly_summary.csv"),
        "seed_per": read_csv("reports/morl_canonical_seedfix_yearly_per_seed.csv"),
        "pi": read_csv("outputs/pi_baseline_15min_yearly/pi_yearly_summary.csv"),
    }


# ---------------------------------------------------------------------------
# Table builders (all data-driven)
# ---------------------------------------------------------------------------

def table_main_kpi(d: dict) -> str:
    arch = d["arch"]
    tr = d["transfer"]
    v35 = arch[arch.variant == "v35_calibrated"].iloc[0]
    rows = []
    for scen, label in [("peak_heat_window", "peak"), ("typical_heat_window", "typical")]:
        pv = _scen_row(d["pure"], scen, controller="thermostatic")
        hy = _scen_row(d["hybrid"], scen)
        dv_viol = tr[(tr.variant == "direct_v35") & (tr.scenario == scen)].iloc[0]["boptest_violation_pct"]
        if scen == "peak_heat_window":
            dv_ms, dv_e, dv_r = v35["peak_control_m_s"], v35["peak_energy_kwh"], v35["peak_transfer_temp_rmse_c"]
        else:
            dv_ms, dv_e, dv_r = v35["typical_control_m_s"], v35["typical_energy_kwh"], v35["typical_transfer_temp_rmse_c"]
        rows.append(f"pure v3 PPO & {label} & {f(pv.m_s)} & {f(pv.violation_pct,2)} & {f(pv.rmse_22_c)} & {f(pv.energy_kwh,1)} \\\\")
        rows.append(f"direct v3.5 PPO & {label} & {f(dv_ms)} & {f(dv_viol,2)} & {f(dv_r)} & {f(dv_e,1)} \\\\")
        rows.append(f"hybrid $\\lambda_T=0.10$ & {label} & {f(hy.m_s)} & {f(hy.violation_pct,2)} & {f(hy.rmse_center_c)} & {f(hy.energy_kwh,1)} \\\\")
    return "\n".join(rows)


def table_warmstart(d: dict) -> str:
    w = d["warm"]
    rows = []
    for mode, mlabel in [("scratch", "scratch (random init)"), ("warmstart", "warm-start (from v3.5)")]:
        for scen, label in [("peak_heat_window", "peak"), ("typical_heat_window", "typical")]:
            r = w[(w["mode"] == mode) & (w["scenario"] == scen)].iloc[0]
            rows.append(f"{mlabel} & {label} & {f(r.m_s)} & {f(r.violation_pct,2)} \\\\")
    return "\n".join(rows)


def table_transfer(d: dict) -> str:
    tr = d["transfer"]
    rows = []
    for variant, vlabel in [("pure_v3", "pure v3"), ("hybrid_l010", "hybrid $\\lambda_T=0.10$"), ("direct_v35", "direct v3.5")]:
        for scen, label in [("peak_heat_window", "peak"), ("typical_heat_window", "typical")]:
            r = tr[(tr.variant == variant) & (tr.scenario == scen)].iloc[0]
            rows.append(f"{vlabel} & {label} & {f(r.temp_rmse_c,3)} & {f(r.ms_gap,3)} & {f(r.action_gap_norm,3)} & {int(r.first_divergence_step)} & {tex_escape(r.top_feature)} \\\\")
    return "\n".join(rows)


def table_hdrl(d: dict) -> str:
    h = d["hdrl"]
    lam = {"l000": "0.00", "l003": "0.03", "l005": "0.05", "l010": "0.10"}
    rows = []
    for scen, label in [("peak_heat_window", "peak"), ("typical_heat_window", "typical")]:
        for v in ["l000", "l003", "l005", "l010"]:
            r = h[(h.variant == v) & (h.scenario == scen)].iloc[0]
            rows.append(f"{lam[v]} & {label} & {f(r.m_s,3)} & {f(r.violation_pct,2)} & {f(r.rmse_center_c)} & {f(r.energy_kwh,1)} \\\\")
    return "\n".join(rows)


def table_morl_5d17d(d: dict):
    rec = d["morl_recon"]
    r5 = rec[rec.variant == "MORL_5D_basic_reconstructed"].iloc[0]
    r17 = d["morl_cmp"][d["morl_cmp"].variant == "MORL_17D_power_only"].iloc[0]
    frozen5 = rec[(rec.variant == "MORL_5D_basic") & (rec.evidence_layer == "historical_frozen")].iloc[0]
    rows = [
        f"MORL 5D (current-code rerun) & 5 & {f(r5.rmse_c)} & {f(r5.violation_pct,1)} & {f(r5.m_s,3)} \\\\",
        f"MORL 17D power-only (canonical) & 17 & {f(r17.rmse_c)} & {f(r17.violation_pct,1)} & {f(r17.m_s,3)} \\\\",
    ]
    return "\n".join(rows), frozen5, r5, r17


def table_morl_pareto_seed(d: dict) -> str:
    p = d["pareto"]
    s = d["seed_sum"]

    def pt(label):
        r = p[p.label == label].iloc[0]
        return r

    p0 = pt("comfort_000_energy_100")
    p25 = pt("comfort_025_energy_075")
    p100 = p[p.label == "comfort_100_energy_000"]
    n50 = s[s.canonical == "comfort_050_energy_050"].iloc[0]
    n75 = s[s.canonical == "comfort_075_energy_025"].iloc[0]
    rows = [
        f"0/100 (seed 42) & {f(p0.rmse_mean)} & {f(p0.violation_pct_mean,2)} & {f(p0.ms_mean,3)} & energy-only collapse \\\\",
        f"25/75 (seed 42) & {f(p25.rmse_mean)} & {f(p25.violation_pct_mean,2)} & {f(p25.ms_mean,3)} & energy-weighted usable \\\\",
        f"50/50 (N=5 mean$\\pm$std) & {f(n50.rmse_mean)}$\\pm${f(n50.rmse_std,3)} & {f(n50.violation_pct_mean,2)}$\\pm${f(n50.violation_pct_std,2)} & {f(n50.ms_mean,3)}$\\pm${f(n50.ms_std,3)} & neutral, CV={f(n50.ms_cv,2)} \\\\",
        f"75/25 (N=5 mean$\\pm$std) & {f(n75.rmse_mean)}$\\pm${f(n75.rmse_std,3)} & {f(n75.violation_pct_mean,2)}$\\pm${f(n75.violation_pct_std,2)} & {f(n75.ms_mean,3)}$\\pm${f(n75.ms_std,3)} & practical, CV={f(n75.ms_cv,2)} \\\\",
    ]
    for lbl, name in [("comfort_080_energy_020", "80/20 (seed 42)"), ("comfort_100_energy_000", "100/0 (seed 42)")]:
        sub = p[p.label == lbl]
        if len(sub):
            r = sub.iloc[0]
            tag = "legacy canonical" if "080" in lbl else "best seed-42 comfort"
            rows.append(f"{name} & {f(r.rmse_mean)} & {f(r.violation_pct_mean,2)} & {f(r.ms_mean,3)} & {tag} \\\\")
    return "\n".join(rows)


def table_morl_per_seed(d: dict) -> str:
    ps = d["seed_per"]
    rows = []
    cmap = {"comfort_050_energy_050": "50/50", "comfort_075_energy_025": "75/25"}
    for canon, clabel in cmap.items():
        for _, r in ps[ps.canonical == canon].iterrows():
            rows.append(f"{clabel} & {int(r.seed)} & {f(r.rmse_mean)} & {f(r.within_1c_pct_mean,1)} & {f(r.violation_pct_mean,2)} & {f(r.energy_kwh_sum,1)} & {f(r.ms_mean,3)} \\\\")
    return "\n".join(rows)


def table_pi(d: dict):
    pi = d["pi"]
    return {
        "rmse": f(pi["rmse"].mean(), 2),
        "mae": f(pi["mae"].mean(), 2),
        "viol": f(pi["viol_pct"].mean(), 1),
        "energy": f(pi["energy_kwh"].mean(), 1),
        "ms": f(pi["ms"].mean(), 3),
    }


def load_env_reward() -> dict:
    import yaml
    with (ROOT / "configs/env.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def table_reward(cfg: dict) -> str:
    m = cfg.get("morl", {})
    c = cfg.get("comfort_shaping", {})
    rows = [
        ("Comfort band", f"{f(m.get('temp_low',21),1)} / {f(m.get('temp_high',24),1)} C", "comfort interval (band\\_low/high)"),
        ("Comfort deadband", f"{f(c.get('deadband_c',0.5),2)} C", "soft margin around band edges"),
        ("In-band bonus", f"{f(c.get('band_bonus',0.05),3)}/step", "reward for staying inside the band"),
        ("Undershoot / overshoot weight", f"{f(c.get('undershoot_weight',1.15),2)} / {f(c.get('overshoot_weight',1.15),2)}", "generic violation penalties"),
        ("Cold-ambient asymmetry", f"{f(c.get('cold_undershoot_weight',1.6),2)} (amb $<{f(c.get('cold_amb_threshold_c',8.0),1)}$ C)", "extra cold-undershoot penalty"),
        ("Hot-ambient asymmetry", f"{f(c.get('hot_overshoot_weight',1.8),2)} (amb $>{f(c.get('hot_amb_threshold_c',24.0),1)}$ C)", "extra hot-overshoot penalty"),
        ("Heating action bonus", f"{f(c.get('heating_action_bonus',0.04),2)} ($T_{{\\mathrm{{sup}}}}\\ge {f(c.get('heating_t_supply_c',29.0),1)}$ C)", "anti-degenerate shaping"),
        ("Cooling action bonus", f"{f(c.get('cooling_action_bonus',0.06),2)} ($T_{{\\mathrm{{sup}}}}\\le {f(c.get('cooling_t_supply_c',21.0),1)}$ C)", "anti-degenerate shaping"),
        ("MORL weights ($w_c/w_e/w_s$)", f"{f(m.get('w_comfort',0.8),2)} / {f(m.get('w_energy',0.2),2)} / {f(m.get('w_safety',0.0),2)}", "canonical scalarization"),
        ("Energy scale", f"${m.get('energy_scale','2e-4')}$ (W$\\to$reward)", "energy-to-reward conversion"),
    ]
    return "\n".join(f"{a} & {b} & {c_} \\\\" for a, b, c_ in rows)


def table_obs17() -> str:
    # Static, verified from envs/tsup_features.py (BASIC=5, TIME=4, FORECAST=5,
    # prev-action=2, delta=1; total 17). obs_mode=extended in configs/env.yaml.
    rows = [
        ("Physical state (basic)", "5", "$T_{\\mathrm{zone}}$, CO$_2$, clipped-log power, prev. $T_{\\mathrm{sup}}$, $T_{\\mathrm{amb}}$"),
        ("Cyclic time", "4", "hour and day sine/cosine encodings"),
        ("Ambient forecast", "5", "$T_{\\mathrm{amb}}$ at $+1,+3,+6,+12,+24$ h"),
        ("Previous action", "2", "$(a_{T_{\\mathrm{sup}}}, a_{\\mathrm{fan}})$ from last step"),
        ("Temperature delta", "1", "causal-smoothed $\\Delta T_{\\mathrm{zone}}$"),
        ("Total (extended)", "17", "obs\\_mode = extended"),
    ]
    return "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)


def table_scenarios(manifest: dict) -> str:
    rows = []
    roles = {"peak_heat_window": "January coldest, heating stress test",
             "typical_heat_window": "February moderate, deployment-realistic"}
    for s in manifest.get("scenarios", []):
        nm = s["name"]
        rows.append(
            f"\\texttt{{{tex_escape(nm)}}} & {int(s['start_day_index'])} & {int(float(s['start_time_sec']))} & "
            f"{int(s['duration_days'])} & {f(s['daily_mean_t_amb_c'],1)} & {roles.get(nm,'')} \\\\")
    rows.append("yearly evaluation & 12 months & --- & 14/mo & varied & MORL + PI yearly summary \\\\")
    return "\n".join(rows)


def fig_hdrl_architecture() -> None:
    """Schematic of the seasonal HDRL hierarchy."""
    fig, ax = plt.subplots(figsize=(11.0, 4.7))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "HDRL: seasonal hierarchical controller", ha="center", fontsize=13, weight="bold", color="#1f2933")
    ax.text(0.5, 0.875, "high-level seasonal gate $k(t)$ routes to one of two low-level PPO setpoint specialists",
            ha="center", fontsize=9.2, style="italic", color=SLATE)
    _box(ax, 0.02, 0.46, 0.17, 0.20, "observation\n$s_t$ (17D)", NAVY, "#eef5fb")
    _box(ax, 0.255, 0.46, 0.17, 0.20, "seasonal gate\n$k(t)$", PURPLE, "#f4f1fa")
    _box(ax, 0.49, 0.60, 0.24, 0.17, "winter PPO\n5M steps, cold-biased shaping", TEAL, "#edf8f7", fs=7.9)
    _box(ax, 0.49, 0.33, 0.24, 0.17, "summer PPO\n7M steps, warm shaping", AMBER, "#fff6ea", fs=7.9)
    _box(ax, 0.78, 0.46, 0.20, 0.20, "supply-temp\naction $a_t$\n$\\to$ BOPTEST", GREEN, "#eef8ee", fs=8.0)
    _arrow(ax, (0.19, 0.56), (0.255, 0.56), SLATE)
    _arrow(ax, (0.425, 0.58), (0.49, 0.66), TEAL)
    _arrow(ax, (0.425, 0.54), (0.49, 0.42), AMBER)
    _arrow(ax, (0.73, 0.66), (0.78, 0.58), TEAL)
    _arrow(ax, (0.73, 0.42), (0.78, 0.54), AMBER)
    ax.text(0.5, 0.115, "$\\pi(a_t\\mid s_t)=\\pi_{k(t)}(a_t\\mid s_t)$: each specialist solves a narrower, better-conditioned control problem.",
            ha="center", fontsize=8.6, color="#374151")
    ax.text(0.5, 0.045, "The low level is already comfort-aware, so the v3.5 temperature censor over-regularizes it; HDRL is best at $\\lambda_T=0$.",
            ha="center", fontsize=8.2, style="italic", color=BURGUNDY)
    _save(fig, "fig_block2_hdrl_architecture")


def fig_morl_pipeline() -> None:
    """Four-stage MORL pipeline schematic."""
    fig, ax = plt.subplots(figsize=(11.4, 4.4))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "MORL: preference-conditioned four-stage pipeline", ha="center", fontsize=13, weight="bold", color="#1f2933")
    ax.text(0.5, 0.875, "17D observation + power-only hybrid backend ($\\lambda_T=0$, $\\lambda_P=5{\\times}10^{-5}$)",
            ha="center", fontsize=9.2, style="italic", color=SLATE)
    boxes = [
        (0.015, "(1) Pretrain", "2M steps\n$w=(0.80,0.20,0)$", NAVY, "#eef5fb"),
        (0.260, "(2) ERAM", "20$\\times$100k iters\n$w_0=(0.34,0.33,0.33)$\n$\\tau_w=0.35$", TEAL, "#edf8f7"),
        (0.510, "(3) Live finetune", "100k steps on BOPTEST\nlr $10^{-4}$, $\\pm3$ d jitter", AMBER, "#fff6ea"),
        (0.760, "(4) Yearly eval", "12 monthly\n14-day windows", GREEN, "#eef8ee"),
    ]
    for x, title, body, col, fc in boxes:
        _box(ax, x, 0.42, 0.225, 0.30, f"{title}\n\n{body}", col, fc, fs=8.0)
    for x in [0.24, 0.49, 0.74]:
        _arrow(ax, (x, 0.57), (x + 0.02, 0.57), SLATE)
    ax.text(0.5, 0.115, "preference weights $w=(w_c,w_e,w_s)$ condition the scalar reward $r=w_c r^{c}+w_e r^{e}+w_s r^{s}$;",
            ha="center", fontsize=8.6, color="#374151")
    ax.text(0.5, 0.045, "MORL is the only family with a live-BOPTEST finetune; PPO/HDRL are evaluated zero-shot.",
            ha="center", fontsize=8.2, style="italic", color=SLATE)
    _save(fig, "fig_block2_morl_pipeline")


def table_ms_decomp(d: dict) -> str:
    """Full data-driven m_s = r_time + r_sev decomposition across all controllers
    (r_time = violation/100; r_sev = m_s - r_time)."""
    arch = d["arch"]; tr = d["transfer"]; hdrl = d["hdrl"]; ss = d["seed_sum"]; pi = d["pi"]
    v35 = arch[arch.variant == "v35_calibrated"].iloc[0]

    def row(name, ms, viol):
        rt = float(viol) / 100.0
        rs = max(float(ms) - rt, 0.0)
        return f"{name} & {f(ms,3)} & {f(rt,3)} & {f(rs,3)} \\\\"

    out = []
    for scen, sl in [("peak_heat_window", "peak"), ("typical_heat_window", "typical")]:
        pv = _scen_row(d["pure"], scen, controller="thermostatic")
        hy = _scen_row(d["hybrid"], scen)
        h0 = hdrl[(hdrl.variant == "l000") & (hdrl.scenario == scen)].iloc[0]
        dv_ms = v35["peak_control_m_s"] if scen == "peak_heat_window" else v35["typical_control_m_s"]
        dv_v = tr[(tr.variant == "direct_v35") & (tr.scenario == scen)].iloc[0]["boptest_violation_pct"]
        out.append(row(f"pure v3 ({sl})", pv.m_s, pv.violation_pct))
        out.append(row(f"direct v3.5 ({sl})", dv_ms, dv_v))
        out.append(row(f"hybrid $\\lambda_T{{=}}0.10$ ({sl})", hy.m_s, hy.violation_pct))
        out.append(row(f"HDRL $\\lambda_T{{=}}0$ ({sl})", h0.m_s, h0.violation_pct))
    n50 = ss[ss.canonical == "comfort_050_energy_050"].iloc[0]
    n75 = ss[ss.canonical == "comfort_075_energy_025"].iloc[0]
    out.append(row("MORL 50/50 (yearly, N=5)", n50.ms_mean, n50.violation_pct_mean))
    out.append(row("MORL 75/25 (yearly, N=5)", n75.ms_mean, n75.violation_pct_mean))
    out.append(row("PI (yearly)", pi["ms"].mean(), pi["viol_pct"].mean()))
    return "\n".join(out)


def table_hypotheses() -> str:
    rows = [
        ("H1", "The higher-fidelity twin (v3.5) used directly as the rollout environment yields the best controller.",
         "\\S4--4.5", "FALSIFIED (direct v3.5 $m_s>1.0$)"),
        ("H2", "Calibrated v3.5 is useful when its role changes from dynamics provider to a frozen reward-shaping censor.",
         "\\S5", "SUPPORTED (hybrid is best)"),
        ("H3", "The thermostatic-optimal censor weight $\\lambda_T=0.10$ transfers to the HDRL family.",
         "\\S6", "FALSIFIED (HDRL best at $\\lambda_T=0$)"),
        ("H4", "MORL viability is determined by the reward scalarization alone.",
         "\\S6.5--7", "FALSIFIED (needs the 17D observation)"),
        ("H5", "The single-seed MORL canonical ($m_s\\approx0.10$) is representative.",
         "\\S8--9", "FALSIFIED at N=5 (best of five; CV $0.42$--$0.61$)"),
    ]
    return "\n".join(f"{a} & {b} & {c} & {dd} \\\\" for a, b, c, dd in rows)


def table_nomenclature() -> str:
    rows = [
        (r"$m_s$", "--", "BOPTEST-style maintenance score (lower is better; combines comfort violation and tracking)"),
        (r"RMSE$_T$", r"\si{\celsius}", "live closed-loop zone-temperature RMSE"),
        (r"Violation", r"\si{\percent}", "fraction of steps outside the 21--24 C comfort band"),
        (r"$\lambda_T,\ \lambda_P$", "--", "hybrid temperature / power disagreement weights"),
        (r"$\Delta m_s$", "--", "live-minus-surrogate $m_s$ transfer gap"),
        (r"$g_a$", "--", "L2 action-gap norm (surrogate vs live)"),
        (r"CV", "--", "coefficient of variation (std/mean) of $m_s$ over seeds"),
        (r"$w_c,w_e,w_s$", "--", "MORL comfort / energy / safety preference weights"),
    ]
    return "\n".join(f"{a} & {b} & {tex_escape(c)} \\\\" for a, b, c in rows)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def write_tex(ctx: dict) -> None:
    tex = rf"""\documentclass[11pt,a4paper]{{article}}

\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{array}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{siunitx}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage{{placeins}}

\geometry{{margin=2.0cm}}
\graphicspath{{{{figures/}}}}
\hypersetup{{colorlinks=true, linkcolor=blue!55!black, citecolor=blue!55!black, urlcolor=blue!55!black}}
\captionsetup{{font=small, labelfont=bf}}

\newcommand{{\Tsupply}}{{\ensuremath{{T_{{\mathrm{{sup}}}}}}}}

\begin{{document}}

\setcounter{{section}}{{4}}
\section{{Controller Learning, Hybrid Regularization, and MORL Seed Stability}}
\label{{sec:results2-control}}

Block 1 established a deliberately asymmetric surrogate result: the compact v3 surrogate is the more useful rollout environment for reinforcement learning (RL), whereas the calibrated v3.5 RC--NeuralODE is the more accurate predictive digital twin. Block 2 tests the controller-side consequence of that asymmetry on the live BOPTEST \texttt{{bestest\_air}} runtime environment. The central question is not whether a controller can be trained on a surrogate, but which functional role each surrogate should play inside the learning loop.

The experiments compare five controller families: pure-v3 thermostatic PPO, direct-v3.5 PPO, hybrid-v3/v3.5 PPO, hierarchical DRL (HDRL), and preference-conditioned MORL. All policies are trained on surrogate backends and then evaluated in closed loop against BOPTEST. One asymmetry must be stated up front: unlike thermostatic PPO and HDRL --- which are evaluated strictly zero-shot after surrogate-only training --- MORL additionally includes a short live-BOPTEST finetuning stage, so MORL is used to probe preference-conditioned robustness and seed stability rather than strict zero-shot surrogate transfer. Two targeted 14-day windows are used for the main thermostatic/HDRL comparison: \texttt{{peak\_heat\_window}} (January, daily-mean ambient $-24.4\,^\circ$C) and \texttt{{typical\_heat\_window}} (February, $+2.4\,^\circ$C). MORL and PI reference values additionally use the 12-month yearly evaluation protocol. This difference is intentional: the targeted windows expose controller-family mechanisms, while the yearly protocol exposes seed stability and preference robustness.

Figure~\ref{{fig:block2_pipeline}} summarizes the pipeline. Direct v3.5 is a negative control, not a failed implementation: it asks whether the highest-fidelity predictive surrogate can be used directly as the policy rollout environment (it cannot). The hybrid backend then asks whether the same physical model becomes useful when its role changes from dynamics provider to reward-shaping censor --- yes for thermostatic PPO, no for HDRL at the same $\lambda_T$, and conditionally yes for MORL once the observation interface is widened from 5D to 17D.

\paragraph{{Roadmap boundary and executed path.}}
Block 2 evaluates which functional role each surrogate should play inside the learning loop, strictly in the order of \texttt{{roadmap.md}} Block 2:
\begin{{enumerate}}
  \item train the pure-v3 thermostatic PPO baseline (\S4);
  \item run the direct-v3.5 warm-start negative control (\S4.5);
  \item run the thermostatic hybrid $\lambda_T$ sweep (\S5) and the three-step transfer-gap diagnostics (\S5.5);
  \item run the HDRL $\lambda_T$ sweep (\S6);
  \item run the MORL 5D$\to$17D observation ablation, Pareto sweep, and N=5 canonical seed analysis (\S6.5--\S9);
  \item benchmark the PI reference and rebuild the Block 2 evidence tables (\S10--\S11).
\end{{enumerate}}
Artifact provenance and rebuild commands for every table and figure are catalogued in roadmap Section 11.1.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.94\linewidth]{{fig_block2_pipeline.pdf}}
  \caption{{Block 2 controller-learning pipeline. The experiments separate the rollout model, the physical regularizer, the controller family, the observation interface, and live BOPTEST validation.}}
  \label{{fig:block2_pipeline}}
\end{{figure}}

\begin{{table}}[H]
\centering
\small
\caption{{Nomenclature for Block 2.}}
\label{{tab:nomenclature2}}
\begin{{tabularx}}{{0.95\linewidth}}{{llX}}
\toprule
Symbol & Unit & Meaning \\
\midrule
{ctx['table_nomenclature']}
\bottomrule
\end{{tabularx}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Targeted-window and yearly scenario definitions (data: roadmap Section 11.1).}}
\label{{tab:scenarios}}
\small
\begin{{tabular}}{{lrrrr>{{\raggedright\arraybackslash}}p{{40mm}}}}
\toprule
Scenario & Day idx & Start (s) & Dur. (d) & Ambient ($^\circ$C) & Role \\
\midrule
{ctx['table_scenarios']}
\bottomrule
\end{{tabular}}
\end{{table}}

Block 2 is structured as a sequence of falsifiable hypotheses, each tied to a roadmap section and resolved against live BOPTEST evidence (Table~\ref{{tab:hypotheses}}). This pre-specification-style framing is what lets the negative results (direct v3.5, HDRL over-regularization, MORL seed variance) be read as decisions rather than tuning accidents.

\begin{{table}}[H]
\centering
\caption{{Block 2 falsifiable-hypothesis ledger. Each hypothesis is tested on the live BOPTEST RTE in the indicated roadmap section.}}
\label{{tab:hypotheses}}
\small
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{5mm}}>{{\raggedright\arraybackslash}}p{{66mm}}l>{{\raggedright\arraybackslash}}p{{40mm}}}}
\toprule
 & Hypothesis & roadmap & Verdict \\
\midrule
{ctx['table_hypotheses']}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{PPO interface, observation design, and reward}}

All Block 2 policies use Proximal Policy Optimization (PPO). The shared PPO settings are learning rate $3\times 10^{{-4}}$ during surrogate pretraining, discount factor $\gamma=0.99$ (an effective horizon of $\sim$25 h at the 900 s step), and ten optimization epochs per rollout. The families differ in rollout length, minibatch size, and total timestep budget because the training scripts were developed per family rather than as a single hyperparameter ablation.

\begin{{table}}[H]
\centering
\caption{{PPO training configuration used in Block 2 (verified from the project training scripts and configuration files).}}
\label{{tab:ppo_hparams}}
\small
\begin{{tabular}}{{lllll}}
\toprule
Parameter & Thermostatic PPO & HDRL & MORL pretrain & MORL finetune \\
\midrule
Algorithm & PPO, MlpPolicy & PPO, MlpPolicy & PPO, MlpPolicy & PPO load+continue \\
Learning rate & $3.0{{\times}}10^{{-4}}$ & $3.0{{\times}}10^{{-4}}$ & $3.0{{\times}}10^{{-4}}$ & $1.0{{\times}}10^{{-4}}$ \\
\texttt{{n\_steps}} & 1024 & 1024 & 2048 & inherited \\
\texttt{{batch\_size}} & 4096 & 2048 & 64 & inherited \\
\texttt{{n\_epochs}} & 10 & 10 & 10 & 10 \\
$\gamma$ & 0.99 & 0.99 & 0.99 & 0.99 \\
GAE $\lambda$ & 0.95 & 0.95 & 0.95 & 0.95 \\
Clip range & 0.20 & 0.20 & 0.20 & 0.20 \\
Total timesteps & 10M & 12M & 2M & 100k \\
Seed & 42 & 42 & 42 & 42--46 (N=5) \\
\bottomrule
\end{{tabular}}
\end{{table}}

\paragraph{{Thermostatic PPO baseline (roadmap \S4).}} Block 2 opens with the control baseline: a single-level thermostatic PPO controller trained purely on the v3 rollout surrogate (the canonical 1-hour-step checkpoint \texttt{{rc\_node\_v3\_tsupply.pt}}) with no v3.5 regularization ($\lambda_T=\lambda_P=0$). The actor and critic are separate MLP heads (Stable-Baselines3 \texttt{{MlpPolicy}}): the actor maps the 17D observation \eqref{{eq:obs17}} to a tanh-squashed continuous supply-temperature command \eqref{{eq:action_map}}, while the critic estimates the state value entering the PPO advantage. Training runs for 10M surrogate steps under the shared settings of Table~\ref{{tab:ppo_hparams}}; with entropy coefficient zero the evaluated policy is deterministic. Crucially, the trained policy is run \emph{{zero-shot}} on the live BOPTEST RTE with no live finetuning, so the baseline is already a strict surrogate-to-simulator transfer test. It is the reference against which the two negative controls (direct v3.5, warm-start) and the hybrid are measured.

The canonical observation interface is the 17-dimensional extended TSup-style vector:
\begin{{equation}}
  s_t =
  \left[
  x^{{\mathrm{{phys}}}}_t,\,
  x^{{\mathrm{{time}}}}_t,\,
  \widehat{{T}}^{{\mathrm{{amb}}}}_{{t+1:t+24h}},\,
  a_{{t-1}},\,
  \Delta T^{{\mathrm{{zone}}}}_t
  \right] \in \mathbb{{R}}^{{17}}.
  \label{{eq:obs17}}
\end{{equation}}
It contains 5 physical states (zone temperature, CO$_2$, clipped-log power, previous supply temperature, ambient), 4 cyclic time features, 5 ambient forecasts ($+1,+3,+6,+12,+24$ h), 2 previous-action terms, and 1 causal-smoothed $\Delta T_{{\mathrm{{zone}}}}$. The failed MORL baseline uses only the earlier 5D observation, which lacks sufficient actuation and forecast context.

\begin{{table}}[H]
\centering
\caption{{Extended 17D observation feature groups (verified from \texttt{{envs/tsup\_features.py}}).}}
\label{{tab:obs17}}
\small
\begin{{tabularx}}{{\linewidth}}{{llX}}
\toprule
Feature group & Dim. & Contents \\
\midrule
{ctx['table_obs17']}
\bottomrule
\end{{tabularx}}
\end{{table}}

The action is a single normalized supply-temperature command,
\begin{{equation}}
  a_t \in [-1,1],
  \qquad
  T^{{\mathrm{{sup}}}}_t = 18 + \tfrac{{a_t+1}}{{2}}(35-18)\quad [^\circ\mathrm{{C}}],
  \label{{eq:action_map}}
\end{{equation}}
with a 1.0 C deadband and a per-step rate limit; the comfort band is $21$--$24\,^\circ$C.

\paragraph{{Evaluation metric.}} The headline maintenance score combines the comfort-violation rate and the worst-case relative severity over a rollout,
\begin{{equation}}
\begin{{aligned}}
  m_s &= r_{{\mathrm{{time}}}} + r_{{\mathrm{{sev}}}},
  \qquad
  r_{{\mathrm{{time}}}} = \frac{{1}}{{N}}\sum_{{t=1}}^{{N}} \mathbf{{1}}\!\left[\,T_t < T_{{\ell}}\ \text{{or}}\ T_t > T_{{h}}\,\right], \\[3pt]
  r_{{\mathrm{{sev}}}} &= \max_{{t}}\, \max\!\left(\frac{{(T_{{\ell}}-T_t)_+}}{{T_{{\ell}}}},\ \frac{{(T_t-T_{{h}})_+}}{{T_{{h}}}}\right).
\end{{aligned}}
\label{{eq:ms}}
\end{{equation}}
with $T_{{\ell}}=21\,^\circ$C and $T_{{h}}=24\,^\circ$C (source: \texttt{{evaluation/benchmark\_bestest\_air\_article7\_style.py}}). Hence $r_{{\mathrm{{time}}}}$ is the fraction of steps outside the band (violation\,$\%=100\,r_{{\mathrm{{time}}}}$) and $r_{{\mathrm{{sev}}}}$ is the single worst relative band exceedance; lower $m_s$ is better. RMSE$_T$ is reported against the band center $T^{{\star}}=22.5\,^\circ$C.

\begin{{table}}[H]
\centering
\caption{{Reward-shaping parameters (verified from \texttt{{configs/env.yaml}}).}}
\label{{tab:reward}}
\small
\begin{{tabularx}}{{\linewidth}}{{llX}}
\toprule
Component & Value & Role \\
\midrule
{ctx['table_reward']}
\bottomrule
\end{{tabularx}}
\end{{table}}

The per-step comfort term is piecewise linear with an asymmetric, ambient-dependent slope (source \texttt{{envs/backends/boptest\_backend.py}}):
\begin{{equation}}
  r^{{\mathrm{{comfort}}}}_t =
  \begin{{cases}}
    -w_u\,(T_{{\ell}} - T_t) + b_{{\mathrm{{heat}}}} & T_t < T_{{\ell}}, \\
    +\beta & T_{{\ell}}+\delta \le T_t \le T_h-\delta, \\
    -w_o\,(T_t - T_h) + b_{{\mathrm{{cool}}}} & T_t > T_h,
  \end{{cases}}
  \label{{eq:comfort}}
\end{{equation}}
where $w_u=1.60$ when $T_{{\mathrm{{amb}}}}\le 8\,^\circ$C (else $1.15$), $w_o=1.80$ when $T_{{\mathrm{{amb}}}}\ge 24\,^\circ$C (else $1.15$), the in-band bonus is $\beta=0.05$, the deadband is $\delta=0.5\,^\circ$C, and $b_{{\mathrm{{heat}}}},b_{{\mathrm{{cool}}}}$ are small action-direction bonuses. The energy term is $r^{{\mathrm{{energy}}}}_t=-\eta P_t$ with $\eta=2\times10^{{-4}}$, and MORL scalarizes the objectives as $r_t = w_c\,r^{{\mathrm{{comfort}}}}_t + w_e\,r^{{\mathrm{{energy}}}}_t + w_s\,r^{{\mathrm{{safety}}}}_t$. The asymmetric cold/hot weights encode that recovering a cold zone in cold weather is physically harder than the symmetric reverse.

\subsection{{Hybrid backend: mathematical role of v3.5}}

The hybrid backend changes the role of the calibrated physical surrogate. Instead of rolling out the policy on v3.5 directly, PPO rolls out on v3 and evaluates frozen v3.5 in parallel on the same state-action pair. The per-step reward is augmented by a disagreement penalty:
\begin{{align}}
  r^{{\mathrm{{hyb}}}}_t
  &= r^{{\mathrm{{comfort}}}}_t + r^{{\mathrm{{smooth}}}}_t + r^{{\mathrm{{energy}}}}_t
  - \lambda_T \left|T^{{v3}}_{{t+1}}-T^{{v3.5}}_{{t+1}}\right|
  - \lambda_P \left|P^{{v3}}_{{t+1}}-P^{{v3.5}}_{{t+1}}\right|.
  \label{{eq:hybrid_reward}}
\end{{align}}
For the canonical thermostatic hybrid, $\lambda_T=0.10$ and $\lambda_P=5.0\times 10^{{-5}}$; PPO otherwise computes the advantage $A_t = r^{{\mathrm{{hyb}}}}_t + \gamma V(s_{{t+1}}) - V(s_t)$ unchanged. Thus v3.5 is neither a second policy loss nor a direct dynamics model: it is a frozen physics-informed censor that discourages the policy from entering state-action regions where the smooth v3 rollout and the calibrated physical model disagree. Across the canonical hybrid traces (\texttt{{reports/hybrid\_disagreement\_summary.csv}}, overall), the mean temperature disagreement is ${ctx['dis_temp_mean']}\,^\circ$C (p95 ${ctx['dis_temp_p95']}\,^\circ$C) and the mean power disagreement is ${ctx['dis_pow_mean']}$ W (p95 ${ctx['dis_pow_p95']}$ W) --- large enough to shape learning, bounded enough to stay informative.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.90\linewidth]{{fig_block2_reward_shaping.pdf}}
  \caption{{Hybrid reward-shaping mechanism. v3 supplies rollout dynamics; frozen v3.5 supplies a disagreement penalty on the same state-action transition.}}
  \label{{fig:hybrid_reward}}
\end{{figure}}

\subsection{{Thermostatic PPO: direct-v3.5 failure and hybrid success (roadmap \S4--\S5)}}

Table~\ref{{tab:main_kpi}} and Figure~\ref{{fig:live_kpi}} summarize the main Block 2 controller result. Pure v3 PPO is already usable ($m_s={ctx['pure_peak_ms']}$ peak, ${ctx['pure_typ_ms']}$ typical). Direct v3.5 PPO fails catastrophically despite v3.5's superior Block 1 predictive fidelity: live violation reaches {ctx['dv_peak_viol']}\% on peak and {ctx['dv_typ_viol']}\% on typical, with RMSE above $4.3\,^\circ$C. The hybrid backend resolves the conflict: $m_s={ctx['hyb_peak_ms']}$ on peak and ${ctx['hyb_typ_ms']}$ on typical, with violation below $5\%$ on both windows and lower energy than pure v3 on the peak window.

\begin{{table}}[H]
\centering
\caption{{Canonical Block 2 live BOPTEST controller comparison on the targeted 14-day windows. Data sources for every row are catalogued in roadmap Section 11.1.}}
\label{{tab:main_kpi}}
\small
\begin{{tabular}}{{llrrrr}}
\toprule
Policy/backend & Scenario & $m_s$ & Violation (\%) & RMSE$_T$ ($^\circ$C) & Energy (kWh) \\
\midrule
{ctx['table_main_kpi']}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.96\linewidth]{{final17_fig06_live_boptest_controller_comparison.pdf}}
  \caption{{Live BOPTEST KPI comparison for pure v3, direct v3.5, and hybrid PPO. Direct v3.5 is a negative control: higher predictive fidelity does not imply a useful RL training environment.}}
  \label{{fig:live_kpi}}
\end{{figure}}

\paragraph{{Reading $m_s$ step by step.}} The decomposition $m_s=r_{{\mathrm{{time}}}}+r_{{\mathrm{{sev}}}}$ (Eq.~\ref{{eq:ms}}) exposes the failure structure directly (Figure~\ref{{fig:ms_decomp}}). Direct v3.5's peak $m_s=1.046$ is dominated by $r_{{\mathrm{{time}}}}\approx 0.77$ (out of band 77\% of the time) plus a large worst-case severity $r_{{\mathrm{{sev}}}}\approx 0.28$; the hybrid's peak $m_s=0.087$ is almost entirely a small $r_{{\mathrm{{sev}}}}$ with $r_{{\mathrm{{time}}}}<0.05$. The hybrid therefore does not merely lower the average error --- it removes the sustained band departures that dominate the direct-v3.5 score.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.82\linewidth]{{fig_block2_ms_decomposition.pdf}}
  \caption{{Data-driven decomposition $m_s=r_{{\mathrm{{time}}}}+r_{{\mathrm{{sev}}}}$ on the live BOPTEST windows. Direct v3.5 fails through sustained violation ($r_{{\mathrm{{time}}}}$); pure v3 and the hybrid keep both terms small.}}
  \label{{fig:ms_decomp}}
\end{{figure}}

The time-series and action diagnostics in Figures~\ref{{fig:closed_loop_traces}} and~\ref{{fig:action_phase}} explain the mechanism. Direct v3.5 learns a bang-bang-like control law that drives the live simulator outside the comfort band; in phase space it places extreme actions in temperature-error regimes the live building does not support. We note explicitly that this destabilization mechanism is \emph{{hypothesized}} (higher advantage-estimator variance under sharper surrogate predictions, and/or overfitting to sub-step physical structure unusable at the 15-min cadence) and is not directly measured here; discriminating the two is deferred to future work.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.96\linewidth]{{block2_q1_polish_closed_loop_disturbance.pdf}}
  \caption{{Closed-loop BOPTEST traces with ambient disturbance, comfort band, and physical actuator limits.}}
  \label{{fig:closed_loop_traces}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.96\linewidth]{{block2_q1_polish_phase_density.pdf}}
  \caption{{Phase portrait as empirical state-action density. The saturation bands near $a_0=\pm1$ expose the bang-bang behavior of direct v3.5 PPO.}}
  \label{{fig:action_phase}}
\end{{figure}}

\paragraph{{Hybrid $\lambda_T$ sweep and canonical selection (roadmap \S5).}} The canonical hybrid operating point was selected by a thermostatic sweep over the temperature-disagreement weight $\lambda_T\in\{{0.05,0.10,0.15\}}$ at fixed $\lambda_P=5\times10^{{-5}}$ (Table~\ref{{tab:hybrid_sweep}}). The mid setting $\lambda_T=0.10$ (\texttt{{hybrid\_l010}}) is canonical: per the roadmap selection rule it retains the energy advantage over pure v3 while avoiding the stronger comfort degradation seen at the weaker ($0.05$) and stronger ($0.15$) censor settings. Only the canonical point's live-BOPTEST KPIs are retained as a frozen artifact; the bracketing points served selection only.

\begin{{table}}[H]
\centering
\caption{{Thermostatic hybrid $\lambda_T$ sweep design (roadmap \S5; $\lambda_P=5\times10^{{-5}}$ throughout). The mid point is the retained canonical.}}
\label{{tab:hybrid_sweep}}
\small
\begin{{tabular}}{{lll}}
\toprule
$\lambda_T$ & Tag & Role \\
\midrule
0.05 & hybrid\_l005 & weaker censor (sweep bracket) \\
0.10 & hybrid\_l010 & \textbf{{canonical}} (retained; $m_s={ctx['hyb_peak_ms']}$ peak, ${ctx['hyb_typ_ms']}$ typical) \\
0.15 & hybrid\_l015 & stronger censor (sweep bracket) \\
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Warm-start negative control (roadmap \S4.5)}}

A second negative control tests whether v3.5 is useful as a policy \emph{{initializer}} rather than a reward-shaping censor. The benchmark (\texttt{{run\_block2.py warmstart}}) runs four steps: \emph{{(1) pretrain}} a thermostatic policy on the calibrated v3.5 surrogate; \emph{{(2) scratch finetune}} a thermostatic policy from random initialization directly on the live BOPTEST RTE; \emph{{(3) warm-start finetune}} a second policy on BOPTEST initialized from the v3.5-pretrained checkpoint of step~1; and \emph{{(4) evaluate}} both finetuned policies on the same BOPTEST windows and compare. The comparison is therefore internally controlled --- scratch and warm-start differ only in initialization. It does not help: warm-started policies are markedly worse than the scratch finetune (Table~\ref{{tab:warmstart}}), raising $m_s$ by roughly two to three times on both windows. The problem is the role assigned to the surrogate during early policy formation, not a lack of fine-tuning; v3.5 belongs in the reward as a censor, not in the weights as an initializer.

\begin{{table}}[H]
\centering
\caption{{Direct-v3.5 warm-start utility (\texttt{{outputs/block2\_thermostatic\_warmstart\_utility/comparison\_summary.csv}}).}}
\label{{tab:warmstart}}
\small
\begin{{tabular}}{{llrr}}
\toprule
Mode & Scenario & $m_s$ & Violation (\%) \\
\midrule
{ctx['table_warmstart']}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.82\linewidth]{{block2_warmstart_negative_eval_kpis.pdf}}
  \caption{{Warm-start negative control. Pretraining on direct v3.5 and then fine-tuning on the hybrid backend is inferior to training the hybrid policy from scratch.}}
  \label{{fig:warmstart}}
\end{{figure}}

\subsection{{Transfer-gap diagnostics (roadmap \S5.5)}}

This diagnostic (roadmap \S5.5) compares three controllers --- pure v3, hybrid $\lambda_T=0.10$, and direct v3.5 --- in three steps. \emph{{Step 1}}: a standalone direct-v3.5 thermostatic policy is trained on the calibrated v3.5 environment (the failure control; \texttt{{thermostatic-train --variant v35\_direct}}), separate from the warm-start checkpoint of the previous section because it answers a different question (zero-shot live transfer rather than warm-start utility). \emph{{Step 2}}: all three policies are run in closed loop on the live BOPTEST RTE for the two 14-day windows (\texttt{{thermostatic-transfer --variant all}}). \emph{{Step 3}}: for each policy we compute four transfer diagnostics (\texttt{{thermostatic-diagnose}}) --- the temperature tracking gap (live RMSE$_T$), the surrogate-vs-live maintenance-score mismatch
\begin{{equation}}
\begin{{aligned}}
  \Delta m_s &= m_s^{{\mathrm{{surrogate}}}} - m_s^{{\mathrm{{BOPTEST}}}}
  \qquad(\text{{negative}}\Rightarrow\text{{surrogate optimistic}}), \\[3pt]
  g_a &= \frac{{1}}{{N}}\sum_{{t=1}}^{{N}}\big\|\,a_t^{{\mathrm{{BOPTEST}}}}-a_t^{{\mathrm{{surrogate}}}}\,\big\|_{{2}} .
\end{{aligned}}
\label{{eq:transfer_gap}}
\end{{equation}}
the action-gap norm $g_a$, and the first divergence step (earliest 15-min step where surrogate and live trajectories separate beyond threshold), with the top driver feature attributed to the divergence.

Direct v3.5 has the largest transfer mismatch ($|\Delta m_s|\approx 0.9$--$1.0$, $g_a\approx 2.0$, live RMSE$_T>4.3\,^\circ$C), and its top divergence driver is $t_{{\mathrm{{zone}}}}$: its sharp temperature dynamics produce live actions unlike the surrogate rollout. Hybrid $\lambda_T=0.10$ has the smallest gap ($|\Delta m_s|\approx 0.02$) and, on the typical window, holds the BOPTEST-consistent action for 16 steps before drifting (Table~\ref{{tab:transfer}}). The action-gap norm is the most diagnostic column: $g_a=2.0$ for direct v3.5 means the policy sits at the opposite action bound from what BOPTEST would choose throughout the episode, i.e.\ a learned bang-bang exploit of the surrogate.

\begin{{table}}[H]
\centering
\caption{{Transfer diagnostics across three backends. $\Delta m_s=$ surrogate $-$ live (negative $\Rightarrow$ surrogate optimistic); Temp gap is the live closed-loop RMSE$_T$ (data: roadmap Section 11.1).}}
\label{{tab:transfer}}
\small
\begin{{tabular}}{{llrrrrl}}
\toprule
Variant & Scenario & Temp gap ($^\circ$C) & $\Delta m_s$ & Action gap & First div. & Top driver \\
\midrule
{ctx['table_transfer']}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.86\linewidth]{{block1_q1_fig10_transfer_gap_diagnostics.pdf}}
  \caption{{Transfer-gap diagnostics. Direct v3.5 has high action-gap norm and live-surrogate mismatch; the hybrid backend suppresses both.}}
  \label{{fig:transfer_gap}}
\end{{figure}}

\subsection{{HDRL sensitivity: the hybrid weight is controller-family specific (roadmap \S6)}}

HDRL here is a \emph{{seasonal}} hierarchy. A high-level seasonal selector routes control to one of two low-level PPO setpoint specialists --- a winter agent (trained 5M steps on cold-season day ranges with cold-biased comfort shaping) and a summer agent (7M steps on warm-season ranges) --- each acting on the same 17D observation and supply-temperature action. Formally the controller is $\pi(a_t\mid s_t)=\pi_{{k(t)}}(a_t\mid s_t)$ with the high-level gate $k(t)\in\{{\text{{winter}},\text{{summer}}\}}$ chosen by season; the hierarchy supplies regime-aware temporal abstraction, so each low-level specialist solves a narrower, better-conditioned control problem than a single global policy (Figure~\ref{{fig:hdrl_arch}}).

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.96\linewidth]{{fig_block2_hdrl_architecture.pdf}}
  \caption{{HDRL architecture: a high-level seasonal gate $k(t)$ routes the 17D observation to one of two low-level PPO setpoint specialists (winter / summer). Because the low level is already comfort-aware, the v3.5 temperature censor over-regularizes it.}}
  \label{{fig:hdrl_arch}}
\end{{figure}}

The HDRL experiment asks whether the thermostatic hybrid setting $\lambda_T=0.10$ transfers to this hierarchical controller. It does not (Table~\ref{{tab:hdrl}}). The mechanism is over-regularization: because each seasonal specialist already encodes comfort-aware structure (season-tuned shaping plus the high-level gate), the additional v3.5 temperature-disagreement censor constrains an already comfort-aware low-level loop, biasing it toward conservative under-heating; performance therefore degrades monotonically with $\lambda_T$, and the correct transfer keeps only the power channel ($\lambda_T=0$). HDRL performs best at $\lambda_T=0.00$ on both windows and degrades monotonically as temperature-disagreement regularization is increased. This shows the correct physical-censor strength depends on the controller family and its action decomposition, not on a universal weight.

\begin{{table}}[H]
\centering
\caption{{HDRL sweep over $\lambda_T$ on the targeted windows; best is $\lambda_T=0$, and performance degrades monotonically as the temperature censor strengthens (data: roadmap Section 11.1).}}
\label{{tab:hdrl}}
\small
\begin{{tabular}}{{llrrrr}}
\toprule
$\lambda_T$ & Scenario & $m_s$ & Violation (\%) & RMSE$_T$ ($^\circ$C) & Energy (kWh) \\
\midrule
{ctx['table_hdrl']}
\bottomrule
\end{{tabular}}
\end{{table}}

\paragraph{{Engineering reading.}} As $\lambda_T$ rises, energy falls slightly (the censor biases the low-level loop toward conservative under-heating, e.g. $329.6\to300.4$ kWh on peak) but comfort collapses --- peak violation roughly quadruples ($6.1\%\to23.0\%$) and RMSE$_T$ nearly doubles. HDRL already supplies comfort-aware temporal abstraction through its scheduler, so an additional temperature-disagreement censor is redundant and over-constrains the setpoint controller; the correct transfer of the Block 1 recipe to HDRL keeps only the power channel ($\lambda_T=0$, $\lambda_P=5\times10^{{-5}}$).

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.88\linewidth]{{block2_hdrl_lambda_sweep_sensitivity.pdf}}
  \caption{{HDRL $\lambda_T$ sensitivity. The thermostatic regularization weight does not transfer; the best HDRL policy uses no temperature-disagreement term.}}
  \label{{fig:hdrl_sweep}}
\end{{figure}}

\subsection{{MORL observation ablation: 5D failure to 17D success (roadmap \S6.5--\S7)}}

MORL uses a four-stage pipeline that differs from the single-stage PPO families (roadmap Section 7): (1) a 2M-step surrogate \emph{{pretrain}} on the 17D hybrid backend with canonical $(w_c,w_e,w_s)=(0.80,0.20,0.00)$; (2) an \emph{{ERAM}} weight-adaptation stage (20 iterations of 100k steps from initial weights $0.34/0.33/0.33$, weight-update temperature $\tau_w=0.35$); (3) a 100k-step \emph{{finetune}} on the live BOPTEST RTE at learning rate $10^{{-4}}$ with $\pm3$-day episode-start jitter; and (4) a 12-month \emph{{yearly evaluation}}. MORL is the only family with a live-BOPTEST finetune; the thermostatic/HDRL families are evaluated zero-shot after surrogate-only training, which is a strictly harder transfer (Figure~\ref{{fig:morl_pipeline}}).

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.98\linewidth]{{fig_block2_morl_pipeline.pdf}}
  \caption{{MORL four-stage pipeline: surrogate pretrain $\to$ ERAM preference adaptation $\to$ live-BOPTEST finetune $\to$ yearly evaluation, all on the 17D power-only hybrid backend. The preference vector conditions the scalarized reward.}}
  \label{{fig:morl_pipeline}}
\end{{figure}}

MORL initially failed with a 5D observation (zone temperature, ambient, hour, day, occupancy). Under the current code path, a reconstructed 5D rerun obtains RMSE$_T={ctx['m5_rmse']}\,^\circ$C, violation ${ctx['m5_viol']}\%$, and $m_s={ctx['m5_ms']}$; the originally frozen 5D artifact was even worse (RMSE$_T={ctx['m5frozen_rmse']}\,^\circ$C, $m_s={ctx['m5frozen_ms']}$) and is retained only as an audit artifact. Replacing the observation with the 17D TSup-style vector recovers a usable policy: RMSE$_T={ctx['m17_rmse']}\,^\circ$C, violation ${ctx['m17_viol']}\%$, $m_s={ctx['m17_ms']}$ (Table~\ref{{tab:morl5d17d}}). The dominant MORL bottleneck was the observation geometry, not the reward scalarization alone.

\begin{{table}}[H]
\centering
\caption{{MORL observation-interface ablation. The backend is hybrid in both cases; only the observation interface changes. The current-code reconstructed 5D rerun is the reproducible main-paper evidence (roadmap Section 11.1).}}
\label{{tab:morl5d17d}}
\small
\begin{{tabular}}{{lrrrr}}
\toprule
Variant & Obs dim & RMSE$_T$ ($^\circ$C) & Violation (\%) & $m_s$ \\
\midrule
{ctx['table_morl_5d17d']}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.84\linewidth]{{final17_fig09_morl_5d_failure_17d_success.pdf}}
  \caption{{MORL 5D failure and 17D recovery. The observation interface, not only the scalarized reward, determines whether MORL is viable.}}
  \label{{fig:morl5d17d}}
\end{{figure}}

\paragraph{{Power-only backend and claim discipline (roadmap \S7, \S9, \S13).}} MORL inherits the controller-family lesson of the HDRL sweep: it runs on the \emph{{power-only}} hybrid backend ($\lambda_T=0$, $\lambda_P=5\times10^{{-5}}$), because temperature-disagreement censoring hurts non-thermostatic families. The MORL claim is deliberately narrow and audit-protected. Before the seed expansion, commit \texttt{{93df9b3}} froze the canonical plan and the seed-45/46 falsification predictions; after $N=5$, commit \texttt{{62dc859}} appended the observed variance without rewriting the pre-specification, and a replay test confirmed bit-identical BOPTEST trajectories for a fixed checkpoint (so the spread is training stochasticity, not simulator noise). The defensible reading is therefore: under the fixed 17D power-only backend, MORL is substantially stronger than the reconstructed 5D interface and yields a usable comfort--energy Pareto structure, but the $N=5$ canonical variance (CV $0.42$--$0.61$) is too high for a deployment-stability claim --- the single-seed canonical is the best of five, not the median.

\subsection{{MORL Pareto front and N=5 seed variance (roadmap \S8--\S9)}}

The MORL Pareto sweep varies comfort/energy weights with safety weight zero (Table~\ref{{tab:morl_pareto_seed}}). Energy-only control collapses comfort ($m_s={ctx['p0_ms']}$, violation ${ctx['p0_viol']}\%$); comfort-leaning settings are far more stable. The single-seed (seed 42) Pareto points are reported separately from the $N=5$ canonical extensions, because the seed analysis is the central audit result: the neutral 50/50 canonical has $m_s={ctx['n50_ms']}\pm{ctx['n50_std']}$ (CV ${ctx['n50_cv']}$, 95\% $t$-CI $[{ctx['n50_ci_lo']},{ctx['n50_ci_hi']}]$) and the practical 75/25 canonical has $m_s={ctx['n75_ms']}\pm{ctx['n75_std']}$ (CV ${ctx['n75_cv']}$, 95\% $t$-CI $[{ctx['n75_ci_lo']},{ctx['n75_ci_hi']}]$). Seed 46 is an outlier in both groups (Table~\ref{{tab:morl_per_seed}}). Because the replay audit produced bit-identical BOPTEST trajectories for a fixed policy, this variance is attributed to PPO/ERAM training stochasticity, not simulator noise. The single-seed canonical ($m_s\approx0.10$) is therefore the \emph{{best}} of five, not the median.

\begin{{table}}[H]
\centering
\caption{{MORL Pareto (seed 42) and N=5 canonical seed analysis (data: roadmap Section 11.1).}}
\label{{tab:morl_pareto_seed}}
\small
\begin{{tabular}}{{lrrll}}
\toprule
Preference / statistic & RMSE$_T$ ($^\circ$C) & Violation (\%) & $m_s$ & Interpretation \\
\midrule
{ctx['table_morl_pareto_seed']}
\bottomrule
\end{{tabular}}
\end{{table}}

\paragraph{{Engineering reading.}} The front is strongly asymmetric. Energy-only control (0/100) collapses to a degenerate non-heating policy ($86.8\%$ violation, $m_s={ctx['p0_ms']}$): minimizing energy simply means not heating. Comfort-only (100/0) is the best single-seed $m_s$ but the most energy-hungry, and the practical 75/25 point recovers near-comfort-only comfort at lower energy --- the recommended deployment compromise. The crucial caveat is the seed dimension: the single-seed front overstates reliability, because the $N=5$ extension turns both canonicals into wide distributions (CV ${ctx['n50_cv']}$--${ctx['n75_cv']}$), so the Pareto curve should be read as a best-seed envelope, not a deployment guarantee.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.86\linewidth]{{block2_q1_polish_morl_pareto_ellipses.pdf}}
  \caption{{MORL comfort--energy Pareto front with N=5 confidence ellipses for the two canonical points. Non-canonical points are seed-42 only.}}
  \label{{fig:morl_pareto}}
\end{{figure}}

\begin{{table}}[H]
\centering
\caption{{Per-seed MORL yearly metrics for the two canonical weight pairs (data: roadmap Section 11.1).}}
\label{{tab:morl_per_seed}}
\small
\begin{{tabular}}{{lrrrrrr}}
\toprule
Pair (c/e) & Seed & RMSE$_T$ & Within 1$^\circ$C (\%) & Violation (\%) & Energy (kWh) & $m_s$ \\
\midrule
{ctx['table_morl_per_seed']}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Seasonal variance falsification (roadmap \S9; audit \S13)}}

At $N=3$, the practical canonical appeared to show near-deterministic winter behavior and a seasonal inversion relative to the neutral canonical, motivating a pre-specified falsification test before seeds 45 and 46 were trained. The direction-specific predictions (February winter $\sigma(m_s)<0.005$; June summer $\sigma(m_s)>0.05$; winter neutral/practical variance ratio $>20$) failed at $N=5$: February practical $\sigma(m_s)$ rose to order $0.17$ and the winter variance ratio collapsed to order one. This is a success of the pre-specification protocol (audit anchors \texttt{{93df9b3}} pre-specification, \texttt{{62dc859}} post-N=5 falsification), not a project failure. The honest, narrower conclusion: MORL is promising in mean performance, especially at comfort-leaning preferences, but is \emph{{not}} deployment-stable without explicit stabilization (validation-based checkpoint selection, early stopping, or ensemble selection), which is left as future work because the canonical protocol fixes final-epoch evaluation.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.88\linewidth]{{block2_morl_17d_seasonal_heatmap.pdf}}
  \caption{{MORL seasonal performance heatmap under the 17D interface. The figure shows monthly structure but does not support the N=3 mechanism claim after N=5 extension.}}
  \label{{fig:morl_heatmap}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.88\linewidth]{{block2_morl_seasonal_variance_inversion.pdf}}
  \caption{{Post-N=5 seasonal variance diagnostic. The earlier seasonal-inversion hypothesis is reported as falsified, not retained as a positive mechanism claim.}}
  \label{{fig:seasonal_falsification}}
\end{{figure}}

\subsection{{PI reference and synthesis (roadmap \S10--\S11)}}

The BOPTEST built-in PI controller is a reproducible reference, not a tuned strong baseline. On the 12-month yearly evaluation it is comfort-poor but energy-frugal: mean RMSE$_T={ctx['pi_rmse']}\,^\circ$C, mean violation ${ctx['pi_viol']}\%$, mean monthly energy ${ctx['pi_energy']}$ kWh, mean $m_s={ctx['pi_ms']}$. All Block 2 RL/MORL agents reach $m_s$ well below PI's ${ctx['pi_ms']}$, dominating it on comfort at comparable or lower energy after accounting for the comfort/energy trade-off.

Table~\ref{{tab:ms_decomp}} consolidates the maintenance-score decomposition across every controller family. It makes the cross-family pattern explicit: usable controllers (pure v3, hybrid, MORL 17D) keep both $r_{{\mathrm{{time}}}}$ and $r_{{\mathrm{{sev}}}}$ small, whereas the failures (direct v3.5, PI) carry a large $r_{{\mathrm{{time}}}}$ --- they spend a large fraction of the horizon outside the band. PI is the mirror image of direct v3.5: both have $r_{{\mathrm{{time}}}}$ near or above $0.6$, but PI fails by chronic under-heating while direct v3.5 fails by saturated overshoot.

\begin{{table}}[H]
\centering
\caption{{Cross-controller maintenance-score decomposition $m_s = r_{{\mathrm{{time}}}} + r_{{\mathrm{{sev}}}}$ (Eq.~\ref{{eq:ms}}). $r_{{\mathrm{{time}}}}$ = fraction of steps outside the band; $r_{{\mathrm{{sev}}}}$ = worst relative band exceedance. Thermostatic/HDRL rows are 14-day windows; MORL/PI rows are 12-month yearly means.}}
\label{{tab:ms_decomp}}
\small
\begin{{tabular}}{{lrrr}}
\toprule
Controller (window/eval) & $m_s$ & $r_{{\mathrm{{time}}}}$ & $r_{{\mathrm{{sev}}}}$ \\
\midrule
{ctx['table_ms_decomp']}
\bottomrule
\end{{tabular}}
\end{{table}}

Block 2 establishes four controller-side claims. First, predictive fidelity and RL training utility are not equivalent: direct v3.5 is the more accurate twin yet fails as a rollout environment (live violation $>77\%$). Second, role separation works --- v3 provides smooth rollout dynamics while frozen v3.5 acts as a physical censor through disagreement shaping --- giving the canonical hybrid ($m_s={ctx['hyb_peak_ms']}$ peak, ${ctx['hyb_typ_ms']}$ typical). Third, the censor strength is controller-family specific: HDRL rejects $\lambda_T=0.10$ and is best at $\lambda_T=0$. Fourth, MORL is viable only with the 17D interface, and its N=5 analysis reveals high seed variance and falsifies the N=3 seasonal-inversion mechanism. The engineering implication is that hybrid surrogate RL is a role-allocation problem --- rollout smoothness, physical censoring, observation geometry, controller family, and seed stabilization are separate design axes. This is the bridge to Block 3, where the fixed \texttt{{bestest\_air}} recipe is transferred to the hydronic BOPTEST family.

\subsection{{Limitations}}

\begin{{itemize}}
  \item \textbf{{Mechanism not measured.}} The direct-v3.5 destabilization is established only in \emph{{direction}}; the gradient-variance vs sub-step-overfit hypotheses are not discriminated here.
  \item \textbf{{Single testcase.}} All Block 2 results are on \texttt{{bestest\_air}}; cross-building transfer is Block 3.
  \item \textbf{{MORL seed stability.}} The canonical MORL claim is narrowed: N=5 CV is ${ctx['n50_cv']}$--${ctx['n75_cv']}$; the single-seed canonical is the best of five and final-epoch evaluation is fixed by protocol.
  \item \textbf{{Per-family PPO hyperparameters differ}} (rollout length / batch size / budget); they were set per training script, so cross-family KPI differences are not a controlled hyperparameter ablation.
  \item \textbf{{MORL alone uses live BOPTEST finetuning}} (100k steps); thermostatic/HDRL are evaluated zero-shot after surrogate-only training, a strictly harder transfer.
\end{{itemize}}

Artifact provenance (which \texttt{{reports/}} and \texttt{{outputs/}} files back every table and figure) and the rebuild commands are documented in \texttt{{roadmap.md}} Section 11.1; this section is regenerated by \texttt{{build\_results2\_overleaf.py}}.

\subsection{{Results II conclusion}}

Block 2 converts the Block 1 surrogate asymmetry into a controller-design principle. The five hypotheses of Table~\ref{{tab:hypotheses}} resolve into one coherent recipe: \emph{{v3 is always the rollout dynamics, v3.5 is always a frozen physical censor, and what changes across controller families is which channels of the v3.5 prediction are censored}}. Direct use of the higher-fidelity twin as the rollout environment fails (live violation $>77\%$, $g_a\approx2.0$); using it as a reward-shaping censor succeeds for thermostatic PPO at $\lambda_T=0.10$ ($m_s={ctx['hyb_peak_ms']}$/${ctx['hyb_typ_ms']}$); the same temperature censor over-regularizes the comfort-aware HDRL hierarchy and must be switched off ($\lambda_T=0$); and MORL is recovered only by widening the observation interface to 17D, after which a power-only censor and a comfort-leaning preference give the best mean behaviour --- but with an audit-protected, deliberately narrowed claim because the $N=5$ canonical variance (CV ${ctx['n50_cv']}$--${ctx['n75_cv']}$) is too high for a deployment-stability statement.

The maintenance-score decomposition (Table~\ref{{tab:ms_decomp}}) gives the unifying physical reading: usable controllers keep both the violation-time term $r_{{\mathrm{{time}}}}$ and the worst-case severity $r_{{\mathrm{{sev}}}}$ small, whereas the two failure modes are mirror images --- direct v3.5 fails by saturated overshoot and PI by chronic under-heating, both with $r_{{\mathrm{{time}}}}$ near or above $0.6$. Engineering-wise, hybrid surrogate RL is therefore not a single architecture but a \emph{{role-allocation}} problem across five separable axes: rollout smoothness, physical censoring, observation geometry, controller family, and seed stabilization. This recipe --- the frozen \texttt{{hybrid\_l010}} thermostatic policy and the MORL canonical --- is the artifact carried into Block 3 (Results III), where it is transferred, with documented adapters, from \texttt{{bestest\_air}} to the hydronic BOPTEST family.

\end{{document}}
"""
    (BASE / "main.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    d = load_block2()
    kpi = table_main_kpi(d)
    morl_5d17d, frozen5, r5, r17 = table_morl_5d17d(d)
    pi = table_pi(d)
    n50 = d["seed_sum"][d["seed_sum"].canonical == "comfort_050_energy_050"].iloc[0]
    n75 = d["seed_sum"][d["seed_sum"].canonical == "comfort_075_energy_025"].iloc[0]
    p0 = d["pareto"][d["pareto"].label == "comfort_000_energy_100"].iloc[0]

    # Q1 additions: reward config, scenario manifest, disagreement stats, N=5 CI.
    import json
    import math
    try:
        reward_tbl = table_reward(load_env_reward())
    except Exception as exc:
        print(f"[warn] reward table fallback: {exc}")
        reward_tbl = table_reward({})
    try:
        manifest = json.loads((ROOT / "outputs/bestest_air_article7_style_15min/scenario_manifest.json").read_text(encoding="utf-8"))
        scen_tbl = table_scenarios(manifest)
    except Exception as exc:
        print(f"[warn] scenario table fallback: {exc}")
        scen_tbl = ""
    dis = read_csv("reports/hybrid_disagreement_summary.csv")
    dov = dis[dis.scenario == "overall"].iloc[0]
    # 95% t-CI (n=5, t_{0.975,4}=2.776) on m_s for the two canonicals.
    tcrit = 2.776
    n50 = d["seed_sum"][d["seed_sum"].canonical == "comfort_050_energy_050"].iloc[0]
    n75 = d["seed_sum"][d["seed_sum"].canonical == "comfort_075_energy_025"].iloc[0]
    ci50 = tcrit * float(n50.ms_std) / math.sqrt(5)
    ci75 = tcrit * float(n75.ms_std) / math.sqrt(5)

    pure_peak = _scen_row(d["pure"], "peak_heat_window", controller="thermostatic")
    pure_typ = _scen_row(d["pure"], "typical_heat_window", controller="thermostatic")
    hyb_peak = _scen_row(d["hybrid"], "peak_heat_window")
    hyb_typ = _scen_row(d["hybrid"], "typical_heat_window")
    tr = d["transfer"]
    dv_peak_v = tr[(tr.variant == "direct_v35") & (tr.scenario == "peak_heat_window")].iloc[0]["boptest_violation_pct"]
    dv_typ_v = tr[(tr.variant == "direct_v35") & (tr.scenario == "typical_heat_window")].iloc[0]["boptest_violation_pct"]

    ctx = {
        "table_nomenclature": table_nomenclature(),
        "table_ms_decomp": table_ms_decomp(d),
        "table_hypotheses": table_hypotheses(),
        "table_reward": reward_tbl,
        "table_obs17": table_obs17(),
        "table_scenarios": scen_tbl,
        "dis_temp_mean": f(dov.temp_disagree_mean_c, 3),
        "dis_temp_p95": f(dov.temp_disagree_p95_c, 2),
        "dis_pow_mean": f(dov.power_disagree_mean_w, 0),
        "dis_pow_p95": f(dov.power_disagree_p95_w, 0),
        "n50_ci_lo": f(float(n50.ms_mean) - ci50, 3), "n50_ci_hi": f(float(n50.ms_mean) + ci50, 3),
        "n75_ci_lo": f(float(n75.ms_mean) - ci75, 3), "n75_ci_hi": f(float(n75.ms_mean) + ci75, 3),
        "table_main_kpi": kpi,
        "table_warmstart": table_warmstart(d),
        "table_transfer": table_transfer(d),
        "table_hdrl": table_hdrl(d),
        "table_morl_5d17d": morl_5d17d,
        "table_morl_pareto_seed": table_morl_pareto_seed(d),
        "table_morl_per_seed": table_morl_per_seed(d),
        "pure_peak_ms": f(pure_peak.m_s), "pure_typ_ms": f(pure_typ.m_s),
        "hyb_peak_ms": f(hyb_peak.m_s), "hyb_typ_ms": f(hyb_typ.m_s),
        "dv_peak_viol": f(dv_peak_v, 1), "dv_typ_viol": f(dv_typ_v, 1),
        "m5_rmse": f(r5.rmse_c), "m5_viol": f(r5.violation_pct, 1), "m5_ms": f(r5.m_s, 3),
        "m5frozen_rmse": f(frozen5.rmse_c, 2), "m5frozen_ms": f(frozen5.m_s, 3),
        "m17_rmse": f(r17.rmse_c), "m17_viol": f(r17.violation_pct, 1), "m17_ms": f(r17.m_s, 3),
        "p0_ms": f(p0.ms_mean, 3), "p0_viol": f(p0.violation_pct_mean, 1),
        "n50_ms": f(n50.ms_mean, 3), "n50_std": f(n50.ms_std, 3), "n50_cv": f(n50.ms_cv, 2),
        "n75_ms": f(n75.ms_mean, 3), "n75_std": f(n75.ms_std, 3), "n75_cv": f(n75.ms_cv, 2),
        "pi_rmse": pi["rmse"], "pi_viol": pi["viol"], "pi_energy": pi["energy"], "pi_ms": pi["ms"],
        "lam_T": "0.10", "lam_P": "5{\\times}10^{-5}",
    }

    # Data-driven m_s = r_time + r_sev decomposition rows for the three backends.
    arch = d["arch"]; tr = d["transfer"]
    v35 = arch[arch.variant == "v35_calibrated"].iloc[0]
    dec_rows = []
    for scen, sl in [("peak_heat_window", "peak"), ("typical_heat_window", "typical")]:
        pv = _scen_row(d["pure"], scen, controller="thermostatic")
        hy = _scen_row(d["hybrid"], scen)
        dv_ms = v35["peak_control_m_s"] if scen == "peak_heat_window" else v35["typical_control_m_s"]
        dv_v = tr[(tr.variant == "direct_v35") & (tr.scenario == scen)].iloc[0]["boptest_violation_pct"]
        for label, ms, viol in [(f"pure v3\n({sl})", pv.m_s, pv.violation_pct),
                                (f"direct v3.5\n({sl})", dv_ms, dv_v),
                                (f"hybrid\n({sl})", hy.m_s, hy.violation_pct)]:
            rt = float(viol) / 100.0
            dec_rows.append((label, rt, max(float(ms) - rt, 0.0)))

    try:
        fig_reward_shaping(ctx)
        fig_ms_decomposition(dec_rows)
        fig_hdrl_architecture()
        fig_morl_pipeline()
    except Exception as exc:
        print(f"[warn] figure regeneration skipped: {exc}")

    write_tex(ctx)
    print(f"Wrote {BASE / 'main.tex'}")
    if "--integrated" in sys.argv:
        sys.path.insert(0, str(BASE.parent))
        from build_integrated_paper import strip_to_body
        (BASE / "section_body.tex").write_text(
            strip_to_body((BASE / "main.tex").read_text(encoding="utf-8")), encoding="utf-8")
        print(f"Wrote {BASE / 'section_body.tex'}")


if __name__ == "__main__":
    main()
