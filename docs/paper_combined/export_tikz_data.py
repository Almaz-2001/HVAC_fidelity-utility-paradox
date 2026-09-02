"""Export clean, pgfplots-friendly .dat tables for the TikZ manuscript figures.

Several reports/*.csv carry quoted fields with embedded commas that pgfplotstable
cannot parse. This reads them with Python's csv module (correct quoting) and writes
minimal space-separated .dat files under docs/paper_combined/tikz_data/. No numbers
are hand-typed here -- every value is copied from the committed artifact. Run:
    python docs/paper_combined/export_tikz_data.py
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "tikz_data"
OUT.mkdir(parents=True, exist_ok=True)


def read_rows(rel: str) -> list[dict]:
    with (ROOT / rel).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_dat(name: str, header: list[str], rows: list[list]) -> None:
    with (OUT / name).open("w", encoding="utf-8") as fh:
        fh.write(" ".join(header) + "\n")
        for r in rows:
            fh.write(" ".join(str(x) for x in r) + "\n")
    print(f"wrote {OUT/name}  ({len(rows)} rows)")


def rie05_waterfall() -> None:
    """v3_hourly / v3_15min_matched / v35_raw / v35_calibrated 24 h rollout RMSE."""
    d = {r["variant"]: float(r["rmse_24h_c"]) for r in read_rows("reports/block1_corpus_matched_comparison.csv")}
    rows = [[k, d[k]] for k in ("v3_hourly", "v3_15min_matched", "v35_raw", "v35_calibrated")]
    write_dat("rie05_waterfall.dat", ["variant", "rmse"], rows)


def hdrl_sweep() -> None:
    """HDRL lambda_temp sweep: per-lambda m_s and violation for peak/typical windows."""
    rows = read_rows("reports/block2_hdrl_lambda_sweep_summary.csv")
    lam = {"l000": 0.00, "l003": 0.03, "l005": 0.05, "l010": 0.10}
    by: dict = {}
    for r in rows:
        by.setdefault(r["variant"], {})[r["scenario"]] = r
    out = []
    for v, l in sorted(lam.items(), key=lambda x: x[1]):
        if v not in by:
            continue
        pk = by[v].get("peak_heat_window", {})
        ty = by[v].get("typical_heat_window", {})
        out.append([l, pk.get("m_s", "nan"), pk.get("violation_pct", "nan"),
                    ty.get("m_s", "nan"), ty.get("violation_pct", "nan")])
    write_dat("hdrl_sweep.dat", ["lambda", "ms_peak", "viol_peak", "ms_typ", "viol_typ"], out)


def morl_5d17d() -> None:
    """MORL observation-interface ablation: 5D rerun, 5D frozen audit, 17D success."""
    rows = read_rows("reports/block2_morl_5d_reconstructed_comparison.csv")
    def pick(pred):
        return next(r for r in rows if pred(r))
    rerun5 = pick(lambda r: r["variant"] == "MORL_5D_basic_reconstructed")
    frozen5 = pick(lambda r: r["variant"] == "MORL_5D_basic" and r["evidence_layer"] == "historical_frozen")
    d17 = pick(lambda r: r["variant"] == "MORL_17D_power_only")
    out = []
    for lab, r in [("5Drerun", rerun5), ("5Dfrozen", frozen5), ("17D", d17)]:
        out.append([lab, r["rmse_c"], r["violation_pct"], r["m_s"]])
    write_dat("morl_5d17d.dat", ["label", "rmse", "viol", "ms"], out)


def runtime_speed() -> None:
    """Environment throughput (env steps/s) + speedup vs BOPTEST, in canonical order."""
    d = {r["backend"]: r for r in read_rows("reports/speed_benchmark_table.csv")}
    order = ["boptest_rte_http", "v3_surrogate", "v35_calibrated_surrogate", "hybrid_v3_v35_surrogate"]
    out = [[d[k]["env_steps_per_sec"], d[k]["speedup_vs_boptest_rte"]] for k in order if k in d]
    write_dat("runtime_speed.dat", ["steps", "speedup"], out)


def ms_decomposition() -> None:
    """m_s = r_time + r_sev per backend x window (order: peak {BB,GB,hybrid}, typical {...})."""
    def scen_row(rel, scen, controller=None):
        for r in read_rows(rel):
            if r["scenario"] == scen and (controller is None or r.get("controller") == controller):
                return r
        return {}
    arch = {r["variant"]: r for r in read_rows("reports/hou_evins_architecture_justification_table.csv")}
    tr = read_rows("reports/hybrid_transfer_comparison.csv")
    v35 = arch["v35_calibrated"]
    def dv_viol(scen):
        for r in tr:
            if r["variant"] == "direct_v35" and r["scenario"] == scen:
                return float(r["boptest_violation_pct"])
        return float("nan")
    out = []
    for scen in ("peak_heat_window", "typical_heat_window"):
        pv = scen_row("outputs/bestest_air_article7_style_15min/summary.csv", scen, "thermostatic")
        hy = scen_row("outputs/block2_thermostatic_hybrid_v3_v35_l010/summary.csv", scen)
        out.append([pv["r_time"], pv["r_sev"]])                               # BB
        ms = float(v35["peak_control_m_s"] if scen == "peak_heat_window" else v35["typical_control_m_s"])
        rt = dv_viol(scen) / 100.0
        out.append([rt, max(ms - rt, 0.0)])                                   # GB
        out.append([hy["r_time"], hy["r_sev"]])                              # hybrid
    write_dat("ms_decomp.dat", ["rtime", "rsev"], out)


def seed_band() -> None:
    """N=3 seed band for BB / matched-BB / GB(direct) / hybrid, m_s and violation per window."""
    import statistics
    sb = {(r["controller"], r["window"]): r for r in read_rows("reports/block2_thermostatic_seed_band.csv")}
    dirs = ["outputs/block13_closed_loop_transfer_no_delta_t_powerlog_tzone",
            "outputs/block13_closed_loop_transfer_no_delta_t_powerlog_tzone_seed43",
            "outputs/block13_closed_loop_transfer_no_delta_t_powerlog_tzone_seed44"]
    wmap = {"peak_heat_window": "peak", "typical_heat_window": "typical"}
    gm = {"peak": [], "typical": []}
    gv = {"peak": [], "typical": []}
    for d in dirs:
        for r in read_rows(d + "/summary.csv"):
            w = wmap.get(r["scenario"])
            if w:
                gm[w].append(float(r["boptest_m_s"]))
                gv[w].append(float(r["boptest_violation_pct"]))
    def row_for(ctrl):
        if ctrl == "__GB__":
            return [statistics.mean(gm["peak"]), statistics.stdev(gm["peak"]),
                    statistics.mean(gm["typical"]), statistics.stdev(gm["typical"]),
                    statistics.mean(gv["peak"]), statistics.mean(gv["typical"])]
        p, t = sb[(ctrl, "peak")], sb[(ctrl, "typical")]
        return [p["m_s_mean"], p["m_s_std"], t["m_s_mean"], t["m_s_std"],
                p["violation_pct_mean"], t["violation_pct_mean"]]
    order = ["pure v3", "matched v3 (15-min)", "__GB__", "hybrid (lambda_T=0.10)"]
    write_dat("seed_band.dat", ["ms_peak", "std_peak", "ms_typ", "std_typ", "viol_peak", "viol_typ"],
              [row_for(c) for c in order])


RIE_MODELS = [("v3", "outputs/surrogate_v3_rollout_prepared_15min/v3"),
              ("raw", "outputs/surrogate_v35_rollout_prepared_15min_episodeaware/raw_v35"),
              ("cal", "outputs/surrogate_v35_rollout_prepared_15min_episodeaware/calibrated_v35")]


def rie04_predictive() -> None:
    """Panel A: per-horizon rollout RMSE with bootstrap 95% CI. Panel B: |error| CDF."""
    import numpy as np
    def boot(vals, seed=42, n=800):
        v = np.asarray(vals, float); v = v[np.isfinite(v)]
        if len(v) == 0:
            return (float("nan"),) * 3
        rng = np.random.default_rng(seed)
        b = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
        return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    we = {m: read_rows(p + "/window_errors.csv") for m, p in RIE_MODELS}
    rowsA = []
    for h in (1.0, 4.0, 8.0, 24.0):
        row = [h]
        for m, _ in RIE_MODELS:
            vals = [float(r["temp_window_rmse_c"]) for r in we[m] if abs(float(r["horizon_h"]) - h) < 1e-6]
            row += list(boot(vals))
        rowsA.append(row)
    write_dat("rie04_horizon.dat",
              ["h", "v3_m", "v3_lo", "v3_hi", "raw_m", "raw_lo", "raw_hi", "cal_m", "cal_lo", "cal_hi"], rowsA)
    grid = np.linspace(0, 5, 60)
    errs = {}
    for m, p in RIE_MODELS:
        e = [r["temp_error_c"] for r in read_rows(p + "/all_full_rollouts.csv")]
        a = np.abs(np.array([float(x) for x in e if x not in ("", "nan", "NaN")]))
        errs[m] = a[np.isfinite(a)]
    rowsB = [[g] + [float((errs[m] <= g).mean()) if len(errs[m]) else 0.0 for m, _ in RIE_MODELS] for g in grid]
    write_dat("rie04_cdf.dat", ["err", "v3", "raw", "cal"], rowsB)


def rie03_stageb() -> None:
    """Stage B C_zon identification trajectory (epoch vs C_zon in 1e5 J/K)."""
    rows = read_rows("outputs/surrogate_v35_inverse_boptest_15min_episodeaware/stage_b_history_v35.csv")
    out = [[r["epoch"], float(r["c_zon_j_per_k"]) / 1e5] for r in rows]
    write_dat("rie03_stageb.dat", ["epoch", "czon"], out)


def rie03_meta() -> None:
    """C_zon prior and final identified value (in 1e5 J/K) for the Stage B panel."""
    import json
    d = json.loads((ROOT / "outputs/surrogate_v35_inverse_boptest_15min_power_head_only/calibration_summary_boptest_v35.json").read_text(encoding="utf-8"))
    write_dat("rie03_meta.dat", ["prior", "final"],
              [[float(d["c_zon_prior_j_per_k"]) / 1e5, float(d["c_zon_final_j_per_k"]) / 1e5]])


def rie03_hist() -> None:
    """One-step temperature-residual density, raw vs calibrated GB."""
    import numpy as np
    def errs(p):
        e = [r["temp_error_c"] for r in read_rows(p + "/all_full_rollouts.csv")]
        a = np.array([float(x) for x in e if x not in ("", "nan", "NaN")])
        return a[np.isfinite(a)]
    raw = errs("outputs/surrogate_v35_rollout_prepared_15min_episodeaware/raw_v35")
    cal = errs("outputs/surrogate_v35_rollout_prepared_15min_episodeaware/calibrated_v35")
    bins = np.linspace(-3, 3, 61)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    hr, _ = np.histogram(raw, bins=bins, density=True)
    hc, _ = np.histogram(cal, bins=bins, density=True)
    write_dat("rie03_hist.dat", ["x", "raw", "cal"], [[ctr[i], hr[i], hc[i]] for i in range(len(ctr))])


def lambda_specificity() -> None:
    """Controller-family specificity of lambda_T: cross-window mean m_s per family."""
    lam = {"l000": 0.00, "l003": 0.03, "l005": 0.05, "l010": 0.10}
    by = {}
    for r in read_rows("reports/block2_hdrl_lambda_sweep_summary.csv"):
        by.setdefault(r["variant"], {})[r["scenario"]] = r
    hdrl = []
    for v, l in sorted(lam.items(), key=lambda x: x[1]):
        if v in by and "peak_heat_window" in by[v] and "typical_heat_window" in by[v]:
            m = 0.5 * (float(by[v]["peak_heat_window"]["m_s"]) + float(by[v]["typical_heat_window"]["m_s"]))
            hdrl.append([l, m])
    write_dat("lam_hdrl.dat", ["lambda", "ms"], hdrl)
    sc = {r["controller"]: r for r in read_rows("reports/block2_fidelity_utility_scatter.csv")}
    def mm(prefix):
        return next(float(r["m_s_mean"]) for c, r in sc.items() if c.startswith(prefix))
    write_dat("lam_ppo.dat", ["lambda", "ms"], [[0.00, mm("v3 (")], [0.10, mm("hybrid")]])
    mr = next(r for r in read_rows("reports/block2_morl_comparison_summary.csv") if "17" in r["variant"])
    write_dat("lam_morl.dat", ["lambda", "ms"], [[float(mr["lambda_temp_disagree"]), float(mr["m_s"])]])


def block1_fig10() -> None:
    """Transfer-gap diagnostics: mean violation / action-gap / first-divergence per backend."""
    import statistics
    tr = read_rows("reports/hybrid_transfer_comparison.csv")
    out = []
    for i, v in enumerate(["pure_v3", "direct_v35", "hybrid_l010"]):
        rs = [r for r in tr if r["variant"] == v]
        out.append([i,
                    statistics.mean(float(r["boptest_violation_pct"]) for r in rs),
                    statistics.mean(float(r["action_gap_norm"]) for r in rs),
                    statistics.mean(float(r["first_divergence_step"]) for r in rs)])
    write_dat("block1_fig10.dat", ["idx", "viol", "gap", "div"], out)


def final17_fig06() -> None:
    """Live controller comparison: m_s / violation / RMSE_T / energy per backend x window."""
    arch = {r["variant"]: r for r in read_rows("reports/hou_evins_architecture_justification_table.csv")}
    tr = read_rows("reports/hybrid_transfer_comparison.csv")
    def viol(tv, scen):
        return next(float(r["boptest_violation_pct"]) for r in tr if r["variant"] == tv and r["scenario"] == scen)
    B = [("v3", "pure_v3"), ("v35_calibrated", "direct_v35"), ("hybrid_l010", "hybrid_l010")]
    out = []
    for wi, win in enumerate(["peak", "typical"]):
        row = [wi]
        row += [arch[av][f"{win}_control_m_s"] for av, _ in B]
        row += [viol(tv, f"{win}_heat_window") for _, tv in B]
        row += [arch[av][f"{win}_transfer_temp_rmse_c"] for av, _ in B]
        row += [arch[av][f"{win}_energy_kwh"] for av, _ in B]
        out.append(row)
    write_dat("final17_fig06.dat",
              ["win", "ms_bb", "ms_gb", "ms_hy", "viol_bb", "viol_gb", "viol_hy",
               "rmse_bb", "rmse_gb", "rmse_hy", "en_bb", "en_gb", "en_hy"], out)


def main() -> None:
    rie05_waterfall()
    hdrl_sweep()
    morl_5d17d()
    runtime_speed()
    ms_decomposition()
    seed_band()
    rie04_predictive()
    rie03_stageb()
    rie03_meta()
    rie03_hist()
    lambda_specificity()
    block1_fig10()
    final17_fig06()
    morl_17d_heatmap()
    morl_inversion()
    fig13_verdict()
    conceptual()
    warmstart()
    rie07_episodes()
    regime_progression()
    runtime_fidelity()
    closed_loop()
    phase_points()


_TRACES = {"v3": "outputs/bestest_air_article7_style_15min/traces/typical_heat_window_thermostatic.csv",
           "v35": "outputs/block2_bestest_air_15min_thermostatic_v35/traces/typical_heat_window_thermostatic.csv",
           "hy": "outputs/block2_thermostatic_hybrid_v3_v35_l010/traces/typical_heat_window_thermostatic.csv"}


def closed_loop() -> None:
    """Closed-loop typical-window traces (ambient, zone temp, supply cmd, power) x 3 controllers."""
    data = {k: read_rows(v) for k, v in _TRACES.items()}
    n = min(288, *(len(d) for d in data.values()))
    t0 = float(data["v3"][0]["sim_time_sec"])
    out = []
    for i in range(n):
        hour = (float(data["v3"][i]["sim_time_sec"]) - t0) / 3600.0
        row = [hour, data["v3"][i]["t_amb_c"]]
        row += [data[k][i]["t_zone_c"] for k in ("v3", "v35", "hy")]
        row += [data[k][i]["t_supply_cmd_c"] for k in ("v3", "v35", "hy")]
        row += [float(data[k][i]["p_total_w"]) / 1000.0 for k in ("v3", "v35", "hy")]
        out.append(row)
    write_dat("closed_loop.dat",
              ["hour", "amb", "tz_v3", "tz_v35", "tz_hy", "ts_v3", "ts_v35", "ts_hy", "p_v3", "p_v35", "p_hy"], out)


def phase_points() -> None:
    """Action phase portrait points (thermal error vs normalised action) per controller."""
    for k, rel in _TRACES.items():
        rows = read_rows(rel)
        write_dat(f"phase_{k}.dat", ["err", "a0"],
                  [[float(r["t_zone_c"]) - 22.5, r["a0"]] for r in rows])


def regime_progression() -> None:
    """Primary-testcase surrogate fidelity across recalibration regimes (RMSE_T, power MAE)."""
    rows = read_rows("reports/block3_bestest_hydronic_heat_pump_transfer_summary.csv")
    order = [("partial", "stage_c_top5_heads"), ("partial", "stage_c_allrows_power"),
             ("partial", "stage_c_allrows_heads"), ("full", "stage_abc_allrows_heads")]
    out = []
    for i, (reg, art) in enumerate(order):
        r = next((x for x in rows if x["regime"] == reg and x["artifact"] == art), None)
        if r:
            out.append([i, r["rmse_t_c"], r["power_mae_w"]])
    write_dat("regime_prog.dat", ["i", "rmse", "pmae"], out)


def runtime_fidelity() -> None:
    """Throughput vs 24 h rollout RMSE feasibility (BOPTEST + BB / GB / hybrid surrogates)."""
    spd = {r["backend"]: r for r in read_rows("reports/speed_benchmark_table.csv")}
    sc = {r["controller"]: r for r in read_rows("reports/block2_fidelity_utility_scatter.csv")}
    def rmse(prefix):
        return next(float(r["rmse_24h_c"]) for c, r in sc.items() if c.startswith(prefix))
    rows = [(spd["boptest_rte_http"]["env_steps_per_sec"], 0.0),
            (spd["v3_surrogate"]["env_steps_per_sec"], rmse("v3 (")),
            (spd["v35_calibrated_surrogate"]["env_steps_per_sec"], rmse("v3.5")),
            (spd["hybrid_v3_v35_surrogate"]["env_steps_per_sec"], rmse("hybrid"))]
    write_dat("runtime_fid.dat", ["i", "steps", "rmse"], [[i, s, r] for i, (s, r) in enumerate(rows)])


def warmstart() -> None:
    """Warm-start negative control: peak-heat trace + peak KPI bars (scratch vs warm-start)."""
    def summ(kind, scen):
        return next(r for r in read_rows(f"outputs/block2_thermostatic_warmstart_utility/{kind}/summary.csv") if r["scenario"] == scen)
    sp, wp = summ("scratch_eval", "peak_heat_window"), summ("warmstart_eval", "peak_heat_window")
    write_dat("warmstart_kpi.dat", ["i", "scratch", "warm"],
              [[0, sp["m_s"], wp["m_s"]], [1, sp["violation_pct"], wp["violation_pct"]], [2, sp["energy_kwh"], wp["energy_kwh"]]])
    st = read_rows("outputs/block2_thermostatic_warmstart_utility/scratch_eval/traces/peak_heat_window_thermostatic.csv")
    wt = read_rows("outputs/block2_thermostatic_warmstart_utility/warmstart_eval/traces/peak_heat_window_thermostatic.csv")
    n = min(96, len(st), len(wt))
    t0 = float(st[0]["sim_time_sec"])
    write_dat("warmstart_trace.dat", ["hour", "scratch", "warm"],
              [[(float(st[i]["sim_time_sec"]) - t0) / 3600.0, st[i]["t_zone_c"], wt[i]["t_zone_c"]] for i in range(n)])


def rie07_episodes() -> None:
    """Replicative validity: per-episode temperature RMSE for v3 / raw GB / calibrated GB."""
    def rmse(p):
        return [float(r["temp_rmse_c"]) for r in read_rows(p + "/episode_summary.csv")]
    v3 = rmse("outputs/surrogate_v3_rollout_prepared_15min/v3")
    raw = rmse("outputs/surrogate_v35_rollout_prepared_15min_episodeaware/raw_v35")
    cal = rmse("outputs/surrogate_v35_rollout_prepared_15min_episodeaware/calibrated_v35")
    n = min(len(v3), len(raw), len(cal))
    write_dat("rie07.dat", ["i", "v3", "raw", "cal"], [[i, v3[i], raw[i], cal[i]] for i in range(n)])


def fig13_verdict() -> None:
    """Frozen-controller transfer verdict + m_s/tau ratio per testcase (identical across regimes)."""
    tm = {r["testcase"]: r for r in read_rows("reports/block3_transfer_matrix.csv")}
    order = ["bestest_hydronic_heat_pump", "bestest_hydronic", "singlezone_commercial_hydronic"]
    out = []
    for i, tc in enumerate(order):
        r = tm[tc]
        ratio = float(r["m_s_rl"]) / float(r["pass_threshold_m_s"])
        v = r["none_controller_verdict"]
        code = (0 if v == "PASS" else -1) if tc == "singlezone_commercial_hydronic" else (1 if v == "PASS" else -1)
        out.append([i, ratio, code])
    write_dat("fig13_verdict.dat", ["i", "ratio", "code"], out)


def conceptual() -> None:
    """Evidence matrix: RMSE / roughness fold / saturation / m_s per backend (BB, matched-BB, GB, hybrid)."""
    import numpy as np
    sc = {r["controller"]: r for r in read_rows("reports/block2_fidelity_utility_scatter.csv")}
    def g(prefix, col):
        return next(float(r[col]) for c, r in sc.items() if c.startswith(prefix))
    sh = {r["surrogate"]: float(r["rel_roughness"]) for r in read_rows("reports/block2_mechanism_surface_sharpness.csv")}
    base = sh["v3 hourly (1h)"]
    def sat_of(rel):
        a = np.array([float(r["a0"]) for r in read_rows(rel)])
        return 100.0 * float((np.abs(a) > 0.9).mean())
    tr = {"v3": "bestest_air_article7_style_15min", "matched": "bestest_air_pure_v3_15min",
          "v35": "block2_bestest_air_15min_thermostatic_v35", "hybrid": "block2_thermostatic_hybrid_v3_v35_l010"}
    rmse = {"v3": g("v3 (", "rmse_24h_c"), "matched": g("matched", "rmse_24h_c"), "v35": g("v3.5", "rmse_24h_c"), "hybrid": g("hybrid", "rmse_24h_c")}
    ms = {"v3": g("v3 (", "m_s_mean"), "matched": g("matched", "m_s_mean"), "v35": g("v3.5", "m_s_mean"), "hybrid": g("hybrid", "m_s_mean")}
    rough = {"v3": 1.0, "matched": sh["v3 matched (15min)"] / base, "v35": sh["v3.5 calibrated"] / base, "hybrid": 1.0}
    sat = {k: sat_of(f"outputs/{tr[k]}/traces/peak_heat_window_thermostatic.csv") for k in tr}
    order = ["v3", "matched", "v35", "hybrid"]
    write_dat("conceptual.dat", ["i", "rmse", "rough", "sat", "ms"],
              [[i, rmse[k], rough[k], sat[k], ms[k]] for i, k in enumerate(order)])


MONTHS = ["Jan_Winter", "Feb_Winter", "Mar_Spring", "Apr_Spring", "May_Spring", "Jun_Summer",
          "Jul_Summer", "Aug_Summer", "Sep_Autumn", "Oct_Autumn", "Nov_Autumn", "Dec_Winter"]


def _norm(a):
    import numpy as np
    a = np.asarray(a, float); lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else a * 0.0


def morl_17d_heatmap() -> None:
    """MORL 17D yearly seasonal validation: violation / energy / m_s per month (+ per-row norm)."""
    rows = [r for r in read_rows("outputs/morl_hybrid_v3_v35_power_only_17d/seed42/yearly_eval/morl_yearly_summary.csv")
            if r["name"].upper() != "MEAN"]
    viol = [float(r["viol_pct"]) for r in rows]
    energy = [float(r["energy_kwh"]) for r in rows]
    ms = [float(r["ms"]) for r in rows]
    vn, en, mn = _norm(viol), _norm(energy), _norm(ms)
    write_dat("morl_17d.dat", ["i", "viol", "energy", "ms", "vn", "en", "mn"],
              [[i, viol[i], energy[i], ms[i], vn[i], en[i], mn[i]] for i in range(len(rows))])


def morl_inversion() -> None:
    """Seasonal variance inversion: across-seed m_s std for Practical vs Neutral (+ shared norm)."""
    prac = {r["scenario"]: float(r["ms_std"]) for r in read_rows("reports/morl_practical_canonical_monthly_variance_diagnostic.csv")}
    neu = {r["scenario"]: float(r["ms_std"]) for r in read_rows("reports/morl_neutral_canonical_monthly_variance_diagnostic.csv")}
    allv = [prac[s] for s in MONTHS if s in prac] + [neu[s] for s in MONTHS if s in neu]
    lo, hi = min(allv), max(allv)
    def n(v):
        return (v - lo) / (hi - lo) if hi > lo else 0.0
    write_dat("morl_inv.dat", ["i", "prac", "neu", "pn", "nn"],
              [[i, prac.get(MONTHS[i], 0), neu.get(MONTHS[i], 0), n(prac.get(MONTHS[i], 0)), n(neu.get(MONTHS[i], 0))]
               for i in range(12)])


if __name__ == "__main__":
    main()
