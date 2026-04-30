from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

V35_15MIN_SUMMARY_JSON = (
    ROOT / "outputs" / "surrogate_v35_inverse_boptest_15min_power_head_only" / "calibration_summary_boptest_v35.json"
)
V35_15MIN_PRE_ROLLOUT_SUMMARY_JSON = (
    ROOT / "outputs" / "surrogate_v35_inverse_boptest_15min" / "calibration_summary_boptest_v35.json"
)
V35_HOURLY_LEGACY_SUMMARY_JSON = (
    ROOT
    / "outputs"
    / "surrogate_v35_inverse_boptest_prior420_heads_only"
    / "calibration_summary_boptest_v35.json"
)

MORL_V35_15MIN_ARTIFACT_ROOT = ROOT / "outputs" / "morl_surrogate_ppo_v35_15min"
