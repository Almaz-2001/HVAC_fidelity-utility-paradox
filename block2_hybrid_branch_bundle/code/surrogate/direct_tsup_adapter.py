from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]

LEGACY_V3 = "legacy_v3"
V35_RAW = "v35_raw"
V35_CALIBRATED = "v35_calibrated"
HYBRID_V3_V35 = "hybrid_v3_v35"

SUPPORTED_DIRECT_TSUP_KINDS = {
    LEGACY_V3,
    V35_RAW,
    V35_CALIBRATED,
    HYBRID_V3_V35,
}

_DEFAULT_LEGACY_V3_CANDIDATES = [
    REPO_ROOT / "outputs" / "surrogate_v2" / "rc_node_v3_tsupply.pt",
    Path("/app/outputs/surrogate_v2/rc_node_v3_tsupply.pt"),
]
_DEFAULT_V35_SUMMARY_CANDIDATES = [
    REPO_ROOT
    / "outputs"
    / "surrogate_v35_inverse_boptest_prior420_heads_only"
    / "calibration_summary_boptest_v35.json",
    Path("/app/outputs/surrogate_v35_inverse_boptest_prior420_heads_only/calibration_summary_boptest_v35.json"),
]


def _resolve_kind(value: str | None) -> str:
    raw = (value or LEGACY_V3).strip().lower()
    aliases = {
        "legacy": LEGACY_V3,
        "legacy_v3": LEGACY_V3,
        "v3": LEGACY_V3,
        "raw_v35": V35_RAW,
        "v35_raw": V35_RAW,
        "v35": V35_RAW,
        "calibrated_v35": V35_CALIBRATED,
        "v35_calibrated": V35_CALIBRATED,
        "calibrated": V35_CALIBRATED,
        "hybrid_v3_v35": HYBRID_V3_V35,
        "hybrid": HYBRID_V3_V35,
    }
    kind = aliases.get(raw, raw)
    if kind not in SUPPORTED_DIRECT_TSUP_KINDS:
        raise ValueError(
            f"Unsupported direct-TSup surrogate kind: {value}. "
            f"Supported: {sorted(SUPPORTED_DIRECT_TSUP_KINDS)}"
        )
    return kind


def _resolve_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _resolve_path(raw: str | Path | None, anchors: list[Path] | None = None) -> Path | None:
    if raw is None:
        return None

    path = Path(raw)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(path)
        for anchor in anchors or []:
            candidates.append(anchor / path)
        candidates.append(REPO_ROOT / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1] if candidates else None


def _resolve_torch_device(value: str | torch.device | None, default: str = "cpu") -> torch.device:
    if isinstance(value, torch.device):
        raw = value.type
    else:
        raw = str(value or default).strip().lower()
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raw = "cpu"
    return torch.device(raw)


class DirectTSupModelAdapter(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        kind: str,
        metadata: dict[str, Any] | None = None,
        comparison_model: nn.Module | None = None,
        comparison_kind: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.kind = _resolve_kind(kind)
        self.metadata = metadata or {}
        self.comparison_model = comparison_model
        self.comparison_kind = _resolve_kind(comparison_kind) if comparison_kind is not None else None
        base_model = model.surrogate if hasattr(model, "surrogate") else model
        self.P_MAX = float(getattr(base_model, "P_MAX", 5500.0))

    @staticmethod
    def _forward_kind(
        model: nn.Module,
        kind: str,
        t_zone: torch.Tensor,
        t_amb: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kind == LEGACY_V3:
            return model(t_zone, t_amb, hour, day, a0, a1)

        if kind == V35_RAW:
            if hasattr(model, "surrogate"):
                t_surr, _, p_surr, _, _, _ = model(t_zone, t_amb, hour, day, a0, a1)
                return t_surr, p_surr
            return model(t_zone, t_amb, hour, day, a0, a1)

        if kind == V35_CALIBRATED:
            t_surr, t_cal, p_surr, p_cal, _, _ = model(t_zone, t_amb, hour, day, a0, a1)
            return t_cal, p_cal

        raise RuntimeError(f"Unhandled surrogate kind: {kind}")

    def forward(
        self,
        t_zone: torch.Tensor,
        t_amb: torch.Tensor,
        hour: torch.Tensor,
        day: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        primary_kind = LEGACY_V3 if self.kind == HYBRID_V3_V35 else self.kind
        return self._forward_kind(self.model, primary_kind, t_zone, t_amb, hour, day, a0, a1)

    def step_numpy(
        self,
        t_zone: float,
        t_amb: float,
        hour: float,
        day: float,
        a0: float,
        a1: float,
        device: str | torch.device | None = None,
    ) -> tuple[float, float]:
        step = self.step_with_aux_numpy(
            t_zone=t_zone,
            t_amb=t_amb,
            hour=hour,
            day=day,
            a0=a0,
            a1=a1,
            device=device,
        )
        return float(step["t_next"]), float(step["p_total"])

    def step_with_aux_numpy(
        self,
        t_zone: float,
        t_amb: float,
        hour: float,
        day: float,
        a0: float,
        a1: float,
        device: str | torch.device | None = None,
    ) -> dict[str, float | None]:
        torch_device = _resolve_torch_device(device, default="cpu")
        with torch.no_grad():
            t_next, p_total = self(
                torch.tensor([t_zone], dtype=torch.float32, device=torch_device),
                torch.tensor([t_amb], dtype=torch.float32, device=torch_device),
                torch.tensor([hour], dtype=torch.float32, device=torch_device),
                torch.tensor([day], dtype=torch.float32, device=torch_device),
                torch.tensor([a0], dtype=torch.float32, device=torch_device),
                torch.tensor([a1], dtype=torch.float32, device=torch_device),
            )
            comparison_t = None
            comparison_p = None
            if self.kind == HYBRID_V3_V35 and self.comparison_model is not None and self.comparison_kind is not None:
                comparison_t_tensor, comparison_p_tensor = self._forward_kind(
                    self.comparison_model,
                    self.comparison_kind,
                    torch.tensor([t_zone], dtype=torch.float32, device=torch_device),
                    torch.tensor([t_amb], dtype=torch.float32, device=torch_device),
                    torch.tensor([hour], dtype=torch.float32, device=torch_device),
                    torch.tensor([day], dtype=torch.float32, device=torch_device),
                    torch.tensor([a0], dtype=torch.float32, device=torch_device),
                    torch.tensor([a1], dtype=torch.float32, device=torch_device),
                )
                comparison_t = float(comparison_t_tensor[0].detach().cpu())
                comparison_p = float(comparison_p_tensor[0].detach().cpu())
        primary_t = float(t_next[0].detach().cpu())
        primary_p = float(p_total[0].detach().cpu())
        return {
            "t_next": primary_t,
            "p_total": primary_p,
            "comparison_t_next": comparison_t,
            "comparison_p_total": comparison_p,
            "temp_disagreement": None if comparison_t is None else abs(primary_t - comparison_t),
            "power_disagreement": None if comparison_p is None else abs(primary_p - comparison_p),
        }

    def describe(self) -> dict[str, Any]:
        return dict(self.metadata)


def _load_legacy_v3_model(
    legacy_model_path: Path,
    device: torch.device,
) -> DirectTSupModelAdapter:
    from surrogate.rc_node_v2 import RCNeuralODEv2

    checkpoint = torch.load(legacy_model_path, map_location=device, weights_only=False)
    model = RCNeuralODEv2(hidden_dim=int(checkpoint.get("hidden_dim", 64)))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return DirectTSupModelAdapter(
        model=model,
        kind=LEGACY_V3,
        metadata={
            "kind": LEGACY_V3,
            "legacy_model_path": str(legacy_model_path),
            "device": str(device),
        },
    )


def _load_v35_from_base_model(
    base_model_path: Path,
    device: torch.device,
    c_zon_init: float,
    c_zon_min: float,
    q_scale: float,
    runtime_step_sec: float | None = None,
    legacy_step_sec: float | None = None,
) -> nn.Module:
    from surrogate.rc_node_v35 import load_v35_from_v2_checkpoint

    model = load_v35_from_v2_checkpoint(
        checkpoint_path=str(base_model_path),
        device=device,
        c_zon_init=float(c_zon_init),
        c_zon_min=float(c_zon_min),
        q_scale=float(q_scale),
        dt_seconds=runtime_step_sec,
        legacy_step_seconds=legacy_step_sec or 3600.0,
    )
    model.to(device)
    model.eval()
    return model


def _load_v35_staged_model(
    summary_json_path: Path,
    checkpoint_path: Path | None,
    base_model_path: Path | None,
    device: torch.device,
    c_zon_min: float,
    q_scale: float,
    runtime_step_sec: float | None = None,
    legacy_step_sec: float | None = None,
) -> tuple[nn.Module, dict[str, Any], Path, Path]:
    from surrogate.inverse_problem_boptest_v35 import StagedCalibratedSurrogateV35

    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    anchors = [summary_json_path.parent]
    resolved_base_model = _resolve_path(base_model_path or summary.get("model_path"), anchors=anchors)
    if resolved_base_model is None:
        raise FileNotFoundError(
            f"Could not resolve v3 base model path from summary: {summary_json_path}"
        )

    c_zon_init = float(summary.get("c_zon_prior_j_per_k", summary.get("c_zon_final_j_per_k", 5.3e5)))
    summary_runtime_step_sec = float(summary.get("runtime_step_sec", 3600.0))
    summary_legacy_step_sec = float(summary.get("legacy_checkpoint_step_sec", 3600.0))
    resolved_runtime_step_sec = float(runtime_step_sec or summary_runtime_step_sec)
    resolved_legacy_step_sec = float(legacy_step_sec or summary_legacy_step_sec)

    if runtime_step_sec is not None and abs(float(runtime_step_sec) - summary_runtime_step_sec) > 1e-6:
        raise ValueError(
            "Calibrated v3.5 checkpoint step mismatch: "
            f"summary was calibrated at {summary_runtime_step_sec:.0f}s but runtime requested "
            f"{float(runtime_step_sec):.0f}s. Recalibrate v3.5 for the target step size first."
        )

    warm_model = _load_v35_from_base_model(
        base_model_path=resolved_base_model,
        device=device,
        c_zon_init=c_zon_init,
        c_zon_min=c_zon_min,
        q_scale=q_scale,
        runtime_step_sec=resolved_runtime_step_sec,
        legacy_step_sec=resolved_legacy_step_sec,
    )
    temp_head_feature_set = str(summary.get("temp_head_feature_set", "v1"))
    staged_model = StagedCalibratedSurrogateV35(warm_model, temp_head_feature_set=temp_head_feature_set).to(device)

    resolved_checkpoint = _resolve_path(
        checkpoint_path or summary_json_path.with_name("rc_node_v35_boptest_staged_calibrated.pt"),
        anchors=anchors,
    )
    if resolved_checkpoint is None or not resolved_checkpoint.exists():
        raise FileNotFoundError(
            f"Could not resolve calibrated v3.5 checkpoint for summary: {summary_json_path}"
        )

    checkpoint = torch.load(resolved_checkpoint, map_location=device, weights_only=False)
    staged_model.surrogate.load_state_dict(checkpoint["surrogate_state"])
    if "temp_head_state" in checkpoint:
        staged_model.temp_head.load_state_dict(checkpoint["temp_head_state"], strict=False)
    if "power_head_state" in checkpoint:
        staged_model.power_head.load_state_dict(checkpoint["power_head_state"], strict=False)
    staged_model.eval()
    return staged_model, summary, resolved_base_model, resolved_checkpoint


def load_direct_tsup_adapter(
    kind: str | None = None,
    *,
    legacy_model_path: str | Path | None = None,
    summary_json: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    base_model_path: str | Path | None = None,
    device: str | torch.device | None = None,
    c_zon_min: float = 5.0e4,
    q_scale: float = 3000.0,
    runtime_step_sec: float | None = None,
    legacy_step_sec: float | None = None,
) -> DirectTSupModelAdapter:
    resolved_kind = _resolve_kind(kind)
    torch_device = _resolve_torch_device(device, default="cpu")

    if resolved_kind == LEGACY_V3:
        resolved_legacy = _resolve_path(
            legacy_model_path,
            anchors=[REPO_ROOT],
        ) or _resolve_existing_path(_DEFAULT_LEGACY_V3_CANDIDATES)
        if not resolved_legacy.exists():
            raise FileNotFoundError(f"Legacy direct-TSup surrogate not found: {resolved_legacy}")
        return _load_legacy_v3_model(resolved_legacy, torch_device)

    if resolved_kind == HYBRID_V3_V35:
        primary_adapter = load_direct_tsup_adapter(
            kind=LEGACY_V3,
            legacy_model_path=legacy_model_path,
            device=torch_device,
            runtime_step_sec=runtime_step_sec,
            legacy_step_sec=legacy_step_sec,
        )
        comparison_adapter = load_direct_tsup_adapter(
            kind=V35_CALIBRATED,
            summary_json=summary_json,
            checkpoint_path=checkpoint_path,
            base_model_path=base_model_path,
            device=torch_device,
            c_zon_min=c_zon_min,
            q_scale=q_scale,
            runtime_step_sec=runtime_step_sec,
            legacy_step_sec=legacy_step_sec,
        )
        return DirectTSupModelAdapter(
            model=primary_adapter.model,
            kind=HYBRID_V3_V35,
            metadata={
                "kind": HYBRID_V3_V35,
                "primary": primary_adapter.describe(),
                "comparison": comparison_adapter.describe(),
                "device": str(torch_device),
            },
            comparison_model=comparison_adapter.model,
            comparison_kind=V35_CALIBRATED,
        )

    resolved_summary = _resolve_path(summary_json, anchors=[REPO_ROOT])
    if resolved_summary is None:
        resolved_summary = _resolve_existing_path(_DEFAULT_V35_SUMMARY_CANDIDATES)

    if resolved_summary.exists():
        staged_model, summary, resolved_base, resolved_ckpt = _load_v35_staged_model(
            summary_json_path=resolved_summary,
            checkpoint_path=_resolve_path(checkpoint_path, anchors=[resolved_summary.parent, REPO_ROOT]),
            base_model_path=_resolve_path(base_model_path, anchors=[resolved_summary.parent, REPO_ROOT]),
            device=torch_device,
            c_zon_min=c_zon_min,
            q_scale=q_scale,
            runtime_step_sec=runtime_step_sec,
            legacy_step_sec=legacy_step_sec,
        )
        return DirectTSupModelAdapter(
            model=staged_model,
            kind=resolved_kind,
            metadata={
                "kind": resolved_kind,
                "summary_json": str(resolved_summary),
                "checkpoint_path": str(resolved_ckpt),
                "base_model_path": str(resolved_base),
                "device": str(torch_device),
                "c_zon_final_j_per_k": summary.get("c_zon_final_j_per_k"),
                "stage_c_mode": summary.get("stage_c_mode"),
                "runtime_step_sec": float(summary.get("runtime_step_sec", runtime_step_sec or 3600.0)),
                "legacy_step_sec": float(summary.get("legacy_checkpoint_step_sec", legacy_step_sec or 3600.0)),
            },
        )

    if resolved_kind == V35_RAW:
        resolved_base = _resolve_path(
            base_model_path or legacy_model_path,
            anchors=[REPO_ROOT],
        ) or _resolve_existing_path(_DEFAULT_LEGACY_V3_CANDIDATES)
        if not resolved_base.exists():
            raise FileNotFoundError(
                f"Could not resolve base model for v3.5 raw surrogate: {resolved_base}"
            )
        raw_model = _load_v35_from_base_model(
            base_model_path=resolved_base,
            device=torch_device,
            c_zon_init=5.3e5,
            c_zon_min=c_zon_min,
            q_scale=q_scale,
            runtime_step_sec=runtime_step_sec,
            legacy_step_sec=legacy_step_sec,
        )
        return DirectTSupModelAdapter(
            model=raw_model,
            kind=V35_RAW,
            metadata={
                "kind": V35_RAW,
                "base_model_path": str(resolved_base),
                "device": str(torch_device),
                "summary_json": None,
                "checkpoint_path": None,
                "runtime_step_sec": float(runtime_step_sec or 3600.0),
                "legacy_step_sec": float(legacy_step_sec or 3600.0),
            },
        )

    raise FileNotFoundError(
        "Calibrated v3.5 surrogate requires a valid calibration summary/checkpoint. "
        f"Tried summary path: {resolved_summary}"
    )
