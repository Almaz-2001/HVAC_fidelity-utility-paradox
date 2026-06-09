"""Compatibility wrapper for the canonical surrogate backbone trainer."""

from __future__ import annotations

from surrogate.train_surrogate_backbone import (
    BOPTESTBackboneDataset,
    BOPTESTDatasetV2,
    benchmark_backbone_speed,
    benchmark_speed_v2,
    compute_metrics,
    evaluate_rollout_rmse,
    main,
    train_backbone,
    train_v2,
    validate_backbone_safety,
    validate_safety,
)

__all__ = [
    "BOPTESTBackboneDataset",
    "BOPTESTDatasetV2",
    "benchmark_backbone_speed",
    "benchmark_speed_v2",
    "compute_metrics",
    "evaluate_rollout_rmse",
    "main",
    "train_backbone",
    "train_v2",
    "validate_backbone_safety",
    "validate_safety",
]


if __name__ == "__main__":
    main()
