from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCK_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data" / "block_1_2_surrogate_rmse"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "block_1_2_surrogate_rmse"

BLOCK_NAME = "1.2"

SURROGATE_STEP_SEC = 900
LEGACY_V35_STEP_SEC = 3600

COMFORT_LOW_C = 21.0
COMFORT_HIGH_C = 24.0
COMFORT_CENTER_C = (COMFORT_LOW_C + COMFORT_HIGH_C) / 2.0

DEFAULT_SAFE_T_ZONE_MIN_C = 17.5
DEFAULT_SAFE_T_ZONE_MAX_C = 28.0

DEFAULT_TRAIN_T_ZONE_MIN_C = 17.0
DEFAULT_TRAIN_T_ZONE_MAX_C = 29.0
DEFAULT_TRAIN_ABS_DELTA_T_MAX_C = 2.5
DEFAULT_TRAIN_P_TOTAL_MAX_W = 6000.0

DEFAULT_PREPARED_DATASET_CSV = DATA_ROOT / "boptest_block12_15min_prepared.csv"
DEFAULT_COLLECTED_DATASET_CSV = DATA_ROOT / "boptest_block12_15min_collected.csv"
DEFAULT_COLLECTED_TRAIN_SUBSET_CSV = DATA_ROOT / "boptest_block12_15min_collected_train_subset.csv"
DEFAULT_HYBRID_DATASET_CSV = DATA_ROOT / "boptest_block12_15min_hybrid_anchor.csv"

DEFAULT_PREPARED_OUTPUT_DIR = OUTPUT_ROOT / "prepared_15min_dataset"
DEFAULT_COLLECTED_OUTPUT_DIR = OUTPUT_ROOT / "collected_15min_dataset"
DEFAULT_HYBRID_OUTPUT_DIR = OUTPUT_ROOT / "hybrid_15min_dataset"

DEFAULT_TRAINING_OUTPUT_ROOT = OUTPUT_ROOT
DEFAULT_PREPARED_TRAIN_RUN = "prepared_15min_baseline"
DEFAULT_PREPARED_ROLLOUT_TRAIN_RUN = "prepared_15min_rollout_select"
DEFAULT_PREPARED_ROLLOUT_LONG_TRAIN_RUN = "prepared_15min_rollout_long"
DEFAULT_COLLECTED_TRAIN_RUN = "collected_15min_focus"
DEFAULT_HYBRID_TRAIN_RUN = "hybrid_15min_anchor"
