from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_1_2_surrogate_rmse.training.train_block12_backbone import main


if __name__ == "__main__":
    main()
