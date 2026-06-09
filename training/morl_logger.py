from __future__ import annotations
from pathlib import Path
import csv
from typing import Dict, Any


class MORLCSVLogger:
    def __init__(self, out_dir: str = "outputs", filename: str = "morl_log.csv"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(out_dir) / filename
        self._initialized = False
        self._writer = None
        self._f = None

    def _init(self, fieldnames):
        self._f = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._f, fieldnames=fieldnames)
        self._writer.writeheader()
        self._initialized = True

    def log(self, row: Dict[str, Any]):
        if not self._initialized:
            self._init(list(row.keys()))
        self._writer.writerow(row)

    def close(self):
        if self._f:
            self._f.flush()
            self._f.close()
            self._f = None
            self._writer = None
            self._initialized = False
