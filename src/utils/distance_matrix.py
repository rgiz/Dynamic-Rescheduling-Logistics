from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Tuple


class DistanceMatrix:
    """Lazy‑loads a symmetric travel‑time & distance matrix based on XY coordinates.

    Saved on disk as a compressed NumPy `.npz` with keys `time` and `dist` for
    easy, dependency‑free reloads.
    """

    def __init__(self, npz_path: Path):
        self._path = Path(npz_path)
        self._loaded: bool = False
        self._time: np.ndarray | None = None
        self._dist: np.ndarray | None = None

    def _load(self) -> None:
        if not self._loaded:
            data = np.load(self._path)
            self._time = data["time"]
            self._dist = data["dist"]
            self._loaded = True

    def travel_time(self, i: int, j: int) -> float:
        self._load()
        return float(self._time[i, j])

    def travel_dist(self, i: int, j: int) -> float:
        self._load()
        return float(self._dist[i, j])