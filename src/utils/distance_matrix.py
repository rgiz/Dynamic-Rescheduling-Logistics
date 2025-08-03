from __future__ import annotations

import numpy as np
from pathlib import Path


class DistanceMatrix:
    """Lazy‑loads a symmetric travel‑time & distance matrix saved as a NumPy `.npz`.

    The archive must contain two identically‑shaped 2‑D arrays:
    * **time** – minutes between locations
    * **dist** – kilometres (or same unit) between locations
    The row/column index is expected to match the integer `location_id` field
    in your trips and route tables.  Loading is deferred until the first call
    to `time()` or `dist()` so that unit tests can mock the class cheaply.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._time: np.ndarray | None = None
        self._dist: np.ndarray | None = None

    # ------------------------------------------------------------------ internals
    def _ensure(self):
        if self._time is None or self._dist is None:
            data = np.load(self.path)
            self._time = data["time"]
            self._dist = data["dist"]

    # ------------------------------------------------------------------ api
    def time(self, i: int, j: int) -> float:  # minutes
        self._ensure()
        return float(self._time[i, j])

    def dist(self, i: int, j: int) -> float:  # km
        self._ensure()
        return float(self._dist[i, j])