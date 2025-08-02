from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from ortools.sat.python import cp_model

from .candidate_gen import CandidateInsertion


class RescheduleModel:
    """Wraps CP‑SAT model that picks one feasible insertion per disrupted trip."""

    def __init__(self, candidates: List[CandidateInsertion], weights: Dict[str, float]):
        self.cands = [c for c in candidates if c.feasible]
        self.w = weights
        self.model = cp_model.CpModel()
        self._var: dict[int, cp_model.IntVar] = {}
        self._build()

    def _build(self):
        m = self.model
        # vars ------------------------------------------------------------
        for idx, _ in enumerate(self.cands):
            self._var[idx] = m.NewBoolVar(f"x_{idx}")

        # each trip once --------------------------------------------------
        by_trip = defaultdict(list)
        for idx, c in enumerate(self.cands):
            by_trip[c.trip_id].append(idx)
        for idxs in by_trip.values():
            m.Add(sum(self._var[i] for i in idxs) == 1)

        # objective -------------------------------------------------------
        w1 = self.w.get("w1", 1.0)
        m.Minimize(sum(int(c.deadhead_prev * w1) * self._var[i] for i, c in enumerate(self.cands)))

    def solve(self, time_limit_s=5):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_s
        status = solver.Solve(self.model)
        return status, solver