from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from ortools.sat.python import cp_model

from .candidate_gen import CandidateGenerator, CandidateInsertion
from .cpsat_model import RescheduleModel


class LoopController:
    """Two‑iteration loop; trips still unplaced → outsourced."""

    def __init__(self, gen: CandidateGenerator, weights: dict):
        self.gen = gen
        self.w = weights

    def reschedule(
        self, routes: List[pd.DataFrame], disrupted: pd.DataFrame
    ) -> Tuple[dict[str, CandidateInsertion], List[str]]:
        remaining = disrupted.copy()
        placed: dict[str, CandidateInsertion] = {}
        depth = 0
        while not remaining.empty and depth < 2:
            depth += 1
            cands: List[CandidateInsertion] = []
            for _, trip in remaining.iterrows():
                for r_df in routes:
                    cands.extend(self.gen.generate(r_df, trip))
            model = RescheduleModel(cands, self.w)
            status, solver = model.solve()
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                break
            newly_placed = set()
            for idx, var in model._var.items():
                if solver.BooleanValue(var):
                    ins = model.cands[idx]
                    placed[ins.trip_id] = ins
                    newly_placed.add(ins.trip_id)
            remaining = remaining[~remaining["trip_id"].isin(newly_placed)]
        outsourced = remaining["trip_id"].tolist()
        return placed, outsourced