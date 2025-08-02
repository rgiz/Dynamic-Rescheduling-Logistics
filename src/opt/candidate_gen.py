"""Directory: src/opt/candidate_gen.py"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from utils.distance_matrix import DistanceMatrix

# policy constants -----------------------------------------------------------
BREAK_MIN = 30
DUTY_MAX_MIN = 12 * 60
SLIP_MAX_MIN = 60


@dataclass
class CandidateInsertion:
    route_id: str
    trip_id: str
    position: int              # insertion index (0 … len(route))
    deadhead_prev: float       # minutes prev_end ➜ new_start
    deadhead_next: float       # minutes new_end ➜ next_start (if any)
    extra_duty: float          # minutes added to duty window
    feasible: bool


class CandidateGenerator:
    """Generate *all* insertion positions of a disrupted trip into a route.

    Feasibility rules:
      • driver takes BREAK_MIN before starting disrupted trip
      • resulting downstream start‑time slip ≤ SLIP_MAX_MIN
      • total duty ≤ DUTY_MAX_MIN
    If inserted at EOS (end‑of‑shift) there is no slip constraint.
    """

    def __init__(self, dist: DistanceMatrix):
        self.dist = dist

    # helper --------------------------------------------------------------
    @staticmethod
    def _as_min(value) -> float:
        if isinstance(value, pd.Timestamp):
            return value.value / 6e10  # ns ➜ min
        elif isinstance(value, pd.Timedelta):
            return value.total_seconds() / 60
        return float(value)

    # public --------------------------------------------------------------
    def generate(self, route: pd.DataFrame, trip: pd.Series) -> List[CandidateInsertion]:
        if route.empty:
            return []

        route = route.sort_values("start_time").reset_index(drop=True)
        first_start = self._as_min(route.loc[0, "start_time"])
        last_end = self._as_min(route.loc[route.index[-1], "end_time"])
        duty_planned = last_end - first_start

        t_id = trip["trip_id"]
        t_start_plan = self._as_min(trip["start_time"])
        t_dur = float(trip["duration_min"])
        t_end_plan = t_start_plan + t_dur
        s_loc, e_loc = int(trip["start_loc"]), int(trip["end_loc"])

        results: List[CandidateInsertion] = []

        for pos in range(len(route) + 1):
            # prev anchor -------------------------------------------------
            if pos == 0:
                prev_end_t = first_start
                prev_end_loc = int(route.loc[0, "start_loc"])
            else:
                prev_end_t = self._as_min(route.loc[pos - 1, "end_time"])
                prev_end_loc = int(route.loc[pos - 1, "end_loc"])

            dead_prev = self.dist.time(prev_end_loc, s_loc)
            start_actual = prev_end_t + dead_prev + BREAK_MIN
            feasible = start_actual <= t_start_plan

            dead_next = 0.0
            extra_duty = 0.0
            if feasible and pos < len(route):
                next_start_plan = self._as_min(route.loc[pos, "start_time"])
                next_start_loc = int(route.loc[pos, "start_loc"])

                dead_next = self.dist.time(e_loc, next_start_loc)
                next_start_actual = t_end_plan + dead_next
                slip = next_start_actual - next_start_plan
                if slip > SLIP_MAX_MIN:
                    feasible = False
                extra_duty = max(0.0, slip)
            else:
                # inserted at end
                extra_duty = dead_prev + BREAK_MIN + t_dur

            if feasible and duty_planned + extra_duty > DUTY_MAX_MIN:
                feasible = False

            results.append(
                CandidateInsertion(
                    route_id=route.loc[0, "route_id"] if "route_id" in route.columns else "r?",
                    trip_id=t_id,
                    position=pos,
                    deadhead_prev=dead_prev,
                    deadhead_next=dead_next,
                    extra_duty=extra_duty,
                    feasible=feasible,
                )
            )
        return results