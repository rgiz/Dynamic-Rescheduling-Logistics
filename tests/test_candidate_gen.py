"""Directory: tests/test_candidate_gen.py"""
import pandas as pd
from pandas import Timestamp

from utils.distance_matrix import DistanceMatrix
from opt.candidate_gen import CandidateGenerator


class DummyDist(DistanceMatrix):
    def __init__(self):
        pass

    def time(self, i: int, j: int):
        return 10  # 10 min everywhere

    def dist(self, i: int, j: int):
        return 5


def test_insertion_feasible():
    gen = CandidateGenerator(DummyDist())
    route = pd.DataFrame(
        {
            "route_id": ["R1"],
            "trip_id": ["T0"],
            "start_loc": [0],
            "start_time": [Timestamp("2025-08-02 08:00")],
            "end_loc": [1],
            "end_time": [Timestamp("2025-08-02 09:00")],
            "duration_min": [60],
        }
    )
    trip = pd.Series(
        {
            "trip_id": "D1",
            "start_loc": 1,
            "start_time": Timestamp("2025-08-02 10:00"),
            "end_loc": 2,
            "duration_min": 45,
        }
    )
    cands = gen.generate(route, trip)
    assert any(c.feasible for c in cands)