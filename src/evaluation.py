from __future__ import annotations

import pandas as pd
from typing import List

from utils.distance_matrix import DistanceMatrix
from opt.candidate_gen import CandidateGenerator, CandidateInsertion


def evaluate_feasible_insertions(
    disrupted_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    dist_npz: str | None = "data/dist_matrix.npz",
) -> pd.DataFrame:
    """
    Return a tidy DataFrame listing every *feasible* insertion of each disrupted
    trip into each route, according to the 30-min break, ≤60-min slip, ≤12-h duty
    rules baked into CandidateGenerator.

    Parameters
    ----------
    disrupted_df : DataFrame
        Rows = trips produced by `simulate_new_jobs`.
    routes_df : DataFrame
        One row per route (driver) **OR** an exploded table where each row is a
        trip already on that route.  Either way it must contain
        `route_id, trip_id, start_loc, start_time, end_loc, end_time, duration_min`.
    dist_npz : str
        Path to `dist_matrix.npz` generated earlier.

    Returns
    -------
    DataFrame with columns:
        ['disrupted_trip', 'route_id', 'position',
         'deadhead_prev', 'deadhead_next', 'extra_duty']
    """
    dist = DistanceMatrix(dist_npz)
    gen = CandidateGenerator(dist)

    # explode routes_df into one DataFrame **per route** (needed by generator)
    route_tables: List[pd.DataFrame] = [
        grp.reset_index(drop=True)
        for _, grp in routes_df.sort_values("start_time").groupby("route_id")
    ]

    rows = []
    for _, trip in disrupted_df.iterrows():
        for route_df in route_tables:
            cands = gen.generate(route_df, trip)
            for c in cands:
                if not c.feasible:
                    continue
                rows.append(
                    dict(
                        disrupted_trip=c.trip_id,
                        route_id=c.route_id,
                        position=c.position,
                        deadhead_prev=c.deadhead_prev,
                        deadhead_next=c.deadhead_next,
                        extra_duty=c.extra_duty,
                    )
                )

    return pd.DataFrame(rows)