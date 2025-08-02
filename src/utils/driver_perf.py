from __future__ import annotations

import pandas as pd


def lateness_score(route_df: pd.DataFrame) -> float:
    """Return a 0‑1 score based on historical lateness (higher = worse)."""
    lateness = (route_df["actual_end"] - route_df["planned_end"]).clip(lower=0)
    minutes_late = lateness.dt.total_seconds() / 60
    if minutes_late.empty:
        return 0.0
    return float(minutes_late.mean() / 60)  # normalise via 60‑minute window