import pandas as pd

def evaluate_feasible_insertions(new_jobs: pd.DataFrame, 
                                 df_routes: pd.DataFrame, 
                                 time_limit: float = 720,  # e.g., 12 hours
                                 distance_limit: float = 400.0) -> pd.DataFrame:
    """
    For each new job, return all routes that could accommodate the job
    without breaching hard limits.
    """
    candidates = []
    for _, job in new_jobs.iterrows():
        for _, route in df_routes.iterrows():
            new_time = route['route_total_time'] + job['trip_duration_minutes']
            new_distance = route['route_total_distance'] + job['segment_osrm_distance']
            if new_time <= time_limit and new_distance <= distance_limit:
                candidates.append({
                    "job_id": job["job_id"],
                    "route_schedule_uuid": route["route_schedule_uuid"],
                    "new_total_time": new_time,
                    "new_total_distance": new_distance,
                    "original_time": route['route_total_time'],
                    "original_distance": route['route_total_distance'],
                    "added_time": job['trip_duration_minutes'],
                    "added_distance": job['segment_osrm_distance']
                })
    return pd.DataFrame(candidates)