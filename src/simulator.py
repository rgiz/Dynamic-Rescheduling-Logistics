import pandas as pd
import numpy as np

def simulate_new_jobs(df_trips: pd.DataFrame, n_jobs: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Simulate new jobs to be reassigned by sampling existing trips.
    These jobs will be treated as unallocated and needing coverage.
    """
    np.random.seed(seed)
    new_jobs = df_trips.sample(n=n_jobs).copy()
    new_jobs["job_id"] = ["NEW_JOB_" + str(i) for i in range(n_jobs)]
    new_jobs.reset_index(drop=True, inplace=True)
    return new_jobs