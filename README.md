# Dynamic Trip Rescheduling System

A sophisticated multi-objective optimization system for reassigning disrupted logistics trips while maintaining regulatory compliance and minimizing operational costs.

## Quick Start

```bash
# Clone repository
git clone <repository-url>
cd dynamic-trip-rescheduling

# Install dependencies
pip install -r requirements.txt

# Generate distance matrix
python scripts/generate_distance_matrix.py

# Run End to End optimization demo
# 1. Run cells in notebooks/data_preprocessing.ipynb
# 2. Set optimization weights in top cell of notebooks/dynamic_rescheduling_system.ipynb
# 3. Run main execution cell in notebooks/dynamic_rescheduling_system.ipynb
```

## Overview

When trips lose their assigned drivers due to disruptions, this system intelligently reassigns them to available drivers using a cascading optimization approach that considers multi-day schedules, driver capacity, and regulatory constraints. This is a proof of concept, which demonstrates CPSAT with Optuna and optimization via Bayesian Tuning.

**Core Features:**

- Multi-driver cascading optimization
- Realistic regulatory compliance
- Multi-objective optimization with configurable weights
- Multi-day lookahead to protect weekend rest and minimize knock-on effects

## Notebook Demo

See `notebooks/dynamic_trip_rescheduler.ipynb` for a complete end-to-end demonstration, including baseline and optimized scenarios.
Data pre-processing is handled in `notebooks/data_preprocessing.ipynb` and visualization of the basic data can be found in `notebooks/data_viz.ipynb`

<details>
<summary>Constraints</summary>

### Hard Constraints

- Daily Duty: ≤13 hours/day
- Weekend Rest: ≥45 hours
- Emergency Rests: ≤2/week

### Soft Constraints

- Inter-day rest: ≥11 hours (≥9 emergency)
- Delay tolerance: ≤2 hours
- Cost minimization: deadhead travel, reassignments
</details>

<details>
<summary>Configuration Parameters : User defined in Dynamic Trip Rescheduling Notebook</summary>

```python
DEFAULT_WEIGHTS = {
    'deadhead_cost': 1.0,
    'delay_penalty': 5.0,
    'reassignment_cost': 10.0,
    'emergency_rest_penalty': 50.0,
    'outsourcing_cost': 200.0
}
```

</details>

## Architecture

- **Candidate Generator** – finds feasible insertions and cascading swaps.
- **CP-SAT Solver** – constraint programming engine from OR-Tools.
- **Bayesian Tuner** – adjusts cost/quality weights automatically.
- **Loop Controller** – escalation from reassignment to delay to outsourcing.

## Performance (from example run)

| Metric         | Baseline | Optimized |
| -------------- | -------- | --------- |
| Success Rate   | 100%     | 100%      |
| Total Cost     | £1260    | £1016     |
| Cost Reduction | –        | 19.4%     |

## Future Work

- Upgrade to cuOpt for a production-ready system with better latency
- Integrated UI for Operators to enable decision making
- Real-time updates with live traffic
- ML disruption prediction
- Driver preference handling
- Vehicle assignment integration
