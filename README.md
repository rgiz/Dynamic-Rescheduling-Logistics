# Dynamic Trip Rescheduling System

A sophisticated multi-objective optimization system for reassigning disrupted logistics trips while maintaining regulatory compliance and minimizing operational costs.

## Overview

When trips lose their assigned drivers due to disruptions (illness, vehicle breakdown, etc.), this system intelligently reassigns them to available drivers using a cascading optimization approach that considers multi-day schedules, driver capacity, and regulatory constraints.

## Key Features

### 🚛 **Multi-Driver Cascading Optimization**

- Automatically explores driver reassignment chains (Driver A → Disrupted Trip → Driver B takes A's original trip)
- Prefers reassignments over delays to maintain service quality
- Optimizes across multiple drivers simultaneously to find lowest-impact solutions

### ⏰ **Realistic Regulatory Compliance**

- **Daily Duty Limits**: 13-hour maximum working time per driver per day
- **Inter-Day Rest**: 11-hour minimum rest between working days (9-hour emergency fallback)
- **Weekend Breaks**: 45-hour minimum weekend rest with exact start times
- **Emergency Rest Quotas**: Maximum 2 emergency rests per driver per week

### 📊 **Multi-Objective Optimization**

- **Cost Minimization**: Deadhead travel, overtime, outsourcing costs
- **Service Quality**: Schedule adherence, delay minimization
- **Configurable Weights**: Bayesian optimization layer for cost vs. quality tradeoffs

### 🗓️ **Multi-Day Lookahead**

- Considers impacts across entire weekly schedules
- Up to 2-hour delay tolerance for subsequent trips
- Weekend break validation and protection

## System Architecture

### Data Processing Pipeline

```
Raw Logistics Data → Cleaned Segments → Aggregated Trips → Multi-Day Routes
```

- **Segments**: Individual delivery legs (origin → destination)
- **Trips**: Complete driver work days (8-13 hours, multiple segments)
- **Routes**: Multi-day driver assignments (spanning weeks)

### Optimization Components

#### 1. **Enhanced Candidate Generator** (`src/opt/candidate_gen.py`)

- Generates feasible insertion positions for disrupted trips
- Considers daily capacity utilization across multiple drivers
- Explores cascading reassignment opportunities
- Validates regulatory constraints (rest periods, duty limits)

#### 2. **Multi-Objective CP-SAT Model** (`src/opt/cpsat_model.py`)

- OR-Tools CP-SAT solver for complex constraint satisfaction
- Multi-driver assignment variables with cascading logic
- Cost vs. quality objective functions with configurable weights
- Emergency rest quota tracking per driver

#### 3. **Intelligent Loop Controller** (`src/opt/loop_controller.py`)

- Iterative optimization with increasing solution complexity
- Fallback strategies: reassignment → delay → emergency rest → outsourcing
- Multi-day impact assessment and weekend break protection

#### 4. **Bayesian Hyperparameter Tuning** (`src/opt/bayesian_tuner.py`)

- Automated cost vs. quality weight optimization
- Historical performance learning
- A/B testing framework for different optimization strategies

## Constraint Hierarchy

### Hard Constraints (Must Satisfy)

1. **Daily Duty**: ≤13 hours working time per driver per day
2. **Weekend Rest**: ≥45 hours, exact start times for new week
3. **Emergency Quotas**: ≤2 emergency rests per driver per week

### Soft Constraints (Prefer to Satisfy)

1. **Standard Rest**: 11-hour inter-day rest (vs. 9-hour emergency)
2. **Schedule Adherence**: ≤2 hours delay for subsequent trips
3. **Cost Minimization**: Minimize deadhead travel and reassignments

## Usage Examples

### Basic Disruption Handling

```python
from src.opt import LoopController, CandidateGenerator
from src.utils import DistanceMatrix

# Load data
df_routes = pd.read_csv("data/processed/routes.csv")
df_trips = pd.read_csv("data/processed/trips.csv")
disrupted_trips = simulate_disruptions(df_trips, n=5)

# Initialize optimizer
dist_matrix = DistanceMatrix("data/dist_matrix.npz")
generator = CandidateGenerator(dist_matrix)
controller = LoopController(generator, weights={'cost': 1.0, 'quality': 2.0})

# Optimize reassignments
assignments, outsourced = controller.reschedule(df_routes, disrupted_trips)
```

### Multi-Objective Optimization

```python
# Cost-focused optimization
cost_weights = {'deadhead': 1.0, 'delays': 5.0, 'outsource': 50.0}
cost_assignments, _ = controller.reschedule(df_routes, disrupted_trips, cost_weights)

# Quality-focused optimization
quality_weights = {'deadhead': 0.5, 'delays': 10.0, 'outsource': 100.0}
quality_assignments, _ = controller.reschedule(df_routes, disrupted_trips, quality_weights)
```

### Bayesian Weight Tuning

```python
from src.opt import BayesianTuner

tuner = BayesianTuner(historical_data="data/performance_history.csv")
optimal_weights = tuner.optimize_weights(
    objective='balanced',  # 'cost', 'quality', or 'balanced'
    n_trials=100
)
```

## Data Requirements

### Input Data Structure

- **df_cleaned**: Granular delivery segments with OSRM routing data
- **df_trips**: Aggregated driver work days (8-13 hours each)
- **df_routes**: Multi-day driver schedules spanning weeks
- **center_coordinates**: Geographic locations for distance calculations

### Key Columns

```python
# Trips (driver work days)
'trip_uuid', 'route_schedule_uuid', 'od_start_time', 'od_end_time',
'trip_duration_minutes', 'source_center', 'destination_center'

# Routes (multi-day schedules)
'route_schedule_uuid', 'route_start_time', 'route_end_time',
'route_total_time', 'num_trips'
```

## Performance Metrics

### Solution Quality

- **Feasibility Rate**: % of disrupted trips successfully reassigned
- **Service Impact**: Average delay to existing scheduled trips
- **Driver Utilization**: % of available daily capacity used
- **Emergency Rest Usage**: % of weekly emergency quota consumed

### Operational Efficiency

- **Total Deadhead**: Extra travel time/distance from reassignments
- **Outsourcing Cost**: Trips requiring external vendor coverage
- **Optimization Runtime**: Time to find solutions
- **Solution Stability**: Consistency across multiple runs

## Installation & Setup

```bash
# Clone repository
git clone <repository-url>
cd dynamic-trip-rescheduling

# Install dependencies
pip install -r requirements.txt

# Generate distance matrix
python scripts/generate_distance_matrix.py

# Run optimization demo
python scripts/run_rescheduler.py --disruptions 10 --strategy balanced
```

## Configuration

### Regulatory Parameters

```python
DAILY_DUTY_LIMIT = 13 * 60  # minutes
STANDARD_REST = 11 * 60     # minutes
EMERGENCY_REST = 9 * 60     # minutes
WEEKEND_REST = 45 * 60      # minutes
MAX_EMERGENCY_PER_WEEK = 2
MAX_DELAY_TOLERANCE = 2 * 60  # minutes
```

### Optimization Weights

```python
DEFAULT_WEIGHTS = {
    'deadhead_cost': 1.0,      # per minute of extra travel
    'delay_penalty': 5.0,      # per minute of schedule delay
    'reassignment_cost': 10.0, # per driver reassignment
    'emergency_rest_penalty': 50.0,  # per emergency rest used
    'outsourcing_cost': 200.0  # per trip outsourced
}
```

## Future Enhancements

- **Real-time Updates**: Integration with live traffic and driver status
- **Machine Learning**: Predictive models for disruption likelihood
- **Driver Preferences**: Incorporation of driver availability and preferences
- **Fleet Optimization**: Integration with vehicle assignment optimization
- **API Integration**: REST API for real-time optimization requests

## Contributing

See `CONTRIBUTING.md` for development guidelines and testing procedures.

## License

Apache License - see `LICENSE` file for details.
