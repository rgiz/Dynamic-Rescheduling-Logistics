# Model Card

## Model Description

**Input:**

- Processed route and trip data in CSV format
- Distance/time matrices in `.npz` format
- List of disrupted trips with timings and locations

**Output:**

- Reassignment plan mapping disrupted trips to drivers or outsourcing
- Cost and service impact metrics

**Model Architecture:**

- Candidate generation with cascading logic
- OR-Tools CP-SAT solver for hard/soft constraints
- Multi-objective cost function (cost vs service)
- Bayesian optimization for weight tuning

## Performance

Example (20 disrupted trips, simulated):

| Metric         | Baseline | Optimized |
| -------------- | -------- | --------- |
| Success Rate   | 100%     | 100%      |
| Total Cost     | £1260    | £1016     |
| Cost Reduction | –        | 19.4%     |

**Measured On:** Simulated disruptions over processed weekly schedules.  
**Metric Definitions:**

- **Success Rate**: % of disrupted trips reassigned
- **Total Cost**: Sum of delay, deadhead, reassignment, emergency rest, outsourcing costs.

## Limitations

- Solver is CPU-bound; runtimes grow with number of disruptions.
- Requires accurate and up-to-date distance/time matrices.
- Bayesian tuning adds extra runtime overhead.

## Trade-offs

- Higher service quality weights can increase cost.
- Lower cost weights may cause more delays.
- Cascading swaps increase solution space (better solutions but longer solve times).
