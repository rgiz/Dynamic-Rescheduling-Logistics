# Dual Distance/Time Matrix Analysis

## MAJOR UPDATE: DUAL MATRIX STRUCTURE
This version generates TWO separate matrices:
- **distance_km**: For cost calculations (£ per kilometer of deadhead)
- **time_minutes**: For service impact calculations (minutes of delay)

## Matrix Statistics

| Metric | Value |
|--------|--------|
| Total Locations | 1,657 |
| Total Segments Processed | 144,867 |
| Unique Location Pairs | 4,688 |
| Matrix Coverage | 0.2% |
| Missing Connections | 2,739,304 |
| No-Connection Flag | -999 |

## Distance Analysis (Cost Basis)

| Metric | Value |
|--------|--------|
| Distance Range | 5.0 - 223.3 km |
| Mean Distance | 20.7 km |
| Median Distance | 22.4 km |

## Time Analysis (Service Impact)

| Metric | Value |
|--------|--------|
| Time Range | 3.0 - 208.0 minutes |
| Mean Time | 17.9 minutes |
| Median Time | 17.0 minutes |

## Location Connectivity

| Metric | Value |
|--------|--------|
| Average Connections | 2.8 |
| Max Connections | 61 |
| Zero Connections | 0 |

## Connectivity Tiers

| Tier | Count | Percentage | Description |
|------|-------|------------|-------------|
| Hub | 8 | 0.5% | >30 connections - Major routing centers |
| High | 34 | 2.1% | 11-30 connections - Regional hubs |
| Medium | 529 | 31.9% | 3-10 connections - Well connected |
| Low | 1086 | 65.5% | 1-2 connections - Limited connectivity |


## Implementation Notes for Code Updates

### CRITICAL: Update Matrix Loading Code
```python
# OLD (broken):
matrix_data = np.load('dist_matrix.npz')
distance_matrix = matrix_data['dist']  # This was confusing time/distance

# NEW (fixed):
matrix_data = np.load('dist_matrix.npz')
distance_km_matrix = matrix_data['distance_km']    # For cost calculations
time_minutes_matrix = matrix_data['time_minutes']  # For delay calculations
```

### For Candidate Generation:
1. **Cost calculations**: Use `distance_km_matrix` × £ per km
2. **Delay calculations**: Use `time_minutes_matrix` (minutes)
3. **No-connection handling**: Both matrices use -999 flag
4. **Hub-spoke logic**: Preserved in both matrices

### Next Phase Requirements:
1. Update `CandidateGeneratorV2` to load both matrices
2. Fix `_calculate_travel_time()` to use time matrix
3. Add `_calculate_travel_distance()` to use distance matrix
4. Update all cost calculations to use kilometers, not estimated miles

### Quality Metrics:
- **0.2% coverage** from segment data
- **0 isolated locations** - handle with outsourcing
- **42 well-connected locations** - prioritize for assignments
- **Dual matrix structure** enables proper cost vs service optimization
