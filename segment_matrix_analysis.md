# Segment-Based Distance Matrix Analysis

## Matrix Statistics

| Metric | Value |
|--------|--------|
| Total Locations | 1,657 |
| Total Segments Processed | 144,867 |
| Unique Location Pairs | 4,688 |
| Matrix Coverage | 0.2% |
| Missing Connections | 2,739,304 |
| No-Connection Flag | -999 |

## Distance Analysis (Real Connections Only)

| Metric | Value |
|--------|--------|
| Distance Range | 5.0 - 223.3 km |
| Mean Distance | 20.7 km |
| Median Distance | 22.4 km |

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


## Implementation Notes

### For Candidate Generation:
1. **Check for no-connection flag**: `if travel_time == -999: skip_candidate()`
2. **Prefer well-connected locations**: Use `location_connectivity.csv` to prioritize
3. **Implement distance limits**: Reject candidates with excessive travel times
4. **Hub-based routing**: Route through 'hub' and 'high' tier locations

### Matrix Usage:
- **Real connections**: Use directly for routing
- **Missing connections (-999)**: Implement fallback strategy
- **Connectivity tiers**: Guide assignment preferences

### Quality Metrics:
- **0.2% coverage** from segment data - much better than trip-level aggregation
- **0 isolated locations** - handle separately
- **42 well-connected locations** - prioritize for assignments
