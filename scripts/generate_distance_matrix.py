#!/usr/bin/env python3
"""
Segment-Based Distance Matrix Generation - FIXED VERSION
=======================================================

MAJOR CHANGES:
- Now generates TWO separate matrices: distance_km and time_minutes
- Saves both matrices in dist_matrix.npz
- Maintains hub-spoke logic with -999 no-connection flags
- Properly handles units: kilometers for cost calculations, minutes for delays

Usage (from repo root):
    python3 scripts/generate_distance_matrix.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

class SegmentBasedMatrixGenerator:
    """Generates DUAL distance/time matrices from segment-level delivery data."""
    
    def __init__(self, no_connection_flag=-999):
        """
        Initialize generator.
        
        Args:
            no_connection_flag: Sentinel value for missing connections (default: -999)
                               This flag can be detected by candidate generation logic
        """
        self.no_connection_flag = no_connection_flag
        self.segments_df = None
        self.all_locations = None
        self.location_to_index = None
        self.location_connectivity = None
        self.distance_matrix_km = None  # NEW: Distance in kilometers
        self.time_matrix_min = None     # NEW: Time in minutes
        
    def load_segment_data(self, project_root: Path):
        """Load segment-level data from cleaned.csv."""
        print("📊 LOADING SEGMENT DATA")
        print("=" * 30)
        
        cleaned_path = project_root / "data" / "processed" / "cleaned.csv"
        if not cleaned_path.exists():
            raise FileNotFoundError(f"Cleaned data not found: {cleaned_path}")
        
        print(f"✅ Loading: {cleaned_path}")
        df_cleaned = pd.read_csv(cleaned_path)
        
        # Verify required columns
        required_cols = ['source_center', 'destination_center', 'segment_osrm_distance']
        missing_cols = [col for col in required_cols if col not in df_cleaned.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract segment data with optional time column
        segment_cols = ['source_center', 'destination_center', 'segment_osrm_distance']
        if 'segment_osrm_time' in df_cleaned.columns:
            segment_cols.append('segment_osrm_time')
            print("✅ Found segment_osrm_time column - will use real time data")
        else:
            print("⚠️ No segment_osrm_time - will estimate from distance at 60 km/h")
        
        # Filter to valid segments
        self.segments_df = df_cleaned[segment_cols].copy()
        self.segments_df = self.segments_df[
            (self.segments_df['source_center'] != self.segments_df['destination_center']) &
            self.segments_df['segment_osrm_distance'].notna()
        ]
        
        print(f"✅ Loaded {len(self.segments_df):,} valid segments")
        
        # Get all unique locations
        self.all_locations = sorted(
            set(self.segments_df['source_center']).union(set(self.segments_df['destination_center']))
        )
        self.location_to_index = {loc: i for i, loc in enumerate(self.all_locations)}
        
        print(f"✅ Found {len(self.all_locations)} unique locations")
        
        return self.segments_df
    
    def calculate_location_connectivity(self):
        """Calculate connectivity statistics for each location."""
        print("\n🔗 CALCULATING LOCATION CONNECTIVITY")
        print("=" * 30)
        
        connections = defaultdict(set)
        connection_distances = defaultdict(list)
        
        # Build connectivity graph
        for _, row in self.segments_df.iterrows():
            src, dst, dist = row['source_center'], row['destination_center'], row['segment_osrm_distance']
            connections[src].add(dst)
            connections[dst].add(src)
            connection_distances[src].append(dist)
            connection_distances[dst].append(dist)
        
        # Calculate connectivity stats
        self.location_connectivity = {}
        
        for location in self.all_locations:
            connection_count = len(connections[location])
            distances = connection_distances[location]
            
            stats = {
                'location': location,
                'connection_count': connection_count,
                'avg_segment_distance': np.mean(distances) if distances else 0,
                'min_segment_distance': min(distances) if distances else 0,
                'max_segment_distance': max(distances) if distances else 0,
                'connectivity_tier': self._classify_connectivity(connection_count)
            }
            
            self.location_connectivity[location] = stats
        
        # Summary stats
        conn_counts = [stats['connection_count'] for stats in self.location_connectivity.values()]
        
        print(f"✅ Connectivity analysis complete:")
        print(f"   Average connections per location: {np.mean(conn_counts):.1f}")
        print(f"   Max connections: {max(conn_counts)}")
        print(f"   Locations with 0 connections: {sum(1 for c in conn_counts if c == 0)}")
        print(f"   Locations with >10 connections: {sum(1 for c in conn_counts if c > 10)}")
        
        return self.location_connectivity
    
    def _classify_connectivity(self, connection_count):
        """Classify location connectivity into tiers."""
        if connection_count == 0:
            return 'isolated'
        elif connection_count <= 2:
            return 'low'
        elif connection_count <= 10:
            return 'medium'
        elif connection_count <= 30:
            return 'high'
        else:
            return 'hub'
    
    def build_dual_matrices(self):
        """Build SEPARATE distance and time matrices from segment data."""
        print(f"\n🏗️ BUILDING DUAL DISTANCE/TIME MATRICES")
        print("=" * 30)
        
        n = len(self.all_locations)
        
        # Initialize both matrices with no-connection flags
        self.distance_matrix_km = np.full((n, n), self.no_connection_flag, dtype=np.float32)
        self.time_matrix_min = np.full((n, n), self.no_connection_flag, dtype=np.float32)
        
        # Fill diagonals (same location = 0)
        np.fill_diagonal(self.distance_matrix_km, 0)
        np.fill_diagonal(self.time_matrix_min, 0)
        
        # Aggregate segment data (median for multiple segments between same locations)
        print("   Aggregating segment pairs...")
        agg_funcs = {'segment_osrm_distance': 'median'}
        if 'segment_osrm_time' in self.segments_df.columns:
            agg_funcs['segment_osrm_time'] = 'median'
        
        segment_pairs = self.segments_df.groupby(['source_center', 'destination_center']).agg(agg_funcs).reset_index()
        
        print(f"   Aggregated to {len(segment_pairs):,} unique location pairs")
        
        # Fill matrices with direct connections
        connections_added = 0
        for _, row in segment_pairs.iterrows():
            src, dst = row['source_center'], row['destination_center']
            distance_km = row['segment_osrm_distance']  # KILOMETERS
            
            if src in self.location_to_index and dst in self.location_to_index:
                i, j = self.location_to_index[src], self.location_to_index[dst]
                
                # DISTANCE MATRIX: Store kilometers (for cost calculations)
                self.distance_matrix_km[i, j] = distance_km
                self.distance_matrix_km[j, i] = distance_km  # Make symmetric
                
                # TIME MATRIX: Store minutes (for delay calculations)
                if 'segment_osrm_time' in row.index and pd.notna(row['segment_osrm_time']):
                    time_min = row['segment_osrm_time']  # Real time data in minutes
                else:
                    # Estimate time from distance at 60 km/h
                    time_min = (distance_km / 60) * 60  # km / (km/h) * (min/h) = minutes
                
                self.time_matrix_min[i, j] = time_min
                self.time_matrix_min[j, i] = time_min  # Make symmetric
                
                connections_added += 1
        
        print(f"✅ Added {connections_added:,} direct connections to both matrices")
        
        # Calculate coverage statistics
        total_pairs = n * (n - 1)  # Exclude diagonal
        real_connections = (self.distance_matrix_km != self.no_connection_flag).sum() - n  # Exclude diagonal
        coverage_percent = real_connections / total_pairs * 100
        
        print(f"📊 Matrix Statistics:")
        print(f"   Matrix size: {n} × {n}")
        print(f"   Real connections: {real_connections:,} ({coverage_percent:.1f}%)")
        print(f"   Missing connections: {total_pairs - real_connections:,} (flagged as {self.no_connection_flag})")
        
        # Show statistics for real connections only
        real_distances = self.distance_matrix_km[(self.distance_matrix_km != self.no_connection_flag) & (self.distance_matrix_km > 0)]
        real_times = self.time_matrix_min[(self.time_matrix_min != self.no_connection_flag) & (self.time_matrix_min > 0)]
        
        if len(real_distances) > 0:
            print(f"   Distance range: {real_distances.min():.1f} - {real_distances.max():.1f} km")
            print(f"   Mean distance: {real_distances.mean():.1f} km")
        
        if len(real_times) > 0:
            print(f"   Time range: {real_times.min():.1f} - {real_times.max():.1f} minutes")
            print(f"   Mean time: {real_times.mean():.1f} minutes")
        
        return self.distance_matrix_km, self.time_matrix_min
    
    def save_dual_matrices_and_metadata(self, project_root: Path):
        """Save BOTH distance and time matrices plus connectivity metadata."""
        print(f"\n💾 SAVING DUAL MATRICES AND METADATA")
        print("=" * 30)
        
        # Save dual matrices in single file
        dist_path = project_root / "data" / "dist_matrix.npz"
        dist_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            dist_path,
            ids=np.array(self.all_locations),
            distance_km=self.distance_matrix_km,    # NEW: Distance in kilometers
            time_minutes=self.time_matrix_min,      # NEW: Time in minutes
            no_connection_flag=self.no_connection_flag,
            # DEPRECATED: Remove these old fields after migration
            dist=self.distance_matrix_km,  # Backward compatibility (temporary)
            time=self.time_matrix_min      # Backward compatibility (temporary)
        )
        
        print(f"✅ Dual matrices saved: {dist_path}")
        print(f"   - distance_km: Cost calculations (£ per km)")
        print(f"   - time_minutes: Delay calculations (service impact)")
        
        # Save connectivity metadata
        connectivity_data = []
        for location in self.all_locations:
            stats = self.location_connectivity[location]
            connectivity_data.append({
                'location_id': location,
                'connection_count': stats['connection_count'],
                'connectivity_tier': stats['connectivity_tier'],
                'avg_segment_distance': stats['avg_segment_distance'],
                'min_segment_distance': stats['min_segment_distance'],
                'max_segment_distance': stats['max_segment_distance']
            })
        
        connectivity_df = pd.DataFrame(connectivity_data)
        connectivity_path = project_root / "data" / "processed" / "location_connectivity.csv"
        connectivity_df.to_csv(connectivity_path, index=False)
        
        print(f"✅ Connectivity metadata saved: {connectivity_path}")
        
        # Show connectivity tier distribution
        tier_counts = connectivity_df['connectivity_tier'].value_counts()
        print(f"\n📊 Connectivity Tier Distribution:")
        for tier, count in tier_counts.items():
            print(f"   {tier:>8}: {count:4d} locations ({count/len(connectivity_df)*100:.1f}%)")
        
        return dist_path, connectivity_path
    
    def create_analysis_report(self, project_root: Path):
        """Create analysis report for the generated dual matrices."""
        
        # Calculate summary statistics
        total_locations = len(self.all_locations)
        total_segments = len(self.segments_df)
        real_connections = (self.distance_matrix_km != self.no_connection_flag).sum() - total_locations
        total_pairs = total_locations * (total_locations - 1)
        coverage = real_connections / total_pairs * 100
        
        # Connectivity stats
        conn_counts = [stats['connection_count'] for stats in self.location_connectivity.values()]
        tier_counts = pd.Series([stats['connectivity_tier'] for stats in self.location_connectivity.values()]).value_counts()
        
        # Real distance and time stats
        real_distances = self.distance_matrix_km[(self.distance_matrix_km != self.no_connection_flag) & (self.distance_matrix_km > 0)]
        real_times = self.time_matrix_min[(self.time_matrix_min != self.no_connection_flag) & (self.time_matrix_min > 0)]
        
        report = f"""# Dual Distance/Time Matrix Analysis

## MAJOR UPDATE: DUAL MATRIX STRUCTURE
This version generates TWO separate matrices:
- **distance_km**: For cost calculations (£ per kilometer of deadhead)
- **time_minutes**: For service impact calculations (minutes of delay)

## Matrix Statistics

| Metric | Value |
|--------|--------|
| Total Locations | {total_locations:,} |
| Total Segments Processed | {total_segments:,} |
| Unique Location Pairs | {real_connections:,} |
| Matrix Coverage | {coverage:.1f}% |
| Missing Connections | {total_pairs - real_connections:,} |
| No-Connection Flag | {self.no_connection_flag} |

## Distance Analysis (Cost Basis)

| Metric | Value |
|--------|--------|
| Distance Range | {real_distances.min():.1f} - {real_distances.max():.1f} km |
| Mean Distance | {real_distances.mean():.1f} km |
| Median Distance | {np.median(real_distances):.1f} km |

## Time Analysis (Service Impact)

| Metric | Value |
|--------|--------|
| Time Range | {real_times.min():.1f} - {real_times.max():.1f} minutes |
| Mean Time | {real_times.mean():.1f} minutes |
| Median Time | {np.median(real_times):.1f} minutes |

## Location Connectivity

| Metric | Value |
|--------|--------|
| Average Connections | {np.mean(conn_counts):.1f} |
| Max Connections | {max(conn_counts)} |
| Zero Connections | {sum(1 for c in conn_counts if c == 0)} |

## Connectivity Tiers

| Tier | Count | Percentage | Description |
|------|-------|------------|-------------|
"""
        
        for tier in ['hub', 'high', 'medium', 'low', 'isolated']:
            if tier in tier_counts:
                count = tier_counts[tier]
                pct = count / total_locations * 100
                descriptions = {
                    'hub': '>30 connections - Major routing centers',
                    'high': '11-30 connections - Regional hubs',
                    'medium': '3-10 connections - Well connected',
                    'low': '1-2 connections - Limited connectivity',
                    'isolated': '0 connections - Disconnected locations'
                }
                report += f"| {tier.title()} | {count} | {pct:.1f}% | {descriptions[tier]} |\n"
        
        report += f"""

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
3. **No-connection handling**: Both matrices use {self.no_connection_flag} flag
4. **Hub-spoke logic**: Preserved in both matrices

### Next Phase Requirements:
1. Update `CandidateGeneratorV2` to load both matrices
2. Fix `_calculate_travel_time()` to use time matrix
3. Add `_calculate_travel_distance()` to use distance matrix
4. Update all cost calculations to use kilometers, not estimated miles

### Quality Metrics:
- **{coverage:.1f}% coverage** from segment data
- **{sum(1 for c in conn_counts if c == 0)} isolated locations** - handle with outsourcing
- **{sum(1 for c in conn_counts if c > 10)} well-connected locations** - prioritize for assignments
- **Dual matrix structure** enables proper cost vs service optimization
"""
        
        # Save report
        report_path = project_root / "dual_matrix_analysis.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Analysis report saved: {report_path}")
        return report_path

def main():
    """Generate segment-based dual distance/time matrices."""
    
    parser = argparse.ArgumentParser(description='Generate dual distance/time matrices from segment data')
    parser.add_argument('--no-connection-flag', type=float, default=-999,
                       help='Flag value for missing connections (default: -999)')
    parser.add_argument('--project-root', type=Path, default=None,
                       help='Project root directory (default: auto-detect)')
    
    args = parser.parse_args()
    
    print("🚛 DUAL DISTANCE/TIME MATRIX GENERATION")
    print("=" * 60)
    print("MAJOR UPDATE: Generates separate distance_km and time_minutes matrices")
    print("Strategy: Use actual delivery segments from cleaned.csv")
    print(f"No-connection flag: {args.no_connection_flag}")
    print("=" * 60)
    
    # Determine project root
    if args.project_root:
        project_root = args.project_root
    else:
        project_root = Path(__file__).parent.parent
    
    try:
        # Initialize generator
        generator = SegmentBasedMatrixGenerator(no_connection_flag=args.no_connection_flag)
        
        # Run generation pipeline
        generator.load_segment_data(project_root)
        generator.calculate_location_connectivity()
        generator.build_dual_matrices()  # NEW: Build both matrices
        dist_path, conn_path = generator.save_dual_matrices_and_metadata(project_root)  # NEW: Save both
        report_path = generator.create_analysis_report(project_root)
        
        print(f"\n" + "=" * 60)
        print("✅ DUAL MATRIX GENERATION COMPLETE!")
        print("=" * 60)
        
        print(f"\nFiles generated:")
        print(f"- {dist_path} (dual matrices: distance_km + time_minutes)")
        print(f"- {conn_path} (connectivity metadata)")
        print(f"- {report_path} (analysis report)")
        
        print(f"\n🚨 NEXT STEPS - PHASE 2:")
        print(f"1. Update CandidateGeneratorV2 to load both matrices")
        print(f"2. Fix _calculate_travel_time() to use time_minutes matrix")
        print(f"3. Add _calculate_travel_distance() to use distance_km matrix")
        print(f"4. Update cost calculations to use km not estimated miles")
        print(f"5. Test with: python3 val.py")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)