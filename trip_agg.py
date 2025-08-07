#!/usr/bin/env python3
"""
Trip Aggregation Analysis & Fix
===============================

Investigate and fix trip aggregation issues:
1. Check if trips.csv is correctly aggregated from cleaned.csv
2. Verify start/end locations are properly identified
3. Create distance matrix from segment-level data (cleaned.csv)
4. Compare connectivity between approaches

This should eliminate locations with 0 connections.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_data_flow():
    """Analyze the data flow from cleaned -> trips -> matrix."""
    
    print("🔍 TRIP AGGREGATION ANALYSIS")
    print("=" * 50)
    
    # Load both datasets
    cleaned_path = Path("data/processed/cleaned.csv")
    trips_path = Path("data/processed/trips.csv")
    
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {cleaned_path}")
    if not trips_path.exists():
        raise FileNotFoundError(f"Trips data not found: {trips_path}")
    
    print(f"✅ Loading cleaned data: {cleaned_path}")
    df_cleaned = pd.read_csv(cleaned_path)
    
    print(f"✅ Loading trips data: {trips_path}")
    df_trips = pd.read_csv(trips_path)
    
    print(f"\n📊 DATA COMPARISON")
    print("-" * 30)
    print(f"Cleaned segments: {len(df_cleaned):,}")
    print(f"Aggregated trips: {len(df_trips):,}")
    print(f"Aggregation ratio: {len(df_cleaned)/len(df_trips):.1f} segments per trip")
    
    # Check columns
    print(f"\n📋 COLUMN ANALYSIS")
    print("-" * 30)
    print(f"Cleaned columns: {list(df_cleaned.columns)}")
    print(f"Trips columns: {list(df_trips.columns)}")
    
    # Check location columns specifically
    location_columns_cleaned = [col for col in df_cleaned.columns if 'center' in col.lower() or 'location' in col.lower()]
    location_columns_trips = [col for col in df_trips.columns if 'center' in col.lower() or 'location' in col.lower()]
    
    print(f"\nLocation columns in cleaned: {location_columns_cleaned}")
    print(f"Location columns in trips: {location_columns_trips}")
    
    return df_cleaned, df_trips

def analyze_trip_aggregation_quality(df_cleaned, df_trips):
    """Check if trip aggregation is working correctly."""
    
    print(f"\n🔍 TRIP AGGREGATION QUALITY CHECK")
    print("-" * 30)
    
    # Check if we have the right aggregation columns
    if 'trip_uuid' not in df_cleaned.columns:
        print("❌ No trip_uuid in cleaned data - cannot verify aggregation")
        return
    
    # Sample a few trips and check aggregation
    sample_trips = df_trips['trip_uuid'].head(10)
    
    print(f"Checking aggregation for {len(sample_trips)} sample trips...")
    
    for trip_id in sample_trips:
        print(f"\\nTrip: {trip_id}")
        
        # Get all segments for this trip
        trip_segments = df_cleaned[df_cleaned['trip_uuid'] == trip_id].copy()
        
        if len(trip_segments) == 0:
            print(f"  ❌ No segments found in cleaned data!")
            continue
            
        # Sort by time to get proper sequence
        if 'od_start_time' in trip_segments.columns:
            trip_segments['od_start_time'] = pd.to_datetime(trip_segments['od_start_time'])
            trip_segments = trip_segments.sort_values('od_start_time')
        
        print(f"  Segments: {len(trip_segments)}")
        
        # Check start/end locations
        if 'source_center' in trip_segments.columns and 'destination_center' in trip_segments.columns:
            actual_start = trip_segments.iloc[0]['source_center']
            actual_end = trip_segments.iloc[-1]['destination_center']
            
            # Get what trips.csv shows
            trip_row = df_trips[df_trips['trip_uuid'] == trip_id]
            if len(trip_row) > 0:
                recorded_start = trip_row.iloc[0]['source_center']
                recorded_end = trip_row.iloc[0]['destination_center']
                
                print(f"  Actual start: {actual_start}")
                print(f"  Recorded start: {recorded_start}")
                print(f"  Match: {actual_start == recorded_start}")
                
                print(f"  Actual end: {actual_end}")  
                print(f"  Recorded end: {recorded_end}")
                print(f"  Match: {actual_end == recorded_end}")
                
                if actual_start != recorded_start or actual_end != recorded_end:
                    print(f"  ❌ AGGREGATION ERROR!")
            else:
                print(f"  ❌ Trip not found in trips.csv")
        
        if len(sample_trips) > 3:  # Limit output
            break

def analyze_segment_connectivity(df_cleaned):
    """Analyze connectivity using segment-level data."""
    
    print(f"\n🗺️ SEGMENT-LEVEL CONNECTIVITY ANALYSIS")
    print("-" * 30)
    
    # Check what location columns we have
    location_cols = [col for col in df_cleaned.columns if 'center' in col.lower()]
    print(f"Available location columns: {location_cols}")
    
    if 'source_center' not in df_cleaned.columns or 'destination_center' not in df_cleaned.columns:
        print("❌ No source_center/destination_center in cleaned data")
        return None
    
    # Get all unique segments (direct connections)
    segments = df_cleaned[['source_center', 'destination_center']].copy()
    segments = segments[segments['source_center'] != segments['destination_center']]
    segments = segments.dropna()
    
    print(f"Total segments: {len(segments):,}")
    
    # Get unique locations from segments
    all_locations_segments = set(segments['source_center']).union(set(segments['destination_center']))
    print(f"Unique locations in segments: {len(all_locations_segments)}")
    
    # Count connections per location
    from collections import defaultdict
    connections = defaultdict(set)
    
    for _, row in segments.iterrows():
        src, dst = row['source_center'], row['destination_center'] 
        connections[src].add(dst)
        connections[dst].add(src)
    
    # Analyze connectivity
    connection_counts = {loc: len(conns) for loc, conns in connections.items()}
    
    # Sort by connectivity
    sorted_locations = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\\n📊 SEGMENT CONNECTIVITY STATS:")
    print(f"  Locations with 0 connections: {sum(1 for _, count in connection_counts.items() if count == 0)}")
    print(f"  Locations with 1 connection: {sum(1 for _, count in connection_counts.items() if count == 1)}")
    print(f"  Locations with >10 connections: {sum(1 for _, count in connection_counts.items() if count > 10)}")
    print(f"  Average connections: {np.mean(list(connection_counts.values())):.1f}")
    
    print(f"\\nTop 10 most connected locations (segments):")
    for loc, count in sorted_locations[:10]:
        print(f"  {loc[:15]:15}: {count:3d} connections")
    
    print(f"\\nLeast connected locations (segments):")
    for loc, count in sorted_locations[-5:]:
        print(f"  {loc[:15]:15}: {count:3d} connections")
    
    return segments, all_locations_segments, connection_counts

def compare_trip_vs_segment_connectivity(df_trips, segments_df):
    """Compare connectivity between trip-level and segment-level data."""
    
    print(f"\\n⚖️ CONNECTIVITY COMPARISON")
    print("-" * 30)
    
    # Trip-level connectivity
    trip_pairs = df_trips[['source_center', 'destination_center']].copy()
    trip_pairs = trip_pairs[trip_pairs['source_center'] != trip_pairs['destination_center']]
    trip_pairs = trip_pairs.dropna()
    
    trip_locations = set(trip_pairs['source_center']).union(set(trip_pairs['destination_center']))
    
    # Segment-level connectivity  
    segment_locations = set(segments_df['source_center']).union(set(segments_df['destination_center']))
    
    print(f"Trip-level analysis:")
    print(f"  Unique location pairs: {len(trip_pairs):,}")
    print(f"  Unique locations: {len(trip_locations)}")
    
    print(f"\\nSegment-level analysis:")
    print(f"  Unique location pairs: {len(segments_df):,}")
    print(f"  Unique locations: {len(segment_locations)}")
    
    print(f"\\nComparison:")
    print(f"  Segments have {len(segments_df) - len(trip_pairs):,} more connections")
    print(f"  Segments cover {len(segment_locations) - len(trip_locations)} more locations")
    
    # Find locations that appear in segments but not trips
    segment_only = segment_locations - trip_locations
    trip_only = trip_locations - segment_locations
    
    if segment_only:
        print(f"\\n  Locations only in segments: {len(segment_only)}")
        if len(segment_only) <= 10:
            print(f"    {list(segment_only)[:10]}")
    
    if trip_only:
        print(f"\\n  Locations only in trips: {len(trip_only)}")
        if len(trip_only) <= 10:
            print(f"    {list(trip_only)[:10]}")
    
    return trip_pairs, segments_df

def create_segment_based_matrix(segments_df, all_locations):
    """Create distance matrix from segment-level data."""
    
    print(f"\\n🏗️ CREATING SEGMENT-BASED DISTANCE MATRIX")
    print("-" * 30)
    
    # Check available distance/time columns
    distance_cols = [col for col in segments_df.columns if 'distance' in col.lower()]
    time_cols = [col for col in segments_df.columns if 'time' in col.lower()]
    
    print(f"Available distance columns: {distance_cols}")
    print(f"Available time columns: {time_cols}")
    
    # Use osrm_distance and osrm_time
    distance_col = None
    time_col = None
    
    for col in distance_cols:
        if 'osrm_distance' in col:
            distance_col = col
            break
    
    for col in time_cols:
        if 'osrm_time' in col:
            time_col = col
            break
    
    if distance_col is None:
        print("❌ No osrm_distance column found")
        print(f"   Available columns: {list(segments_df.columns)}")
        return None
    
    if time_col is None:
        print("⚠️ No osrm_time column found, will calculate from distance")
    
    print(f"✅ Using distance column: {distance_col}")
    if time_col:
        print(f"✅ Using time column: {time_col}")
    
    print(f"Building matrix for {len(all_locations)} locations...")
    
    # Group segments and take median distance/time for each pair
    agg_dict = {distance_col: 'median'}
    if time_col:
        agg_dict[time_col] = 'median'
    
    segment_stats = segments_df.groupby(['source_center', 'destination_center']).agg(agg_dict).reset_index()
    
    print(f"Unique segment pairs with distances: {len(segment_stats):,}")
    
    # Check distance statistics
    distances = segment_stats[distance_col].dropna()
    if len(distances) > 0:
        print(f"Distance range: {distances.min():.1f} - {distances.max():.1f} km")
        print(f"Mean distance: {distances.mean():.1f} km")
    
    # Build network graph for shortest paths (simplified)
    import networkx as nx
    
    G = nx.Graph()
    for _, row in segment_stats.iterrows():
        src, dst, dist = row['source_center'], row['destination_center'], row[distance_col]
        if pd.notna(dist) and dist > 0:
            G.add_edge(src, dst, weight=dist)
    
    print(f"Network graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Get connected components
    components = list(nx.connected_components(G))
    print(f"Connected components: {len(components)}")
    
    if len(components) > 1:
        component_sizes = [len(comp) for comp in components]
        print(f"Component sizes: {sorted(component_sizes, reverse=True)}")
        print(f"Largest component: {max(component_sizes)} locations")
    
    # Create distance matrix
    locations_list = sorted(all_locations)
    n = len(locations_list)
    location_to_index = {loc: i for i, loc in enumerate(locations_list)}
    
    distance_matrix = np.full((n, n), np.inf)
    time_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(distance_matrix, 0)
    np.fill_diagonal(time_matrix, 0)
    
    # Fill direct connections
    connections_added = 0
    for _, row in segment_stats.iterrows():
        src, dst = row['source_center'], row['destination_center']
        dist = row[distance_col]
        
        if pd.notna(dist) and src in location_to_index and dst in location_to_index:
            i, j = location_to_index[src], location_to_index[dst]
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Make symmetric
            
            # Handle time
            if time_col and pd.notna(row[time_col]):
                time_val = row[time_col]
                time_matrix[i, j] = time_val
                time_matrix[j, i] = time_val
            else:
                # Calculate time from distance at 60 km/h
                time_val = dist / 60 * 60  # km / (km/h) * (min/h) = minutes
                time_matrix[i, j] = time_val
                time_matrix[j, i] = time_val
            
            connections_added += 1
    
    print(f"Added {connections_added} direct connections")
    
    # Quick connectivity check
    finite_connections = np.isfinite(distance_matrix) & (distance_matrix > 0)
    coverage = finite_connections.sum() / (n * n - n) * 100  # Exclude diagonal
    
    print(f"Matrix coverage: {coverage:.1f}% direct connections")
    
    return distance_matrix, time_matrix, locations_list, location_to_index

def main():
    """Run complete analysis and create segment-based matrix."""
    
    try:
        # Step 1: Analyze data flow
        df_cleaned, df_trips = analyze_data_flow()
        
        # Step 2: Check trip aggregation quality
        analyze_trip_aggregation_quality(df_cleaned, df_trips)
        
        # Step 3: Analyze segment-level connectivity
        segments_df, segment_locations, segment_connections = analyze_segment_connectivity(df_cleaned)
        
        if segments_df is None:
            return 1
        
        # Step 4: Compare approaches
        trip_pairs, segments_df = compare_trip_vs_segment_connectivity(df_trips, segments_df)
        
        # Step 5: Create segment-based matrix
        result = create_segment_based_matrix(segments_df, segment_locations)
        
        if result is not None:
            distance_matrix, time_matrix, locations_list, location_to_index = result
            
            # Quick analysis of the new matrix
            non_diagonal = distance_matrix[~np.eye(len(locations_list), dtype=bool)]
            finite_distances = non_diagonal[np.isfinite(non_diagonal) & (non_diagonal > 0)]
            
            non_diagonal_time = time_matrix[~np.eye(len(locations_list), dtype=bool)]
            finite_times = non_diagonal_time[np.isfinite(non_diagonal_time) & (non_diagonal_time > 0)]
            
            if len(finite_distances) > 0:
                print(f"\\n📊 SEGMENT-BASED MATRIX STATS:")
                print(f"  Direct connections: {len(finite_distances):,}")
                print(f"  Distance range: {finite_distances.min():.1f} - {finite_distances.max():.1f} km")
                print(f"  Mean distance: {finite_distances.mean():.1f} km")
                print(f"  Coverage: {len(finite_distances)/(len(locations_list)**2-len(locations_list))*100:.1f}%")
                
                if len(finite_times) > 0:
                    print(f"  Time range: {finite_times.min():.1f} - {finite_times.max():.1f} minutes")
                    print(f"  Mean time: {finite_times.mean():.1f} minutes ({finite_times.mean()/60:.1f} hours)")
            
            # Save the improved matrix
            print(f"\\n💾 SAVING SEGMENT-BASED MATRIX")
            
            # Replace infinities with reasonable fallback
            if len(finite_distances) > 0:
                fallback_distance = np.percentile(finite_distances, 95)
                fallback_time = np.percentile(finite_times, 95) if len(finite_times) > 0 else (fallback_distance / 60 * 60)
            else:
                fallback_distance = 1000  # 1000 km
                fallback_time = 1000  # 1000 minutes
            
            distance_matrix[np.isinf(distance_matrix)] = fallback_distance
            time_matrix[np.isinf(time_matrix)] = fallback_time
            
            # Save
            output_path = Path("data/dist_matrix_segments.npz")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez_compressed(
                output_path,
                ids=np.array(locations_list),
                dist=distance_matrix.astype(np.float32),
                time=time_matrix.astype(np.float32)
            )
            
            print(f"✅ Segment-based matrix saved: {output_path}")
            print(f"   Fallback distance: {fallback_distance:.1f} km")
            print(f"   Fallback time: {fallback_time:.1f} minutes")
            print(f"   Use this instead of the trip-based matrix for better connectivity!")
        
        print(f"\\n" + "="*50)
        print("✅ ANALYSIS COMPLETE")
        print("="*50)
        
        print(f"\\nKey findings:")
        print(f"- Segment-level data should have much better connectivity")
        print(f"- Trip aggregation may be losing intermediate stops")
        print(f"- Use segment-based matrix for realistic routing")
        print(f"- Matrix saved as: data/dist_matrix_segments.npz")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)