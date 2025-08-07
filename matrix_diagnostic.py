#!/usr/bin/env python3
"""
Distance Matrix Diagnostics
===========================

Analyze the generated distance matrix to understand:
- How many pairs are real vs fallback
- Distribution of actual distances vs 2000km fallback
- Which locations are well-connected vs isolated
- Hub connectivity patterns

This helps understand why candidate generation is failing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def analyze_distance_matrix():
    """Comprehensive analysis of the distance matrix."""
    
    print("🔍 DISTANCE MATRIX DIAGNOSTICS")
    print("=" * 50)
    
    # Find and load the matrix
    possible_paths = [
        Path("data/dist_matrix.npz"),
        Path("../data/dist_matrix.npz"),
        Path.cwd().parent / "data" / "dist_matrix.npz"
    ]
    
    matrix_path = None
    for path in possible_paths:
        if path.exists():
            matrix_path = path
            break
    
    if not matrix_path:
        raise FileNotFoundError(f"Distance matrix not found. Checked: {[str(p) for p in possible_paths]}")
    
    print(f"✅ Loading matrix: {matrix_path}")
    
    # Load matrix data
    dist_data = np.load(str(matrix_path), allow_pickle=True)
    time_matrix = dist_data['time']  # Travel times in minutes
    distance_matrix = dist_data['dist']  # Distances in km
    location_ids = dist_data['ids']
    
    n_locations = len(location_ids)
    print(f"✅ Matrix loaded: {n_locations} locations")
    
    # Basic matrix analysis
    print(f"\n📊 BASIC MATRIX ANALYSIS")
    print("-" * 30)
    
    # Remove diagonal (self-distances)
    non_diagonal_mask = ~np.eye(n_locations, dtype=bool)
    non_diagonal_times = time_matrix[non_diagonal_mask]
    non_diagonal_distances = distance_matrix[non_diagonal_mask]
    
    total_pairs = len(non_diagonal_times)
    print(f"Total location pairs: {total_pairs:,}")
    
    # Analyze fallback usage (2000km = 2000 minutes at 60km/h)
    fallback_threshold = 1990  # Allow for small rounding differences
    fallback_mask = non_diagonal_times >= fallback_threshold
    fallback_count = fallback_mask.sum()
    real_count = total_pairs - fallback_count
    
    print(f"\n🎯 FALLBACK ANALYSIS")
    print("-" * 30)
    print(f"Real connections: {real_count:,} ({real_count/total_pairs*100:.1f}%)")
    print(f"Fallback pairs: {fallback_count:,} ({fallback_count/total_pairs*100:.1f}%)")
    
    if fallback_count > total_pairs * 0.8:
        print("❌ CRITICAL: >80% of pairs use fallback - matrix mostly artificial!")
    elif fallback_count > total_pairs * 0.5:
        print("⚠️ WARNING: >50% of pairs use fallback - network very sparse")
    else:
        print("✅ Reasonable fallback usage")
    
    # Analyze real connections
    real_times = non_diagonal_times[~fallback_mask]
    real_distances = non_diagonal_distances[~fallback_mask]
    
    if len(real_times) > 0:
        print(f"\n📈 REAL CONNECTION ANALYSIS")
        print("-" * 30)
        print(f"Real distance range: {real_distances.min():.1f} - {real_distances.max():.1f} km")
        print(f"Real time range: {real_times.min():.1f} - {real_times.max():.1f} minutes")
        print(f"Real distance mean: {real_distances.mean():.1f} km")
        print(f"Real time mean: {real_times.mean():.1f} minutes ({real_times.mean()/60:.1f} hours)")
    
    # Per-location connectivity analysis
    print(f"\n🏢 LOCATION CONNECTIVITY ANALYSIS")
    print("-" * 30)
    
    location_stats = []
    
    for i, location in enumerate(location_ids):
        # Count real connections for this location
        row_connections = time_matrix[i, :] < fallback_threshold
        col_connections = time_matrix[:, i] < fallback_threshold
        
        # Remove self-connection
        row_connections[i] = False
        col_connections[i] = False
        
        total_real_connections = row_connections.sum() + col_connections.sum()
        # Avoid double counting - connections should be symmetric
        unique_connections = row_connections.sum()  # Just count outbound
        
        location_stats.append({
            'location': str(location),
            'real_connections': unique_connections,
            'connection_percentage': unique_connections / (n_locations - 1) * 100
        })
    
    # Convert to DataFrame and analyze
    connectivity_df = pd.DataFrame(location_stats)
    connectivity_df = connectivity_df.sort_values('real_connections', ascending=False)
    
    print(f"Best connected locations:")
    for i, row in connectivity_df.head(10).iterrows():
        print(f"  {row['location'][:15]:15}: {row['real_connections']:3d} connections ({row['connection_percentage']:4.1f}%)")
    
    print(f"\nWorst connected locations:")
    for i, row in connectivity_df.tail(5).iterrows():
        print(f"  {row['location'][:15]:15}: {row['real_connections']:3d} connections ({row['connection_percentage']:4.1f}%)")
    
    # Connectivity distribution
    conn_counts = connectivity_df['real_connections']
    print(f"\nConnectivity statistics:")
    print(f"  Mean connections per location: {conn_counts.mean():.1f}")
    print(f"  Median connections: {conn_counts.median():.1f}")
    print(f"  Max connections: {conn_counts.max()}")
    print(f"  Locations with 0 connections: {(conn_counts == 0).sum()}")
    print(f"  Locations with <5 connections: {(conn_counts < 5).sum()}")
    print(f"  Locations with >50 connections: {(conn_counts > 50).sum()}")
    
    # Test specific locations from validation
    print(f"\n🔍 VALIDATION LOCATION ANALYSIS")
    print("-" * 30)
    
    test_locations = ['IND000000AAL', 'IND000000AAQ', 'IND000000AAZ']
    location_to_index = {str(loc): i for i, loc in enumerate(location_ids)}
    
    for loc in test_locations:
        if loc in location_to_index:
            idx = location_to_index[loc]
            connections = (time_matrix[idx, :] < fallback_threshold).sum() - 1  # -1 for self
            print(f"  {loc}: {connections} real connections")
            
            # Show distances to other test locations
            for other_loc in test_locations:
                if other_loc != loc and other_loc in location_to_index:
                    other_idx = location_to_index[other_loc]
                    distance = distance_matrix[idx, other_idx]
                    time_min = time_matrix[idx, other_idx]
                    is_fallback = time_min >= fallback_threshold
                    status = "FALLBACK" if is_fallback else "REAL"
                    print(f"    -> {other_loc}: {distance:.1f}km, {time_min:.1f}min ({status})")
        else:
            print(f"  {loc}: NOT FOUND in matrix")
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Time distribution (real vs fallback)
    bins = np.logspace(1, 4, 50)  # Log scale from 10 to 10000 minutes
    ax1.hist(real_times, bins=bins, alpha=0.7, label=f'Real ({len(real_times):,})', color='blue')
    ax1.hist(non_diagonal_times[fallback_mask], bins=bins, alpha=0.7, 
             label=f'Fallback ({fallback_count:,})', color='red')
    ax1.set_xscale('log')
    ax1.set_xlabel('Travel Time (minutes)')
    ax1.set_ylabel('Number of Location Pairs')
    ax1.set_title('Travel Time Distribution: Real vs Fallback')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Connectivity distribution
    ax2.hist(conn_counts, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Real Connections')
    ax2.set_ylabel('Number of Locations')
    ax2.set_title('Location Connectivity Distribution')
    ax2.axvline(conn_counts.mean(), color='red', linestyle='--', 
                label=f'Mean: {conn_counts.mean():.1f}')
    ax2.legend()
    
    # Plot 3: Connectivity vs rank
    ax3.plot(range(1, len(connectivity_df)+1), connectivity_df['real_connections'])
    ax3.set_xlabel('Location Rank (by connectivity)')
    ax3.set_ylabel('Number of Real Connections')
    ax3.set_title('Location Connectivity by Rank')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Real distance distribution
    if len(real_distances) > 0:
        ax4.hist(real_distances, bins=50, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Distance (km)')
        ax4.set_ylabel('Number of Real Connections')
        ax4.set_title('Real Distance Distribution')
        ax4.axvline(real_distances.mean(), color='red', linestyle='--',
                   label=f'Mean: {real_distances.mean():.0f} km')
        ax4.legend()
    
    plt.tight_layout()
    
    # Summary and recommendations
    print(f"\n" + "="*50)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*50)
    
    print(f"\n🔍 Key Findings:")
    print(f"- {fallback_count/total_pairs*100:.1f}% of location pairs use 2000km fallback")
    print(f"- Average location has {conn_counts.mean():.1f} real connections")
    print(f"- {(conn_counts < 5).sum()} locations have <5 real connections")
    print(f"- Validation locations have fallback distances to each other")
    
    print(f"\n💡 Recommendations:")
    
    if fallback_count > total_pairs * 0.7:
        print("❌ MATRIX UNUSABLE: Too many fallback pairs")
        print("   Solutions:")
        print("   1. Use only well-connected locations (top 100-200)")
        print("   2. Implement hub-preference in candidate generation")
        print("   3. Add distance/time limits to reject unrealistic candidates")
        print("   4. Consider regional clustering approach")
    elif fallback_count > total_pairs * 0.3:
        print("⚠️ MATRIX NEEDS FILTERING:")
        print("   1. Reject candidates with >X hour travel times")
        print("   2. Prefer hub locations for assignments")
        print("   3. Add connectivity requirements")
    else:
        print("✅ Matrix quality acceptable with minor filtering")
    
    print(f"\nFor candidate generation:")
    print(f"- Add max travel time limit (e.g., 8-12 hours)")
    print(f"- Prefer locations with >10 real connections")
    print(f"- Consider hub-based assignment strategy")
    
    return {
        'total_pairs': total_pairs,
        'real_pairs': real_count,
        'fallback_pairs': fallback_count,
        'connectivity_stats': connectivity_df,
        'real_times': real_times,
        'fallback_percentage': fallback_count/total_pairs*100
    }

if __name__ == "__main__":
    try:
        results = analyze_distance_matrix()
        plt.show()
        
        print(f"\n✅ Diagnostics complete!")
        print(f"Matrix quality: {100-results['fallback_percentage']:.1f}% real connections")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()