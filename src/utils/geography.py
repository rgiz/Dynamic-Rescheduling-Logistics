#!/usr/bin/env python3
"""
Hub-Based Distance Matrix Generation
===================================

Efficient approach that:
1. Identifies hubs by trip volume
2. Creates tiered distance matrix (hub-to-hub + hub-to-spoke)
3. Generates complete matrix in minutes instead of hours
4. Provides geographic analysis and visualizations

Usage (from repo root):
    python3 scripts/generate_distance_matrix.py

Generates data/dist_matrix.npz using hub-based routing strategy.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import networkx as nx

class HubBasedMatrixGenerator:
    """Generates distance matrix using hub-based routing strategy."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.trips_df = None
        self.hub_df = None
        self.graph = None
        self.distance_matrix = None
        self.location_to_index = None
        self.all_locations = None
        
    def load_trip_data(self):
        """Load and validate trip data."""
        print("📊 LOADING TRIP DATA")
        print("=" * 30)
        
        trips_file = self.project_root / "data" / "processed" / "trips.csv"
        if not trips_file.exists():
            raise FileNotFoundError(f"Trips file not found: {trips_file}")
        
        print(f"✅ Loading: {trips_file}")
        self.trips_df = pd.read_csv(trips_file)
        
        # Filter to valid trips
        valid_trips = self.trips_df[
            (self.trips_df['source_center'] != self.trips_df['destination_center']) &
            self.trips_df['segment_osrm_distance'].notna()
        ]
        
        self.trips_df = valid_trips
        print(f"✅ Found {len(self.trips_df):,} valid trips")
        
        # Get all unique locations
        self.all_locations = sorted(
            set(self.trips_df['source_center']).union(set(self.trips_df['destination_center']))
        )
        self.location_to_index = {loc: i for i, loc in enumerate(self.all_locations)}
        
        print(f"✅ Found {len(self.all_locations)} unique locations")
        return self.trips_df
    
    def analyze_hubs(self):
        """Analyze hub locations by trip volume."""
        print("\n🏢 ANALYZING HUB LOCATIONS")
        print("=" * 30)
        
        # Count trips by location
        outbound = self.trips_df['source_center'].value_counts()
        inbound = self.trips_df['destination_center'].value_counts()
        
        location_stats = []
        for location in self.all_locations:
            out_trips = outbound.get(location, 0)
            in_trips = inbound.get(location, 0)
            total_trips = out_trips + in_trips
            
            location_stats.append({
                'location': location,
                'outbound_trips': out_trips,
                'inbound_trips': in_trips,
                'total_trips': total_trips,
                'trip_balance': out_trips - in_trips
            })
        
        # Create hub DataFrame
        self.hub_df = pd.DataFrame(location_stats)
        self.hub_df = self.hub_df.sort_values('total_trips', ascending=False)
        
        # Define hub tiers
        total_trips_all = len(self.trips_df)
        self.hub_df['trip_percentage'] = (self.hub_df['total_trips'] / total_trips_all * 100)
        
        # Hub classification
        n_locations = len(self.hub_df)
        top_1_percent = max(1, int(n_locations * 0.01))
        top_5_percent = max(5, int(n_locations * 0.05))
        top_10_percent = max(10, int(n_locations * 0.10))
        
        self.hub_df['hub_tier'] = 'Local'
        self.hub_df.iloc[:top_1_percent, self.hub_df.columns.get_loc('hub_tier')] = 'Major Hub'
        self.hub_df.iloc[top_1_percent:top_5_percent, self.hub_df.columns.get_loc('hub_tier')] = 'Regional Hub'
        self.hub_df.iloc[top_5_percent:top_10_percent, self.hub_df.columns.get_loc('hub_tier')] = 'Minor Hub'
        
        # Get hub lists
        self.major_hubs = list(self.hub_df[self.hub_df['hub_tier'] == 'Major Hub']['location'])
        self.regional_hubs = list(self.hub_df[self.hub_df['hub_tier'] == 'Regional Hub']['location'])
        self.minor_hubs = list(self.hub_df[self.hub_df['hub_tier'] == 'Minor Hub']['location'])
        self.all_hubs = self.major_hubs + self.regional_hubs + self.minor_hubs
        
        print(f"✅ Hub analysis complete:")
        print(f"   Major hubs: {len(self.major_hubs)}")
        print(f"   Regional hubs: {len(self.regional_hubs)}")
        print(f"   Minor hubs: {len(self.minor_hubs)}")
        print(f"   Total hub coverage: {self.hub_df.head(len(self.all_hubs))['trip_percentage'].sum():.1f}% of trips")
        
        return self.hub_df
    
    def build_network_graph(self):
        """Build network graph with distance weights."""
        print("\n🗺️ BUILDING NETWORK GRAPH")
        print("=" * 30)
        
        self.graph = nx.Graph()
        
        # Add edges with median distances
        trip_pairs = self.trips_df.groupby(['source_center', 'destination_center'])['segment_osrm_distance'].median()
        
        edges_added = 0
        for (src, dst), dist in trip_pairs.items():
            self.graph.add_edge(src, dst, weight=dist)
            edges_added += 1
        
        print(f"✅ Graph built: {self.graph.number_of_nodes()} nodes, {edges_added} edges")
        return self.graph
    
    def generate_hub_based_matrix(self):
        """Generate distance matrix using hub-based routing strategy."""
        print("\n🎯 GENERATING HUB-BASED DISTANCE MATRIX")
        print("=" * 30)
        
        n = len(self.all_locations)
        self.distance_matrix = np.full((n, n), np.inf)
        
        # Fill diagonal
        np.fill_diagonal(self.distance_matrix, 0)
        
        print("Step 1: Adding direct connections...")
        # Add direct connections from trip data
        direct_connections = 0
        for (src, dst), dist in self.trips_df.groupby(['source_center', 'destination_center'])['segment_osrm_distance'].median().items():
            if src in self.location_to_index and dst in self.location_to_index:
                i, j = self.location_to_index[src], self.location_to_index[dst]
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist  # Make symmetric
                direct_connections += 1
        
        print(f"✅ Added {direct_connections} direct connections")
        
        print("Step 2: Hub-to-hub shortest paths...")
        # Calculate shortest paths between hubs only (much faster)
        hub_indices = [self.location_to_index[hub] for hub in self.all_hubs if hub in self.location_to_index]
        
        # Mini Floyd-Warshall for hubs only
        for k in hub_indices:
            for i in hub_indices:
                for j in hub_indices:
                    if self.distance_matrix[i, k] + self.distance_matrix[k, j] < self.distance_matrix[i, j]:
                        self.distance_matrix[i, j] = self.distance_matrix[i, k] + self.distance_matrix[k, j]
        
        print(f"✅ Computed hub-to-hub paths for {len(hub_indices)} hubs")
        
        print("Step 3: Hub-based routing for non-hubs...")
        # For each non-hub location, find best routing through hubs
        non_hub_locations = [loc for loc in self.all_locations if loc not in self.all_hubs]
        
        for location in non_hub_locations:
            loc_idx = self.location_to_index[location]
            
            # Find nearest hub with direct connection
            nearest_hubs = []
            for hub in self.all_hubs:
                if hub in self.location_to_index:
                    hub_idx = self.location_to_index[hub]
                    if not np.isinf(self.distance_matrix[loc_idx, hub_idx]):
                        nearest_hubs.append((hub, hub_idx, self.distance_matrix[loc_idx, hub_idx]))
            
            if nearest_hubs:
                # Route through nearest hub to reach other locations
                nearest_hubs.sort(key=lambda x: x[2])  # Sort by distance
                primary_hub, primary_hub_idx, hub_distance = nearest_hubs[0]
                
                for other_loc in self.all_locations:
                    if other_loc != location:
                        other_idx = self.location_to_index[other_loc]
                        
                        # Route: location -> primary_hub -> other_location
                        if not np.isinf(self.distance_matrix[primary_hub_idx, other_idx]):
                            routed_distance = hub_distance + self.distance_matrix[primary_hub_idx, other_idx]
                            
                            if routed_distance < self.distance_matrix[loc_idx, other_idx]:
                                self.distance_matrix[loc_idx, other_idx] = routed_distance
                                self.distance_matrix[other_idx, loc_idx] = routed_distance
        
        print("✅ Hub-based routing complete")
        
        print("Step 4: Filling remaining gaps...")
        # Fill remaining infinite distances with reasonable estimates
        finite_distances = self.distance_matrix[np.isfinite(self.distance_matrix) & (self.distance_matrix > 0)]
        
        if len(finite_distances) > 0:
            # Use 95th percentile as fallback for very disconnected pairs
            fallback_distance = np.percentile(finite_distances, 95)
            print(f"   Using {fallback_distance:.1f} km fallback for disconnected pairs")
        else:
            fallback_distance = 2000  # Default fallback
            print(f"   Using {fallback_distance} km default fallback")
        
        # Replace remaining infinities
        infinite_mask = np.isinf(self.distance_matrix)
        self.distance_matrix[infinite_mask] = fallback_distance
        
        infinite_count = infinite_mask.sum()
        total_pairs = n * n - n  # Exclude diagonal
        
        print(f"✅ Matrix generation complete:")
        print(f"   Filled {infinite_count:,} gaps with fallback distance")
        print(f"   Matrix coverage: {((total_pairs - infinite_count) / total_pairs * 100):.1f}% direct/routed")
        
        return self.distance_matrix
    
    def save_distance_matrix(self):
        """Save distance matrix in standard format."""
        print("\n💾 SAVING DISTANCE MATRIX")
        print("=" * 30)
        
        # Convert to time matrix (minutes at 60 km/h)
        speed_kmph = 60
        time_matrix = (self.distance_matrix / speed_kmph * 60).astype(np.float32)
        
        # Save matrix
        dist_path = self.project_root / "data" / "dist_matrix.npz"
        dist_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            dist_path,
            ids=np.array(self.all_locations),
            dist=self.distance_matrix.astype(np.float32),
            time=time_matrix
        )
        
        print(f"✅ Saved distance matrix: {dist_path}")
        
        # Statistics
        non_diagonal = time_matrix[~np.eye(len(self.all_locations), dtype=bool)]
        print(f"📊 Travel Time Statistics:")
        print(f"   Min time: {non_diagonal.min():.1f} minutes ({non_diagonal.min()/60:.1f} hours)")
        print(f"   Max time: {non_diagonal.max():.1f} minutes ({non_diagonal.max()/60:.1f} hours)")
        print(f"   Mean time: {non_diagonal.mean():.1f} minutes ({non_diagonal.mean()/60:.1f} hours)")
        print(f"   Median time: {np.median(non_diagonal):.1f} minutes ({np.median(non_diagonal)/60:.1f} hours)")
        
        return dist_path
    
    def create_analysis_report(self) -> str:
        """Create markdown analysis report."""
        total_trips = len(self.trips_df)
        
        # Hub concentration analysis
        major_hub_traffic = self.hub_df[self.hub_df['hub_tier'] == 'Major Hub']['trip_percentage'].sum()
        top_10_traffic = self.hub_df.head(10)['trip_percentage'].sum()
        
        # Matrix coverage analysis
        non_diagonal_mask = ~np.eye(len(self.all_locations), dtype=bool)
        finite_distances = self.distance_matrix[non_diagonal_mask & np.isfinite(self.distance_matrix)]
        total_pairs = non_diagonal_mask.sum()
        coverage = len(finite_distances) / total_pairs * 100
        
        markdown = f"""
# Hub-Based Distance Matrix Analysis

## Trip Volume Analysis

| Metric | Value |
|--------|-------|
| Total Locations | {len(self.all_locations):,} |
| Total Trips | {total_trips:,} |
| Major Hubs | {len(self.major_hubs)} locations |
| Regional Hubs | {len(self.regional_hubs)} locations |
| Minor Hubs | {len(self.minor_hubs)} locations |

## Traffic Concentration

| Hub Tier | Locations | Traffic Share |
|----------|-----------|---------------|
| Major Hubs | {len(self.major_hubs)} | {major_hub_traffic:.1f}% |
| Regional Hubs | {len(self.regional_hubs)} | {self.hub_df[self.hub_df['hub_tier'] == 'Regional Hub']['trip_percentage'].sum():.1f}% |
| Top 10 Locations | 10 | {top_10_traffic:.1f}% |

## Distance Matrix Quality

| Metric | Value |
|--------|-------|
| Matrix Size | {len(self.all_locations)} × {len(self.all_locations)} |
| Direct/Routed Coverage | {coverage:.1f}% |
| Hub-Based Routes | {(coverage - (self.graph.number_of_edges() * 2 / total_pairs * 100)):.1f}% |
| Average Distance | {np.mean(finite_distances):.1f} km |
| Max Distance | {np.max(finite_distances):.1f} km |

## Top 15 Hub Locations

| Rank | Location | Total Trips | % of Traffic | Hub Tier |
|------|----------|-------------|--------------|----------|
"""
        
        for i, (_, row) in enumerate(self.hub_df.head(15).iterrows(), 1):
            markdown += f"| {i} | {row['location'][:15]} | {row['total_trips']:,} | {row['trip_percentage']:.2f}% | {row['hub_tier']} |\n"
        
        return markdown
    
    def create_visualizations(self):
        """Create analysis visualizations."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Hub traffic concentration
        top_20 = self.hub_df.head(20)
        ax1.bar(range(len(top_20)), top_20['total_trips'], alpha=0.7)
        ax1.set_xlabel('Hub Rank')
        ax1.set_ylabel('Trip Count')
        ax1.set_title('Top 20 Hub Locations by Trip Volume')
        ax1.set_yscale('log')
        
        # Plot 2: Pareto analysis
        cumulative_pct = self.hub_df['trip_percentage'].cumsum()
        ax2.plot(range(1, len(cumulative_pct)+1), cumulative_pct)
        ax2.axhline(80, color='red', linestyle='--', label='80% of trips')
        ax2.set_xlabel('Number of Locations (ranked)')
        ax2.set_ylabel('Cumulative % of Trips')
        ax2.set_title('Pareto Analysis - Traffic Concentration')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Distance distribution
        non_diagonal = self.distance_matrix[~np.eye(len(self.all_locations), dtype=bool)]
        finite_distances = non_diagonal[np.isfinite(non_diagonal) & (non_diagonal > 0)]
        
        ax3.hist(finite_distances, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Distance (km)')
        ax3.set_ylabel('Number of Location Pairs')
        ax3.set_title('Distribution of Inter-Location Distances')
        ax3.axvline(np.mean(finite_distances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(finite_distances):.0f} km')
        ax3.legend()
        
        # Plot 4: Hub tier distribution
        tier_counts = self.hub_df['hub_tier'].value_counts()
        colors = ['red', 'orange', 'yellow', 'lightblue']
        ax4.pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%', colors=colors)
        ax4.set_title('Distribution of Hub Tiers')
        
        plt.tight_layout()
        return fig

def main():
    """Generate hub-based distance matrix."""
    print("🏢 HUB-BASED DISTANCE MATRIX GENERATION")
    print("=" * 60)
    print("Strategy: Hub-based routing with traffic volume analysis")
    print("=" * 60)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    trips_file = project_root / "data" / "processed" / "trips.csv"
    
    if not trips_file.exists():
        print(f"❌ Trips file not found: {trips_file}")
        print("   Make sure you've run data preprocessing first")
        return 1
    
    try:
        # Initialize generator
        generator = HubBasedMatrixGenerator(project_root)
        
        # Run analysis pipeline
        generator.load_trip_data()
        generator.analyze_hubs()
        generator.build_network_graph()
        generator.generate_hub_based_matrix()
        generator.save_distance_matrix()
        
        # Create analysis outputs
        markdown_report = generator.create_analysis_report()
        
        # Save analysis report
        report_path = project_root / "hub_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(markdown_report)
        print(f"✅ Analysis report saved: {report_path}")
        
        # Create and save visualizations
        fig = generator.create_visualizations()
        viz_path = project_root / "hub_analysis_plots.png"
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualizations saved: {viz_path}")
        
        print(f"\n" + "=" * 60)
        print("✅ HUB-BASED MATRIX GENERATION COMPLETE!")
        print("=" * 60)
        
        print(f"\nGenerated files:")
        print(f"- data/dist_matrix.npz (usable distance matrix)")
        print(f"- hub_analysis_report.md (detailed analysis)")
        print(f"- hub_analysis_plots.png (visualizations)")
        
        print(f"\nNext steps:")
        print(f"1. Test with: python3 val.py")
        print(f"2. Review hub analysis report for insights")
        print(f"3. Use hub information to optimize candidate generation")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)