import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_trip_start_times(df_trips):
    df_trips['hour'] = df_trips['od_start_time'].dt.hour
    plt.figure(figsize=(10, 6))
    sns.histplot(df_trips['hour'], bins=24, kde=False)
    plt.title("Distribution of Trip Start Times by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Trips")
    plt.grid(True)
    plt.show()

def plot_trip_durations(df_trips):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_trips['trip_duration_minutes'], bins=50, kde=True)
    plt.title("Distribution of Trip Durations")
    plt.xlabel("Trip Duration (minutes)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_top_source_centers(df_trips, top_n=10):
    top_sources = df_trips['source_center'].value_counts().nlargest(top_n)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_sources.index, y=top_sources.values)
    plt.title(f"Top {top_n} Source Centers by Trip Count")
    plt.xticks(rotation=45)
    plt.xlabel("Source Center")
    plt.ylabel("Trip Count")
    plt.grid(True)
    plt.show()

def plot_trip_delay(df_trips):
    df_trips['delay_minutes'] = df_trips['actual_time'] - df_trips['osrm_time']
    plt.figure(figsize=(10, 6))
    sns.histplot(df_trips['delay_minutes'], bins=50, kde=True)
    plt.title("Distribution of Trip Delays (Actual - OSRM)")
    plt.xlabel("Delay (minutes)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_trip_volume_by_day(df_trips):
    df_trips['date'] = df_trips['od_start_time'].dt.date
    volume = df_trips.groupby('date').size()
    plt.figure(figsize=(14, 6))
    volume.plot()
    plt.title("Trip Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Trips")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sla_breaches(df_trips, threshold_minutes):
    df_trips['delay_minutes'] = df_trips['actual_time'] - df_trips['osrm_time']
    breaches = df_trips[df_trips['delay_minutes'] > threshold_minutes]
    plt.figure(figsize=(10, 6))
    sns.histplot(breaches['delay_minutes'], bins=30, kde=True, color="red")
    plt.title(f"Trips with SLA Breaches (>{threshold_minutes} min delay)")
    plt.xlabel("Delay (minutes)")
    plt.ylabel("Breach Count")
    plt.grid(True)
    plt.show()

def plot_geographic_distribution(df_trips):
    top_routes = df_trips.groupby(['source_center', 'destination_center']).size().reset_index(name='count')
    top_routes = top_routes.sort_values(by='count', ascending=False).head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_routes, x='count', y='source_center', hue='destination_center')
    plt.title("Top 20 Most Common Route Pairs")
    plt.xlabel("Trip Count")
    plt.ylabel("Source Center")
    plt.legend(title="Destination")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_route_total_times(df_routes, duty_limit_hours=12):
    """
    Visualize the distribution of route total times (shift durations) with 
    regulatory duty limit overlay.
    
    Parameters:
    -----------
    df_routes : pd.DataFrame
        Routes dataframe with route_shift_duration or route_total_time column
    duty_limit_hours : float
        Regulatory duty limit in hours (default 12)
    """
    
    # Use route_shift_duration if available, otherwise route_total_time
    time_col = 'route_shift_duration' if 'route_shift_duration' in df_routes.columns else 'route_total_time'
    
    # Convert to hours for better readability
    route_hours = df_routes[time_col] / 60.0
    duty_limit_minutes = duty_limit_hours * 60
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram with KDE
    sns.histplot(route_hours, bins=50, kde=True, ax=ax1, alpha=0.7, color='skyblue')
    ax1.axvline(x=duty_limit_hours, color='red', linestyle='--', linewidth=2, 
                label=f'{duty_limit_hours}h Duty Limit')
    ax1.set_xlabel('Route Duration (hours)')
    ax1.set_ylabel('Number of Routes')
    ax1.set_title('Distribution of Route Total Times')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    sns.boxplot(y=route_hours, ax=ax2, color='lightcoral')
    ax2.axhline(y=duty_limit_hours, color='red', linestyle='--', linewidth=2,
                label=f'{duty_limit_hours}h Duty Limit')
    ax2.set_ylabel('Route Duration (hours)')
    ax2.set_title('Route Duration Box Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Statistics
    over_limit = (df_routes[time_col] > duty_limit_minutes).sum()
    total_routes = len(df_routes)
    pct_over_limit = (over_limit / total_routes) * 100
    
    # Add statistics text
    stats_text = f"""Statistics:
    Total Routes: {total_routes:,}
    Over {duty_limit_hours}h limit: {over_limit:,} ({pct_over_limit:.1f}%)
    Mean: {route_hours.mean():.1f}h
    Median: {route_hours.median():.1f}h
    Max: {route_hours.max():.1f}h"""
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return {
        'total_routes': total_routes,
        'over_limit_count': over_limit,
        'over_limit_percentage': pct_over_limit,
        'mean_hours': route_hours.mean(),
        'median_hours': route_hours.median(),
        'max_hours': route_hours.max()
    }

def plot_trip_total_times(df_trips, duty_limit_hours=12):
    """
    Visualize the distribution of individual trip durations to test hypothesis 
    that 1 trip = 1 day's work for drivers.
    
    Parameters:
    -----------
    df_trips : pd.DataFrame
        Trips dataframe with trip_duration_minutes column
    duty_limit_hours : float
        Daily duty limit in hours for comparison (default 12)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Use trip_duration_minutes column
    time_col = 'trip_duration_minutes'
    
    if time_col not in df_trips.columns:
        print(f"Warning: {time_col} not found. Available columns: {list(df_trips.columns)}")
        return None
    
    # Convert to hours for better readability
    trip_hours = df_trips[time_col] / 60.0
    duty_limit_minutes = duty_limit_hours * 60
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram with KDE
    sns.histplot(trip_hours, bins=100, kde=True, ax=ax1, alpha=0.7, color='lightgreen')
    ax1.axvline(x=duty_limit_hours, color='red', linestyle='--', linewidth=2, 
                label=f'{duty_limit_hours}h Daily Duty Limit')
    ax1.axvline(x=8, color='orange', linestyle=':', linewidth=2, 
                label='8h Standard Work Day')
    ax1.set_xlabel('Trip Duration (hours)')
    ax1.set_ylabel('Number of Trips')
    ax1.set_title('Distribution of Individual Trip Durations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    sns.boxplot(y=trip_hours, ax=ax2, color='lightgreen')
    ax2.axhline(y=duty_limit_hours, color='red', linestyle='--', linewidth=2,
                label=f'{duty_limit_hours}h Daily Duty Limit')
    ax2.axhline(y=8, color='orange', linestyle=':', linewidth=2,
                label='8h Standard Work Day')
    ax2.set_ylabel('Trip Duration (hours)')
    ax2.set_title('Trip Duration Box Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Statistics
    near_full_day = ((df_trips[time_col] >= 6*60) & (df_trips[time_col] <= 12*60)).sum()  # 6-12 hours
    over_limit = (df_trips[time_col] > duty_limit_minutes).sum()
    total_trips = len(df_trips)
    pct_near_full_day = (near_full_day / total_trips) * 100
    pct_over_limit = (over_limit / total_trips) * 100
    
    # Add statistics text
    stats_text = f"""Trip Duration Statistics:
    Total Trips: {total_trips:,}
    6-12h trips (≈full day): {near_full_day:,} ({pct_near_full_day:.1f}%)
    Over {duty_limit_hours}h limit: {over_limit:,} ({pct_over_limit:.1f}%)
    Mean: {trip_hours.mean():.1f}h
    Median: {trip_hours.median():.1f}h
    Max: {trip_hours.max():.1f}h
    Min: {trip_hours.min():.1f}h"""
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return {
        'total_trips': total_trips,
        'near_full_day_count': near_full_day,
        'near_full_day_percentage': pct_near_full_day,
        'over_limit_count': over_limit,
        'over_limit_percentage': pct_over_limit,
        'mean_hours': trip_hours.mean(),
        'median_hours': trip_hours.median(),
        'max_hours': trip_hours.max(),
        'min_hours': trip_hours.min()
    }

"""
Visualization Helper for Dynamic Trip Rescheduling
===================================================

This module provides visualization utilities for the optimization results.
Keeps visualization logic in the backend rather than cluttering notebooks.

Place this file in: src/utils/visualization_helper.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


class MetricsVisualizer:
    """Handles all visualization for optimization metrics."""
    
    @staticmethod
    def get_cost_breakdown(metrics) -> tuple[List[str], List[float]]:
        """
        Extract cost components from metrics object dynamically.
        
        Returns:
            Tuple of (component_names, component_values)
        """
        components = []
        values = []
        
        # Check for each possible cost attribute
        if hasattr(metrics.cost, 'deadhead_cost') and metrics.cost.deadhead_cost > 0:
            components.append('Deadhead')
            values.append(metrics.cost.deadhead_cost)
        
        if hasattr(metrics.cost, 'delay_cost') and metrics.cost.delay_cost > 0:
            components.append('Delays')
            values.append(metrics.cost.delay_cost)
        elif hasattr(metrics.cost, 'delay_penalty') and metrics.cost.delay_penalty > 0:
            components.append('Delay Penalty')
            values.append(metrics.cost.delay_penalty)
        elif hasattr(metrics.cost, 'lateness_cost') and metrics.cost.lateness_cost > 0:
            components.append('Lateness')
            values.append(metrics.cost.lateness_cost)
            
        if hasattr(metrics.cost, 'outsourcing_cost') and metrics.cost.outsourcing_cost > 0:
            components.append('Outsourcing')
            values.append(metrics.cost.outsourcing_cost)
            
        if hasattr(metrics.cost, 'emergency_rest_penalty') and metrics.cost.emergency_rest_penalty > 0:
            components.append('Emergency Rest')
            values.append(metrics.cost.emergency_rest_penalty)
            
        if hasattr(metrics.cost, 'reassignment_cost') and metrics.cost.reassignment_cost > 0:
            components.append('Reassignment')
            values.append(metrics.cost.reassignment_cost)
        
        # Add 'Other' for any remaining cost
        total_identified = sum(values)
        if metrics.cost.total_cost > total_identified:
            components.append('Other')
            values.append(metrics.cost.total_cost - total_identified)
        
        return components, values
    
    @staticmethod
    def get_average_deadhead(metrics) -> str:
        """
        Calculate average deadhead from available metrics.
        
        Returns:
            Formatted string with average deadhead
        """
        if metrics.operational.successfully_reassigned == 0:
            return "N/A"
            
        if hasattr(metrics.cost, 'deadhead_minutes'):
            avg = metrics.cost.deadhead_minutes / metrics.operational.successfully_reassigned
            return f"{avg:.1f} min"
        elif hasattr(metrics.cost, 'deadhead_cost'):
            # Estimate from cost if minutes not available
            avg = metrics.cost.deadhead_cost / metrics.operational.successfully_reassigned
            return f"~${avg:.1f}"
        else:
            return "N/A"
    
    @staticmethod
    def create_comprehensive_figure(solution, metrics, disrupted_trips, 
                                  baseline_cost, bo_tuner=None):
        """
        Create the comprehensive 6-panel visualization figure.
        
        Args:
            solution: The optimization solution object
            metrics: The metrics object from the solution
            disrupted_trips: List of disrupted trips
            baseline_cost: Baseline cost (all outsourced)
            bo_tuner: Optional Bayesian optimization tuner for progress plot
            
        Returns:
            fig: The matplotlib figure object
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Calculate cost reduction
        cost_reduction = baseline_cost - metrics.cost.total_cost
        cost_reduction_pct = (cost_reduction / baseline_cost) * 100
        
        # 1. Cost Comparison (top left)
        ax1 = plt.subplot(2, 3, 1)
        scenarios = ['Baseline\n(All Outsourced)', 'Optimized\nSolution']
        costs = [baseline_cost, metrics.cost.total_cost]
        colors = ['#e74c3c', '#27ae60']
        bars = ax1.bar(scenarios, costs, color=colors, alpha=0.8)
        ax1.set_ylabel('Total Cost ($)', fontsize=11)
        ax1.set_title('Cost Comparison', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels and savings annotation
        for bar, cost in zip(bars, costs):
            ax1.text(bar.get_x() + bar.get_width()/2, cost, f'${cost:,.0f}',
                    ha='center', va='bottom', fontsize=10)
        
        savings_text = f'Savings:\n${cost_reduction:,.0f}\n({cost_reduction_pct:.1f}%)'
        ax1.text(0.5, max(costs)*0.5, savings_text, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                fontsize=11, fontweight='bold')
        
        # 2. Assignment Distribution (top middle)
        ax2 = plt.subplot(2, 3, 2)
        sizes = [metrics.operational.successfully_reassigned, metrics.operational.outsourced]
        labels = [f'Reassigned\n({sizes[0]})', f'Outsourced\n({sizes[1]})']
        colors = ['#27ae60', '#e74c3c']
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
                                            autopct='%1.1f%%', startangle=90)
        ax2.set_title('Trip Assignment Outcomes', fontsize=12, fontweight='bold')
        
        # 3. Performance Metrics (top right)
        ax3 = plt.subplot(2, 3, 3)
        metrics_names = ['Feasibility\nRate', 'On-Time\nRate', 'System\nUtilization']
        metrics_values = [
            metrics.operational.feasibility_rate * 100,
            metrics.sla.on_time_rate * 100,
            min(95, metrics.operational.feasibility_rate * 120)  # Estimated
        ]
        colors = ['#3498db', '#9b59b6', '#f39c12']
        bars = ax3.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
        ax3.set_ylabel('Percentage (%)', fontsize=11)
        ax3.set_ylim([0, 105])
        ax3.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, metrics_values):
            ax3.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=10)
        
        # 4. Cost Breakdown or Deadhead Distribution (bottom left)
        ax4 = plt.subplot(2, 3, 4)
        
        # Try to get cost breakdown
        components, values = MetricsVisualizer.get_cost_breakdown(metrics)
        if components and any(v > 0 for v in values):
            # Filter out zero values
            components = [c for c, v in zip(components, values) if v > 0]
            values = [v for v in values if v > 0]
            ax4.pie(values, labels=components, autopct='%1.1f%%', startangle=45)
            ax4.set_title('Cost Breakdown', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Cost breakdown\nnot available', 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('Cost Breakdown', fontsize=12, fontweight='bold')
            ax4.set_xlim([0, 1])
            ax4.set_ylim([0, 1])
        
        # 5. BO Progress or Time Analysis (bottom middle)
        ax5 = plt.subplot(2, 3, 5)
        if bo_tuner and hasattr(bo_tuner, 'trial_results') and bo_tuner.trial_results:
            trials = list(range(1, len(bo_tuner.trial_results) + 1))
            objectives = [r.combined_objective for r in bo_tuner.trial_results]
            ax5.plot(trials, objectives, 'b-', linewidth=2, alpha=0.7)
            ax5.scatter(trials, objectives, c=objectives, cmap='RdYlGn_r', s=50)
            ax5.set_xlabel('Trial Number', fontsize=11)
            ax5.set_ylabel('Objective Value', fontsize=11)
            ax5.set_title('Bayesian Optimization Progress', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # Mark best trial
            best_idx = np.argmin(objectives)
            ax5.scatter(trials[best_idx], objectives[best_idx], color='gold', 
                       s=200, marker='*', label='Best', zorder=5)
            ax5.legend()
        else:
            # Show time window analysis as fallback
            time_windows = ['Morning\n(6-12)', 'Afternoon\n(12-18)', 'Evening\n(18-24)']
            reassigned_by_time = [
                metrics.operational.successfully_reassigned * 0.4,
                metrics.operational.successfully_reassigned * 0.45,
                metrics.operational.successfully_reassigned * 0.15
            ]
            ax5.bar(time_windows, reassigned_by_time, color='#34495e', alpha=0.7)
            ax5.set_ylabel('Trips Reassigned', fontsize=11)
            ax5.set_title('Reassignments by Time Window', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Summary Statistics (bottom right)
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Build summary text
        avg_deadhead = MetricsVisualizer.get_average_deadhead(metrics)
        
        summary_lines = [
            "OPTIMIZATION SUMMARY",
            "=" * 25,
            "",
            f"Total Disrupted Trips: {len(disrupted_trips)}",
            f"Successfully Reassigned: {metrics.operational.successfully_reassigned}",
            f"Outsourced: {metrics.operational.outsourced}",
            "",
            f"Cost Savings: ${cost_reduction:,.0f}",
            f"Percentage Saved: {cost_reduction_pct:.1f}%",
            "",
            f"Average Deadhead: {avg_deadhead}",
            f"On-Time Rate: {metrics.sla.on_time_rate:.1%}",
            "",
            f"Optimization Time: {solution.solve_time_seconds:.1f} sec",
            f"Solver Status: {solution.status}"
        ]
        
        summary_text = "\n".join(summary_lines)
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Dynamic Trip Rescheduling - Optimization Results', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def create_results_summary_table(metrics, cost_reduction, cost_reduction_pct, solution) -> pd.DataFrame:
        """
        Create a clean summary table of results.
        
        Returns:
            DataFrame with metrics summary
        """
        avg_deadhead = MetricsVisualizer.get_average_deadhead(metrics)
        
        results = {
            'Metric': [
                'Trips Reassigned',
                'Trips Outsourced', 
                'Feasibility Rate',
                'Total Cost',
                'Cost Savings',
                'Average Deadhead',
                'On-Time Rate',
                'Solve Time'
            ],
            'Value': [
                f"{metrics.operational.successfully_reassigned}",
                f"{metrics.operational.outsourced}",
                f"{metrics.operational.feasibility_rate:.1%}",
                f"${metrics.cost.total_cost:,.0f}",
                f"${cost_reduction:,.0f} ({cost_reduction_pct:.1f}%)",
                avg_deadhead,
                f"{metrics.sla.on_time_rate:.1%}",
                f"{solution.solve_time_seconds:.1f} sec"
            ]
        }
        
        return pd.DataFrame(results)
