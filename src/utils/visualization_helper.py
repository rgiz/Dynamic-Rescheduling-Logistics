"""
Visualization Helper for Dynamic Trip Rescheduling
==================================================

Focuses on clear operational metrics:
- Average delay per reassigned trip (including downstream impacts)
- Average extra miles per reassigned trip
- Clear comparisons between baseline and optimized solutions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class MetricsVisualizer:
    """Handles visualization for optimization metrics with focus on operational metrics."""
    
    @staticmethod
    def extract_operational_metrics(solution) -> Dict[str, float]:
        """
        Extract operational metrics from a solution.
        
        Returns:
            Dict with detailed operational metrics
        """
        metrics = {
            'total_delay_minutes': 0,
            'total_deadhead_miles': 0,
            'reassigned_count': 0,
            'outsourced_count': 0,
            'avg_delay_per_reassigned': 0,
            'avg_miles_per_reassigned': 0,
            'max_delay': 0,
            'trips_with_zero_delay': 0,
            'trips_with_delays': 0
        }
        
        if not solution or not solution.assignments:
            return metrics
        
        delays = []
        miles = []
        
        for assignment in solution.assignments:
            # Count assignment types
            if assignment.get('type') == 'outsourced' or assignment.get('driver_id') is None:
                metrics['outsourced_count'] += 1
            else:
                metrics['reassigned_count'] += 1
                
                # Track delays for reassigned trips
                delay = assignment.get('delay_minutes', 0)
                # Only include finite delays
                if delay != float('inf') and delay >= 0:
                    delays.append(delay)
                    metrics['total_delay_minutes'] += delay
                    
                    if delay > 0:
                        metrics['trips_with_delays'] += 1
                        metrics['max_delay'] = max(metrics['max_delay'], delay)
                    else:
                        metrics['trips_with_zero_delay'] += 1
                
                # Track deadhead miles for reassigned trips
                # Check if deadhead_miles is directly available first
                if 'deadhead_miles' in assignment:
                    deadhead_miles = assignment['deadhead_miles']
                    if deadhead_miles != float('inf') and deadhead_miles >= 0:
                        miles.append(deadhead_miles)
                        metrics['total_deadhead_miles'] += deadhead_miles
                else:
                    # Fallback: calculate from deadhead_minutes
                    deadhead_min = assignment.get('deadhead_minutes', 0)
                    if deadhead_min != float('inf') and deadhead_min >= 0:
                        # Convert minutes to miles (assuming 30 mph average speed)
                        deadhead_miles = (deadhead_min / 60) * 30
                        miles.append(deadhead_miles)
                        metrics['total_deadhead_miles'] += deadhead_miles
        
        # Calculate averages for reassigned trips only
        if metrics['reassigned_count'] > 0:
            if delays:  # Only calculate if we have valid delay data
                metrics['avg_delay_per_reassigned'] = sum(delays) / len(delays)
            if miles:   # Only calculate if we have valid mileage data
                metrics['avg_miles_per_reassigned'] = sum(miles) / len(miles)
        
        return metrics
    
    @staticmethod
    def create_comparison_dashboard(baseline_solution, bo_solution, disrupted_trips, 
                                   bo_tuner=None) -> plt.Figure:
        """
        Create a comprehensive dashboard comparing baseline and BO-optimized solutions.
        Focuses on average delay minutes and average extra miles per reassigned trip.
        """
        # Extract metrics
        baseline_metrics = MetricsVisualizer.extract_operational_metrics(baseline_solution)
        bo_metrics = MetricsVisualizer.extract_operational_metrics(bo_solution)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Reassignment Success Pie Chart (top left)
        ax1 = plt.subplot(2, 3, 1)
        
        # Use BO solution for pie chart (or baseline if BO not available)
        metrics_for_pie = bo_metrics if bo_solution else baseline_metrics
        sizes = [metrics_for_pie['reassigned_count'], metrics_for_pie['outsourced_count']]
        
        if sum(sizes) > 0:
            labels = [f"Reassigned\n({sizes[0]} trips)", f"Outsourced\n({sizes[1]} trips)"]
            colors = ['#2ecc71', '#e74c3c']
            explode = (0.05, 0)
            
            wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                               autopct='%1.0f%%', shadow=True, startangle=90)
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_weight('bold')
                autotext.set_fontsize(12)
            
            ax1.set_title(f'Trip Reassignment Results\n({len(disrupted_trips)} Disrupted Trips)', 
                        fontweight='bold', fontsize=12)
        else:
            ax1.text(0.5, 0.5, 'No Assignments', ha='center', va='center', fontsize=14)
            ax1.set_title('Trip Reassignment Results', fontweight='bold', fontsize=12)
        
        # 2. Average Delay Comparison (top center)
        ax2 = plt.subplot(2, 3, 2)
        
        categories = ['Baseline\n(Default)', 'Optimized\n(BO)']
        avg_delays = [
            baseline_metrics['avg_delay_per_reassigned'],
            bo_metrics['avg_delay_per_reassigned']
        ]
        
        # Create bars with different colors
        bars = ax2.bar(categories, avg_delays, color=['#3498db', '#e67e22'], width=0.5, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_delays):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f} min', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax2.set_ylabel('Average Delay per Reassigned Trip (minutes)', fontsize=11)
        ax2.set_title('Service Impact: Average Delay\n(Including Downstream Effects)', fontweight='bold', fontsize=12)
        ax2.set_ylim(0, max(avg_delays) * 1.3 if max(avg_delays) > 0 else 10)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add context: how many trips had delays
        ax2.text(0.02, 0.98, f'Baseline: {baseline_metrics["trips_with_delays"]}/{baseline_metrics["reassigned_count"]} trips delayed',
                transform=ax2.transAxes, fontsize=9, va='top')
        ax2.text(0.02, 0.92, f'Optimized: {bo_metrics["trips_with_delays"]}/{bo_metrics["reassigned_count"]} trips delayed',
                transform=ax2.transAxes, fontsize=9, va='top')
        
        # 3. Average Extra Miles Comparison (top right)
        ax3 = plt.subplot(2, 3, 3)
        
        categories = ['Baseline\n(Default)', 'Optimized\n(BO)']
        avg_miles = [
            baseline_metrics['avg_miles_per_reassigned'],
            bo_metrics['avg_miles_per_reassigned']
        ]
        
        bars = ax3.bar(categories, avg_miles, color=['#9b59b6', '#1abc9c'], width=0.5, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_miles):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f} mi', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax3.set_ylabel('Average Extra Miles per Reassigned Trip', fontsize=11)
        ax3.set_title('Operational Impact: Deadhead Travel\n(Repositioning Distance)', fontweight='bold', fontsize=12)
        ax3.set_ylim(0, max(avg_miles) * 1.3 if max(avg_miles) > 0 else 10)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. BO Optimization Progress (bottom left)
        ax4 = plt.subplot(2, 3, 4)
        
        if bo_tuner and hasattr(bo_tuner, 'trial_results') and bo_tuner.trial_results:
            trials = list(range(1, len(bo_tuner.trial_results) + 1))
            
            # Extract objectives that aren't infinite
            objectives = []
            for r in bo_tuner.trial_results:
                if r.combined_objective != float('inf') and r.feasibility_rate > 0:
                    objectives.append(r.combined_objective)
                else:
                    objectives.append(None)
            
            # Plot valid points
            valid_points = [(i+1, obj) for i, obj in enumerate(objectives) if obj is not None]
            
            if valid_points:
                valid_trials, valid_objs = zip(*valid_points)
                
                # Plot line and points
                ax4.plot(valid_trials, valid_objs, 'b-', alpha=0.4, linewidth=1.5)
                ax4.scatter(valid_trials, valid_objs, c='blue', s=40, alpha=0.6, edgecolors='darkblue')
                
                # Mark best trial with a star
                best_idx = np.argmin(valid_objs)
                ax4.scatter(valid_trials[best_idx], valid_objs[best_idx], 
                          color='gold', s=300, marker='*', edgecolors='black', linewidth=2,
                          label=f'Best (Trial {valid_trials[best_idx]})', zorder=5)
                
                # Add rolling average line for trend
                if len(valid_objs) > 3:
                    window = min(5, len(valid_objs) // 3)
                    rolling_avg = pd.Series(valid_objs).rolling(window=window, min_periods=1).mean()
                    ax4.plot(valid_trials, rolling_avg, 'r--', alpha=0.5, linewidth=2, label='Trend')
                
                ax4.legend(loc='best', framealpha=0.9)
            
            ax4.set_xlabel('Trial Number', fontsize=11)
            ax4.set_ylabel('Combined Objective Score', fontsize=11)
            ax4.set_title('Bayesian Optimization Convergence\n(Lower is Better)', fontweight='bold', fontsize=12)
            ax4.grid(True, alpha=0.3)
            
            # Add trial success rate
            feasible = sum(1 for r in bo_tuner.trial_results if r.feasibility_rate > 0)
            ax4.text(0.02, 0.98, f'Feasible: {feasible}/{len(bo_tuner.trial_results)} trials',
                    transform=ax4.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        else:
            ax4.text(0.5, 0.5, 'No BO Data Available', ha='center', va='center', fontsize=14)
            ax4.set_title('Bayesian Optimization Progress', fontweight='bold', fontsize=12)
        
        # 5. Delay Distribution Comparison (bottom center)
        ax5 = plt.subplot(2, 3, 5)
        
        # Create box plots for delay distributions
        baseline_delays = []
        bo_delays = []
        
        if baseline_solution and baseline_solution.assignments:
            for a in baseline_solution.assignments:
                if a.get('type') != 'outsourced':
                    baseline_delays.append(a.get('delay_minutes', 0))
        
        if bo_solution and bo_solution.assignments:
            for a in bo_solution.assignments:
                if a.get('type') != 'outsourced':
                    bo_delays.append(a.get('delay_minutes', 0))
        
        # Create box plot
        data_to_plot = []
        labels = []
        
        if baseline_delays:
            data_to_plot.append(baseline_delays)
            labels.append('Baseline')
        if bo_delays:
            data_to_plot.append(bo_delays)
            labels.append('Optimized')
        
        if data_to_plot:
            bp = ax5.boxplot(data_to_plot, labels=labels, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2),
                            flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
            
            ax5.set_ylabel('Delay Minutes per Trip', fontsize=11)
            ax5.set_title('Delay Distribution Analysis\n(Reassigned Trips Only)', fontweight='bold', fontsize=12)
            ax5.grid(axis='y', alpha=0.3)
            
            # Add mean markers
            for i, data in enumerate(data_to_plot):
                mean_val = np.mean(data)
                ax5.scatter(i+1, mean_val, color='green', s=100, marker='^', zorder=5)
                ax5.text(i+1, mean_val, f'μ={mean_val:.1f}', ha='center', va='bottom', fontsize=9)
        else:
            ax5.text(0.5, 0.5, 'No Delay Data', ha='center', va='center', fontsize=14)
            ax5.set_title('Delay Distribution Analysis', fontweight='bold', fontsize=12)
        
        # 6. Performance Summary Table (bottom right)
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        # Calculate improvements
        delay_improvement = baseline_metrics['avg_delay_per_reassigned'] - bo_metrics['avg_delay_per_reassigned']
        miles_improvement = baseline_metrics['avg_miles_per_reassigned'] - bo_metrics['avg_miles_per_reassigned']
        
        # Create detailed comparison table
        table_data = [
            ['Metric', 'Baseline', 'Optimized', 'Change'],
            ['', '', '', ''],  # Separator
            ['PER TRIP AVERAGES:', '', '', ''],
            ['Avg Delay (min)', 
             f"{baseline_metrics['avg_delay_per_reassigned']:.1f}",
             f"{bo_metrics['avg_delay_per_reassigned']:.1f}",
             f"{-delay_improvement:+.1f}"],
            ['Avg Extra Miles', 
             f"{baseline_metrics['avg_miles_per_reassigned']:.1f}",
             f"{bo_metrics['avg_miles_per_reassigned']:.1f}",
             f"{-miles_improvement:+.1f}"],
            ['', '', '', ''],  # Separator
            ['TOTALS:', '', '', ''],
            ['Total Delays (min)',
             f"{baseline_metrics['total_delay_minutes']:.0f}",
             f"{bo_metrics['total_delay_minutes']:.0f}",
             f"{bo_metrics['total_delay_minutes'] - baseline_metrics['total_delay_minutes']:+.0f}"],
            ['Total Extra Miles',
             f"{baseline_metrics['total_deadhead_miles']:.0f}",
             f"{bo_metrics['total_deadhead_miles']:.0f}",
             f"{bo_metrics['total_deadhead_miles'] - baseline_metrics['total_deadhead_miles']:+.0f}"],
            ['', '', '', ''],  # Separator
            ['ASSIGNMENTS:', '', '', ''],
            ['Reassigned',
             f"{baseline_metrics['reassigned_count']}",
             f"{bo_metrics['reassigned_count']}",
             f"{bo_metrics['reassigned_count'] - baseline_metrics['reassigned_count']:+d}"],
            ['Outsourced',
             f"{baseline_metrics['outsourced_count']}",
             f"{bo_metrics['outsourced_count']}",
             f"{bo_metrics['outsourced_count'] - baseline_metrics['outsourced_count']:+d}"]
        ]
        
        table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.5)
        
        # Style the header row
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style section headers
        for row_idx in [2, 6, 10]:
            for col_idx in range(4):
                table[(row_idx, col_idx)].set_facecolor('#95a5a6')
                table[(row_idx, col_idx)].set_text_props(weight='bold')
        
        # Color code improvements in the change column
        improvement_rows = [3, 4, 7, 8, 11, 12]  # Rows with actual metrics
        for row_idx in improvement_rows:
            try:
                change_text = table_data[row_idx][3]
                if change_text and change_text not in ['', 'Change']:
                    change_val = float(change_text)
                    
                    # For delays and miles, negative is better (reduction)
                    # For reassigned, positive is better (more internal handling)
                    # For outsourced, negative is better (less outsourcing)
                    if 'Delay' in table_data[row_idx][0] or 'Miles' in table_data[row_idx][0]:
                        color = '#d4f1d4' if change_val <= 0 else '#ffd4d4'
                    elif 'Reassigned' in table_data[row_idx][0]:
                        color = '#d4f1d4' if change_val >= 0 else '#ffd4d4'
                    elif 'Outsourced' in table_data[row_idx][0]:
                        color = '#d4f1d4' if change_val <= 0 else '#ffd4d4'
                    else:
                        color = 'white'
                    
                    table[(row_idx, 3)].set_facecolor(color)
            except (ValueError, IndexError):
                pass
        
        ax6.set_title('Performance Metrics Summary', fontweight='bold', fontsize=12, pad=20)
        
        # Add interpretation text below the figure
        fig.text(0.5, 0.02, 
                f'Key Insight: {"✓ Win-Win" if delay_improvement > 0 and miles_improvement > 0 else "⚖ Trade-off" if (delay_improvement > 0) != (miles_improvement > 0) else "≈ Similar"} | ' +
                f'Avg Delay {"↓" if delay_improvement > 0 else "↑" if delay_improvement < 0 else "="}{abs(delay_improvement):.1f} min | ' +
                f'Avg Miles {"↓" if miles_improvement > 0 else "↑" if miles_improvement < 0 else "="}{abs(miles_improvement):.1f} mi',
                ha='center', fontsize=11, weight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Overall title
        plt.suptitle('Trip Rescheduling Optimization: Operational Performance Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        
        return fig
    
    @staticmethod
    def create_simple_summary(baseline_solution, bo_solution, disrupted_trips) -> pd.DataFrame:
        """
        Create a simple summary DataFrame focusing on operational metrics.
        """
        baseline_metrics = MetricsVisualizer.extract_operational_metrics(baseline_solution)
        bo_metrics = MetricsVisualizer.extract_operational_metrics(bo_solution)
        
        summary_data = {
            'Metric': [
                'Total Disrupted Trips',
                'Successfully Reassigned',
                'Outsourced',
                'Average Delay per Reassigned Trip (min)',
                'Average Extra Miles per Reassigned Trip',
                'Total Delay Minutes',
                'Total Extra Miles',
                'Trips with Zero Delay',
                'Trips with Delays',
                'Maximum Single Delay (min)'
            ],
            'Baseline': [
                len(disrupted_trips),
                baseline_metrics['reassigned_count'],
                baseline_metrics['outsourced_count'],
                f"{baseline_metrics['avg_delay_per_reassigned']:.1f}",
                f"{baseline_metrics['avg_miles_per_reassigned']:.1f}",
                f"{baseline_metrics['total_delay_minutes']:.0f}",
                f"{baseline_metrics['total_deadhead_miles']:.0f}",
                baseline_metrics['trips_with_zero_delay'],
                baseline_metrics['trips_with_delays'],
                f"{baseline_metrics['max_delay']:.0f}"
            ],
            'Optimized (BO)': [
                len(disrupted_trips),
                bo_metrics['reassigned_count'],
                bo_metrics['outsourced_count'],
                f"{bo_metrics['avg_delay_per_reassigned']:.1f}",
                f"{bo_metrics['avg_miles_per_reassigned']:.1f}",
                f"{bo_metrics['total_delay_minutes']:.0f}",
                f"{bo_metrics['total_deadhead_miles']:.0f}",
                bo_metrics['trips_with_zero_delay'],
                bo_metrics['trips_with_delays'],
                f"{bo_metrics['max_delay']:.0f}"
            ]
        }
        
        return pd.DataFrame(summary_data)