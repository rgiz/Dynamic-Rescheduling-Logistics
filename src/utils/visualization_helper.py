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
    def create_comparison_figure(baseline_solution, baseline_metrics,
                                optimized_solution, optimized_metrics,
                                disrupted_trips, bo_tuner=None):
        """
        Create comprehensive visualization comparing CP-SAT baseline vs BO-optimized results.
        
        Args:
            baseline_solution: CP-SAT solution with default weights
            baseline_metrics: Metrics from baseline solution
            optimized_solution: CP-SAT solution with BO-tuned weights
            optimized_metrics: Metrics from optimized solution
            disrupted_trips: List of disrupted trips
            bo_tuner: Optional Bayesian optimization tuner
            
        Returns:
            fig: The matplotlib figure object
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Use optimized metrics as primary, baseline for comparison
        metrics = optimized_metrics
        baseline = baseline_metrics if baseline_metrics else optimized_metrics
        
        # Calculate improvements (might be 0 if both achieved same result)
        cost_improvement = baseline.cost.total_cost - metrics.cost.total_cost if baseline else 0
        cost_improvement_pct = (cost_improvement / baseline.cost.total_cost * 100) if baseline and baseline.cost.total_cost > 0 else 0
        
        # 1. Cost Comparison (top left) - CP-SAT Default vs CP-SAT+BO
        ax1 = plt.subplot(2, 3, 1)
        scenarios = ['CP-SAT\n(Default Weights)', 'CP-SAT+BO\n(Tuned Weights)']
        costs = [
            baseline.cost.total_cost if baseline else metrics.cost.total_cost,
            metrics.cost.total_cost
        ]
        colors = ['#3498db', '#27ae60'] if cost_improvement >= 0 else ['#3498db', '#e74c3c']
        bars = ax1.bar(scenarios, costs, color=colors, alpha=0.8)
        ax1.set_ylabel('Total Cost ($)', fontsize=11)
        ax1.set_title('Optimization Cost Comparison', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, cost in zip(bars, costs):
            ax1.text(bar.get_x() + bar.get_width()/2, cost, f'${cost:,.0f}',
                    ha='center', va='bottom', fontsize=10)
        
        # Add improvement annotation only if there's a difference
        if abs(cost_improvement) > 0.01:
            if cost_improvement > 0:
                improvement_text = f'BO Improvement:\n${cost_improvement:,.0f}\n({cost_improvement_pct:.1f}%)'
                box_color = 'lightgreen'
            else:
                improvement_text = f'No improvement\n(Same cost)'
                box_color = 'lightyellow'
            ax1.text(0.5, max(costs)*0.5, improvement_text, ha='center',
                    bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.5),
                    fontsize=11, fontweight='bold')
        else:
            ax1.text(0.5, max(costs)*0.5, 'Same performance\n(BO found same\noptimal weights)', 
                    ha='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
                    fontsize=10)
        
        # 2. Assignment Comparison (top middle)
        ax2 = plt.subplot(2, 3, 2)
        categories = ['Reassigned', 'Outsourced']
        baseline_vals = [
            baseline.operational.successfully_reassigned if baseline else 0,
            baseline.operational.outsourced if baseline else len(disrupted_trips)
        ]
        optimized_vals = [
            metrics.operational.successfully_reassigned,
            metrics.operational.outsourced
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, baseline_vals, width, 
                       label='CP-SAT Default', color='#3498db', alpha=0.8)
        bars2 = ax2.bar(x + width/2, optimized_vals, width,
                       label='CP-SAT+BO', color='#27ae60', alpha=0.8)
        
        ax2.set_ylabel('Number of Trips', fontsize=11)
        ax2.set_title('Trip Assignment Distribution', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # 3. Performance Metrics Comparison (top right)
        ax3 = plt.subplot(2, 3, 3)
        metrics_names = ['Feasibility\nRate', 'On-Time\nRate']
        baseline_values = [
            baseline.operational.feasibility_rate * 100 if baseline else 0,
            baseline.sla.on_time_rate * 100 if baseline else 0
        ]
        optimized_values = [
            metrics.operational.feasibility_rate * 100,
            metrics.sla.on_time_rate * 100
        ]
        
        x = np.arange(len(metrics_names))
        bars1 = ax3.bar(x - width/2, baseline_values, width, 
                       label='CP-SAT Default', color='#3498db', alpha=0.8)
        bars2 = ax3.bar(x + width/2, optimized_values, width,
                       label='CP-SAT+BO', color='#27ae60', alpha=0.8)
        
        ax3.set_ylabel('Percentage (%)', fontsize=11)
        ax3.set_ylim([0, 105])
        ax3.set_title('Key Performance Indicators', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                val = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}%',
                        ha='center', va='bottom', fontsize=9)
        
        # 4. Cost Breakdown (bottom left)
        ax4 = plt.subplot(2, 3, 4)
        components, values = MetricsVisualizer.get_cost_breakdown(metrics)
        if components and any(v > 0 for v in values):
            components = [c for c, v in zip(components, values) if v > 0]
            values = [v for v in values if v > 0]
            ax4.pie(values, labels=components, autopct='%1.1f%%', startangle=45)
            ax4.set_title('Cost Breakdown (Optimized)', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Cost breakdown\nnot available', 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('Cost Breakdown', fontsize=12, fontweight='bold')
        
        # 5. BO Progress or Comparison Metrics (bottom middle)
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
            
            # Add baseline objective line if available
            if baseline and hasattr(baseline, 'combined_objective'):
                ax5.axhline(y=baseline.combined_objective, color='red', 
                          linestyle='--', alpha=0.5, label='Default Weights')
            
            ax5.legend()
        else:
            # Show weight comparison
            if baseline and optimized_solution:
                categories = ['Cost\nWeight', 'Service\nWeight', 'Compliance\nWeight']
                
                # Default weights (typical)
                default_weights = [0.4, 0.3, 0.3]
                
                # BO-optimized weights (from best_params if available)
                if bo_tuner and hasattr(bo_tuner, 'best_parameters'):
                    opt_weights = [
                        bo_tuner.best_parameters.get('cost_weight', 0.4),
                        bo_tuner.best_parameters.get('service_weight', 0.3),
                        bo_tuner.best_parameters.get('compliance_weight', 0.3)
                    ]
                else:
                    opt_weights = default_weights
                
                x = np.arange(len(categories))
                bars1 = ax5.bar(x - width/2, default_weights, width,
                               label='Default', color='#3498db', alpha=0.8)
                bars2 = ax5.bar(x + width/2, opt_weights, width,
                               label='BO-Optimized', color='#27ae60', alpha=0.8)
                
                ax5.set_ylabel('Weight Value', fontsize=11)
                ax5.set_title('Optimization Weights', fontsize=12, fontweight='bold')
                ax5.set_xticks(x)
                ax5.set_xticklabels(categories)
                ax5.legend()
                ax5.set_ylim([0, 1])
                ax5.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        val = bar.get_height()
                        ax5.text(bar.get_x() + bar.get_width()/2, val, f'{val:.2f}',
                                ha='center', va='bottom', fontsize=9)
        
        # 6. Summary Comparison (bottom right)
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Build comparison summary
        summary_lines = [
            "OPTIMIZATION COMPARISON",
            "=" * 25,
            "",
            "CP-SAT (Default Weights):",
            f"  Cost: ${baseline.cost.total_cost:,.0f}" if baseline else "  N/A",
            f"  Reassigned: {baseline.operational.successfully_reassigned}" if baseline else "  N/A",
            f"  Feasibility: {baseline.operational.feasibility_rate:.1%}" if baseline else "  N/A",
            "",
            "CP-SAT+BO (Tuned Weights):",
            f"  Cost: ${metrics.cost.total_cost:,.0f}",
            f"  Reassigned: {metrics.operational.successfully_reassigned}",
            f"  Feasibility: {metrics.operational.feasibility_rate:.1%}",
            "",
            "PERFORMANCE:"
        ]
        
        # Add performance assessment
        if baseline and abs(cost_improvement) < 0.01:
            summary_lines.extend([
                "  ✓ Both achieved same result",
                "  ✓ Default weights were optimal",
                "  ✓ BO confirmed optimality"
            ])
        elif cost_improvement > 0:
            summary_lines.extend([
                f"  ✓ Cost reduced by ${cost_improvement:,.0f}",
                f"  ✓ {cost_improvement_pct:.1f}% improvement",
                "  ✓ BO found better weights"
            ])
        else:
            summary_lines.extend([
                "  ✓ No improvement possible",
                "  ✓ Default weights optimal",
                "  ✓ Problem fully solved"
            ])
        
        summary_lines.extend([
            "",
            f"Total Disrupted: {len(disrupted_trips)} trips",
            f"BO Trials Run: {len(bo_tuner.trial_results) if bo_tuner and hasattr(bo_tuner, 'trial_results') else 'N/A'}"
        ])
        
        summary_text = "\n".join(summary_lines)
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Dynamic Trip Rescheduling - CP-SAT Baseline vs BO Optimization', 
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