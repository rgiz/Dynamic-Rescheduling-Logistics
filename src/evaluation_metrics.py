"""
Core Evaluation Metrics for Dynamic Trip Rescheduling
======================================================

This module provides the essential metrics for:
1. Cost optimization (deadhead, outsourcing, penalties)
2. SLA compliance (on-time delivery, delays)
3. Legal compliance (rest breaks, duty hours)
4. Operational efficiency (feasibility rates)

These metrics feed into both CP-SAT optimization and Bayesian Optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum


class ViolationType(Enum):
    """Types of constraint violations for compliance tracking."""
    DAILY_DUTY_EXCEEDED = "daily_duty_exceeded"  # >13 hours
    REST_PERIOD_VIOLATED = "rest_period_violated"  # <11h standard, <9h emergency
    EMERGENCY_QUOTA_EXCEEDED = "emergency_quota_exceeded"  # >2 per week
    WEEKEND_BREAK_VIOLATED = "weekend_break_violated"  # <45 hours
    DELAY_TOLERANCE_EXCEEDED = "delay_tolerance_exceeded"  # >2 hours


@dataclass
class CostMetrics:
    """
    Cost components for optimization.
    All costs in same currency unit (e.g., USD or INR).
    """
    deadhead_minutes: float = 0.0  # Total deadhead/empty travel time
    deadhead_cost: float = 0.0  # Cost of deadhead travel
    
    outsourcing_count: int = 0  # Number of trips outsourced
    outsourcing_cost: float = 0.0  # Total outsourcing cost
    
    emergency_rest_count: int = 0  # Number of emergency rests used
    emergency_rest_penalty: float = 0.0  # Penalty for using emergency rests
    
    reassignment_count: int = 0  # Number of reassignments
    reassignment_cost: float = 0.0  # Administrative cost of reassignments
    
    total_cost: float = 0.0  # Sum of all costs
    
    def calculate_total(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate total cost with configurable weights.
        
        Args:
            weights: Dict with keys like 'deadhead_weight', 'outsourcing_weight', etc.
                    If None, uses equal weights.
        """
        if weights is None:
            weights = {
                'deadhead_weight': 1.0,
                'outsourcing_weight': 1.0,
                'emergency_weight': 1.0,
                'reassignment_weight': 1.0
            }
        
        self.total_cost = (
            self.deadhead_cost * weights.get('deadhead_weight', 1.0) +
            self.outsourcing_cost * weights.get('outsourcing_weight', 1.0) +
            self.emergency_rest_penalty * weights.get('emergency_weight', 1.0) +
            self.reassignment_cost * weights.get('reassignment_weight', 1.0)
        )
        return self.total_cost


@dataclass
class SLAMetrics:
    """
    Service Level Agreement compliance metrics.
    Critical for maintaining customer satisfaction.
    """
    total_trips: int = 0
    on_time_deliveries: int = 0
    
    # Delay buckets for granular analysis
    delays_under_30min: int = 0
    delays_30_to_60min: int = 0
    delays_60_to_120min: int = 0
    delays_over_120min: int = 0
    
    average_delay_minutes: float = 0.0
    max_delay_minutes: float = 0.0
    
    # SLA compliance rate (e.g., 95% on-time target)
    on_time_rate: float = 0.0
    sla_target: float = 0.95  # Configurable target
    sla_compliant: bool = False
    
    # Service disruption score (for customer impact)
    service_disruption_score: float = 0.0
    
    def calculate_rates(self) -> None:
        """Calculate SLA compliance rates and scores."""
        if self.total_trips > 0:
            self.on_time_rate = self.on_time_deliveries / self.total_trips
            self.sla_compliant = self.on_time_rate >= self.sla_target
            
            # Service disruption increases with delay severity
            self.service_disruption_score = (
                self.delays_under_30min * 0.1 +
                self.delays_30_to_60min * 0.3 +
                self.delays_60_to_120min * 0.6 +
                self.delays_over_120min * 1.0
            ) / max(self.total_trips, 1)


@dataclass
class ComplianceMetrics:
    """
    Legal and regulatory compliance metrics.
    These are HARD constraints that must be satisfied.
    """
    total_drivers: int = 0
    
    # Violation counts by type
    violations: Dict[ViolationType, int] = field(default_factory=dict)
    
    # Detailed violation records for audit
    violation_details: List[Dict] = field(default_factory=list)
    
    # Compliance rates
    duty_hour_compliance_rate: float = 1.0  # % of days within 13h limit
    rest_period_compliance_rate: float = 1.0  # % of rest periods satisfied
    weekend_break_compliance_rate: float = 1.0  # % of weekends with 45h break
    
    # Overall compliance (all must be satisfied)
    fully_compliant: bool = True
    compliance_score: float = 1.0  # 0-1 score for optimization
    
    def add_violation(self, 
                      violation_type: ViolationType,
                      driver_id: str,
                      timestamp: datetime,
                      details: Dict) -> None:
        """Record a compliance violation."""
        # Increment counter
        if violation_type not in self.violations:
            self.violations[violation_type] = 0
        self.violations[violation_type] += 1
        
        # Store details for audit
        self.violation_details.append({
            'type': violation_type.value,
            'driver_id': driver_id,
            'timestamp': timestamp,
            'details': details
        })
        
        # Update compliance flag
        self.fully_compliant = False
    
    def calculate_compliance_score(self) -> float:
        """
        Calculate weighted compliance score for optimization.
        Hard violations get heavy penalties.
        """
        total_violations = sum(self.violations.values())
        
        if total_violations == 0:
            self.compliance_score = 1.0
        else:
            # Heavy penalties for violations
            duty_violations = self.violations.get(ViolationType.DAILY_DUTY_EXCEEDED, 0)
            rest_violations = self.violations.get(ViolationType.REST_PERIOD_VIOLATED, 0)
            weekend_violations = self.violations.get(ViolationType.WEEKEND_BREAK_VIOLATED, 0)
            
            # Weighted penalty (rest violations are most serious)
            penalty = (
                duty_violations * 0.3 +
                rest_violations * 0.5 +  # Most serious
                weekend_violations * 0.2
            )
            
            # Normalize by number of drivers and days
            max_possible_violations = self.total_drivers * 7  # Rough estimate
            self.compliance_score = max(0, 1 - (penalty / max_possible_violations))
        
        return self.compliance_score


@dataclass
class OperationalMetrics:
    """
    Operational efficiency metrics.
    Track how well the optimization performs.
    """
    total_disrupted_trips: int = 0
    successfully_reassigned: int = 0
    outsourced: int = 0
    cancelled: int = 0
    
    # Feasibility rate - key performance indicator
    feasibility_rate: float = 0.0
    
    # Cascading complexity
    single_driver_reassignments: int = 0
    two_driver_cascades: int = 0
    multi_driver_cascades: int = 0  # 3+ drivers
    max_cascade_depth: int = 0
    
    # Computational performance
    optimization_time_seconds: float = 0.0
    candidates_generated: int = 0
    candidates_evaluated: int = 0
    
    def calculate_feasibility_rate(self) -> float:
        """Calculate the percentage of disrupted trips successfully handled."""
        if self.total_disrupted_trips > 0:
            self.feasibility_rate = (
                self.successfully_reassigned / self.total_disrupted_trips
            )
        return self.feasibility_rate


@dataclass
class OptimizationMetrics:
    """
    Complete metrics suite for optimization and reporting.
    This is the main class that combines all metrics.
    """
    cost: CostMetrics = field(default_factory=CostMetrics)
    sla: SLAMetrics = field(default_factory=SLAMetrics)
    compliance: ComplianceMetrics = field(default_factory=ComplianceMetrics)
    operational: OperationalMetrics = field(default_factory=OperationalMetrics)
    
    # Multi-objective optimization scores
    total_cost_score: float = 0.0  # Normalized 0-1 (lower is better)
    service_quality_score: float = 1.0  # Normalized 0-1 (higher is better)
    compliance_score: float = 1.0  # Normalized 0-1 (higher is better)
    
    # Combined objective for single-objective optimization
    combined_objective: float = 0.0
    
    def calculate_objectives(self, 
                           cost_weight: float = 0.4,
                           service_weight: float = 0.3,
                           compliance_weight: float = 0.3) -> float:
        """
        Calculate multi-objective scores for optimization.
        
        Args:
            cost_weight: Weight for cost minimization (default 0.4)
            service_weight: Weight for service quality (default 0.3)
            compliance_weight: Weight for compliance (default 0.3)
            
        Returns:
            Combined objective value (lower is better)
        """
        # Normalize weights
        total_weight = cost_weight + service_weight + compliance_weight
        cost_weight /= total_weight
        service_weight /= total_weight
        compliance_weight /= total_weight
        
        # Calculate normalized scores
        # Cost: normalize by a reference value (e.g., outsourcing all trips)
        max_expected_cost = self.operational.total_disrupted_trips * 1000  # Placeholder
        self.total_cost_score = min(1.0, self.cost.total_cost / max(max_expected_cost, 1))
        
        # Service quality: based on SLA compliance and delays
        self.sla.calculate_rates()
        self.service_quality_score = (
            self.sla.on_time_rate * 0.6 +  # On-time rate is most important
            (1 - self.sla.service_disruption_score) * 0.4  # Minimize disruption
        )
        
        # Compliance: from compliance metrics
        self.compliance_score = self.compliance.calculate_compliance_score()
        
        # Combined objective (minimize this)
        self.combined_objective = (
            cost_weight * self.total_cost_score +
            service_weight * (1 - self.service_quality_score) +
            compliance_weight * (1 - self.compliance_score)
        )
        
        return self.combined_objective
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for reporting/logging."""
        return {
            'cost': {
                'total_cost': self.cost.total_cost,
                'deadhead_cost': self.cost.deadhead_cost,
                'outsourcing_cost': self.cost.outsourcing_cost,
                'emergency_rest_penalty': self.cost.emergency_rest_penalty,
            },
            'sla': {
                'on_time_rate': self.sla.on_time_rate,
                'average_delay_minutes': self.sla.average_delay_minutes,
                'sla_compliant': self.sla.sla_compliant,
            },
            'compliance': {
                'fully_compliant': self.compliance.fully_compliant,
                'compliance_score': self.compliance.compliance_score,
                'total_violations': sum(self.compliance.violations.values()),
            },
            'operational': {
                'feasibility_rate': self.operational.feasibility_rate,
                'successfully_reassigned': self.operational.successfully_reassigned,
                'optimization_time_seconds': self.operational.optimization_time_seconds,
            },
            'objectives': {
                'cost_score': self.total_cost_score,
                'service_quality_score': self.service_quality_score,
                'compliance_score': self.compliance_score,
                'combined_objective': self.combined_objective,
            }
        }
    
    def print_summary(self) -> None:
        """Print a formatted summary of metrics."""
        print("\n" + "="*60)
        print("OPTIMIZATION METRICS SUMMARY")
        print("="*60)
        
        print("\n📊 OPERATIONAL PERFORMANCE")
        print(f"  • Feasibility Rate: {self.operational.feasibility_rate:.1%}")
        print(f"  • Successfully Reassigned: {self.operational.successfully_reassigned}/{self.operational.total_disrupted_trips}")
        print(f"  • Optimization Time: {self.operational.optimization_time_seconds:.2f}s")
        
        print("\n💰 COST METRICS")
        print(f"  • Total Cost: ${self.cost.total_cost:,.2f}")
        print(f"  • Deadhead Cost: ${self.cost.deadhead_cost:,.2f}")
        print(f"  • Outsourcing Cost: ${self.cost.outsourcing_cost:,.2f}")
        
        print("\n📋 SLA COMPLIANCE")
        print(f"  • On-Time Rate: {self.sla.on_time_rate:.1%}")
        print(f"  • Average Delay: {self.sla.average_delay_minutes:.1f} min")
        print(f"  • SLA Target Met: {'✅' if self.sla.sla_compliant else '❌'}")
        
        print("\n⚖️ REGULATORY COMPLIANCE")
        print(f"  • Fully Compliant: {'✅' if self.compliance.fully_compliant else '❌'}")
        print(f"  • Compliance Score: {self.compliance.compliance_score:.1%}")
        if self.compliance.violations:
            print(f"  • Violations: {sum(self.compliance.violations.values())}")
        
        print("\n🎯 MULTI-OBJECTIVE SCORES")
        print(f"  • Cost Score: {self.total_cost_score:.3f} (lower is better)")
        print(f"  • Service Quality: {self.service_quality_score:.3f} (higher is better)")
        print(f"  • Combined Objective: {self.combined_objective:.3f} (lower is better)")
        print("="*60)


class MetricsCalculator:
    """
    Utility class to calculate metrics from optimization results.
    This bridges the gap between raw optimization output and metrics.
    """
    
    def __init__(self, 
                 cost_per_minute_deadhead: float = 1.0,
                 cost_per_outsourced_trip: float = 500.0,
                 emergency_rest_penalty: float = 100.0,
                 reassignment_admin_cost: float = 20.0):
        """
        Initialize calculator with cost parameters.
        
        Args:
            cost_per_minute_deadhead: Cost per minute of deadhead travel
            cost_per_outsourced_trip: Fixed cost per outsourced trip
            emergency_rest_penalty: Penalty for using emergency rest
            reassignment_admin_cost: Administrative cost per reassignment
        """
        self.cost_per_minute_deadhead = cost_per_minute_deadhead
        self.cost_per_outsourced_trip = cost_per_outsourced_trip
        self.emergency_rest_penalty = emergency_rest_penalty
        self.reassignment_admin_cost = reassignment_admin_cost
    
    def calculate_from_solution(self,
                               original_trips: pd.DataFrame,
                               reassignments: List[Dict],
                               driver_states: Dict) -> OptimizationMetrics:
        """
        Calculate complete metrics from an optimization solution.
        
        Args:
            original_trips: DataFrame of original disrupted trips
            reassignments: List of reassignment decisions
            driver_states: Dict of DriverState objects after optimization
            
        Returns:
            Complete OptimizationMetrics object
        """
        metrics = OptimizationMetrics()
        
        # Set operational basics
        metrics.operational.total_disrupted_trips = len(original_trips)
        
        # Process each reassignment
        for assignment in reassignments:
            if assignment['type'] == 'reassigned':
                metrics.operational.successfully_reassigned += 1
                metrics.cost.reassignment_count += 1
                
                # Check cascade depth
                cascade_depth = assignment.get('cascade_depth', 1)
                if cascade_depth == 1:
                    metrics.operational.single_driver_reassignments += 1
                elif cascade_depth == 2:
                    metrics.operational.two_driver_cascades += 1
                else:
                    metrics.operational.multi_driver_cascades += 1
                metrics.operational.max_cascade_depth = max(
                    metrics.operational.max_cascade_depth, cascade_depth
                )
                
                # Calculate deadhead if present
                if 'deadhead_minutes' in assignment:
                    metrics.cost.deadhead_minutes += assignment['deadhead_minutes']
                
                # Check for delays
                if 'delay_minutes' in assignment:
                    delay = assignment['delay_minutes']
                    if delay <= 0:
                        metrics.sla.on_time_deliveries += 1
                    elif delay < 30:
                        metrics.sla.delays_under_30min += 1
                    elif delay < 60:
                        metrics.sla.delays_30_to_60min += 1
                    elif delay < 120:
                        metrics.sla.delays_60_to_120min += 1
                    else:
                        metrics.sla.delays_over_120min += 1
                    
                    metrics.sla.average_delay_minutes += delay
                    metrics.sla.max_delay_minutes = max(
                        metrics.sla.max_delay_minutes, delay
                    )
                    
            elif assignment['type'] == 'outsourced':
                metrics.operational.outsourced += 1
                metrics.cost.outsourcing_count += 1
                
            elif assignment['type'] == 'cancelled':
                metrics.operational.cancelled += 1
        
        # Calculate costs
        metrics.cost.deadhead_cost = (
            metrics.cost.deadhead_minutes * self.cost_per_minute_deadhead
        )
        metrics.cost.outsourcing_cost = (
            metrics.cost.outsourcing_count * self.cost_per_outsourced_trip
        )
        metrics.cost.reassignment_cost = (
            metrics.cost.reassignment_count * self.reassignment_admin_cost
        )
        
        # Check driver states for compliance
        metrics.compliance.total_drivers = len(driver_states)
        for driver_id, driver_state in driver_states.items():
            # Check for emergency rest usage
            if hasattr(driver_state, 'emergency_rests_used'):
                metrics.cost.emergency_rest_count += driver_state.emergency_rests_used
                
            # Check for violations (would need actual validation logic)
            # This is a placeholder - actual implementation would check constraints
            pass
        
        metrics.cost.emergency_rest_penalty = (
            metrics.cost.emergency_rest_count * self.emergency_rest_penalty
        )
        
        # Calculate totals and rates
        metrics.cost.calculate_total()
        metrics.sla.total_trips = metrics.operational.total_disrupted_trips
        metrics.sla.calculate_rates()
        metrics.operational.calculate_feasibility_rate()
        
        # Calculate final objectives
        metrics.calculate_objectives()
        
        return metrics


# Example usage for Bayesian Optimization objective function
def create_bo_objective(metrics_calculator: MetricsCalculator):
    """
    Create an objective function for Bayesian Optimization.
    
    Returns a function that takes optimization parameters and returns
    the combined objective value to minimize.
    """
    def objective(params):
        """
        Objective function for BO.
        
        Args:
            params: Dict with keys like 'cost_weight', 'service_weight', etc.
            
        Returns:
            Combined objective value (lower is better)
        """
        # Run optimization with these parameters
        # (This would be your CP-SAT model)
        # solution = run_cpsat_with_params(params)
        # metrics = metrics_calculator.calculate_from_solution(solution)
        
        # For now, return placeholder
        # In practice, this would return metrics.combined_objective
        pass
    
    return objective