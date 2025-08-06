"""
Bayesian Optimization Layer for Dynamic Trip Rescheduling
=========================================================

Uses Bayesian Optimization to find optimal weight configurations for the
multi-objective CP-SAT model. Explores the tradeoff between:
- Cost minimization
- Service quality maximization

Compliance rules are enforced as HARD CONSTRAINTS in the CP-SAT model,
not as part of the objective function.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

# Use Optuna for Bayesian Optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    raise ImportError("Optuna is required. Install with: pip install optuna")

from opt.cpsat_model_v2 import CPSATOptimizer, CPSATSolution
from evaluation_metrics import OptimizationMetrics


@dataclass
class BOTrialResult:
    """
    Result from a single Bayesian Optimization trial.
    """
    trial_id: int
    timestamp: datetime
    
    # Parameters tested
    parameters: Dict[str, float]
    
    # Objective values (only cost and service matter for optimization)
    cost_score: float
    service_score: float
    combined_objective: float
    
    # Solution details
    feasibility_rate: float
    total_cost: float
    on_time_rate: float
    violations: int  # Track but don't optimize - should be 0 if constraints work
    
    # Computational metrics
    solve_time_seconds: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'trial_id': self.trial_id,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters,
            'cost_score': self.cost_score,
            'service_score': self.service_score,
            'combined_objective': self.combined_objective,
            'feasibility_rate': self.feasibility_rate,
            'total_cost': self.total_cost,
            'on_time_rate': self.on_time_rate,
            'violations': self.violations,
            'solve_time_seconds': self.solve_time_seconds
        }


class BayesianOptimizationTuner:
    """
    Bayesian Optimization tuner for CP-SAT model weights.
    Optimizes the tradeoff between cost and service quality.
    Compliance is enforced through hard constraints.
    """
    
    def __init__(self,
                 cpsat_optimizer: CPSATOptimizer,
                 results_dir: Optional[Path] = None):
        """
        Initialize the BO tuner.
        
        Args:
            cpsat_optimizer: The CP-SAT optimizer to tune
            results_dir: Directory to save results (optional)
        """
        self.cpsat_optimizer = cpsat_optimizer
        self.results_dir = Path(results_dir) if results_dir else Path("bo_results")
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Trial history
        self.trial_results: List[BOTrialResult] = []
        self.best_parameters: Optional[Dict[str, float]] = None
        
        # Current optimization context
        self.current_disrupted_trips = None
        self.trial_counter = 0
    
    def optimize_single_objective(self,
                                 disrupted_trips: List[Dict],
                                 n_trials: int = 50,
                                 parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict:
        """
        Run single-objective Bayesian Optimization.
        
        Args:
            disrupted_trips: Trips to optimize
            n_trials: Number of BO trials
            parameter_bounds: Optional custom bounds for parameters
            
        Returns:
            Best parameters found
        """
        self.current_disrupted_trips = disrupted_trips
        
        # Set default parameter bounds if not provided
        # Note: We only optimize cost vs service tradeoff, not compliance
        if parameter_bounds is None:
            parameter_bounds = {
                'cost_weight': (0.1, 0.9),  # Will be normalized with service_weight
                'service_weight': (0.1, 0.9),  # Will be normalized with cost_weight
                'max_cascade_depth': (1, 3),
                'max_deadhead_minutes': (30, 120),
                'max_delay_minutes': (30, 180)
            }
        
        return self._optimize_with_optuna(disrupted_trips, n_trials, parameter_bounds)
    
    def _optimize_with_optuna(self,
                             disrupted_trips: List[Dict],
                             n_trials: int,
                             parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict:
        """
        Optimize using Optuna.
        """
        def objective_function(trial):
            # Sample parameters
            params = {}
            for param_name, (low, high) in parameter_bounds.items():
                if param_name == 'max_cascade_depth':
                    params[param_name] = trial.suggest_int(param_name, int(low), int(high))
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)
            
            # Evaluate with CP-SAT
            result = self._evaluate_parameters(params, disrupted_trips)
            
            # Store trial result
            self.trial_results.append(result)
            
            # Return objective to minimize
            # If solution is infeasible (violates hard constraints), return a large penalty
            if result.feasibility_rate == 0 or result.violations > 0:
                return float('inf')
            
            return result.combined_objective
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            study_name=f"trip_rescheduling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Optimize
        study.optimize(objective_function, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        self.best_parameters = study.best_params
        
        # Save results
        self._save_results(study)
        
        return self.best_parameters
    
    def _evaluate_parameters(self,
                            params: Dict[str, float],
                            disrupted_trips: List[Dict]) -> BOTrialResult:
        """
        Evaluate a parameter configuration using CP-SAT.
        
        THIS IS THE KEY METHOD WHERE THE OBJECTIVE IS DEFINED.
        """
        self.trial_counter += 1
        
        # Extract and normalize cost/service weights
        cost_weight = params.get('cost_weight', 0.5)
        service_weight = params.get('service_weight', 0.5)
        
        # Normalize so they sum to 1.0
        total_weight = cost_weight + service_weight
        cost_weight_norm = cost_weight / total_weight
        service_weight_norm = service_weight / total_weight
        
        # CRITICAL: Set compliance_weight to 0 
        # Compliance is enforced through hard constraints, not the objective
        objective_weights = {
            'cost_weight': cost_weight_norm,
            'service_weight': service_weight_norm,
            'compliance_weight': 0.0  # ALWAYS 0 - compliance is a hard constraint
        }
        
        # Configure candidate generator if parameters provided
        if 'max_cascade_depth' in params:
            self.cpsat_optimizer.candidate_generator.max_cascade_depth = int(params['max_cascade_depth'])
        if 'max_deadhead_minutes' in params:
            self.cpsat_optimizer.candidate_generator.max_deadhead_minutes = params['max_deadhead_minutes']
        if 'max_delay_minutes' in params:
            self.cpsat_optimizer.candidate_generator.max_delay_minutes = params['max_delay_minutes']
        
        # Run optimization
        solution = self.cpsat_optimizer.optimize(
            disrupted_trips,
            objective_weights=objective_weights,
            include_cascades=params.get('max_cascade_depth', 2) > 1
        )
        
        # Extract metrics
        if solution.metrics and solution.is_feasible():
            metrics = solution.metrics
            
            # Calculate combined objective (cost vs service only)
            # Lower is better for both components
            combined_objective = (
                cost_weight_norm * metrics.total_cost_score +
                service_weight_norm * (1.0 - metrics.service_quality_score)
            )
            
            result = BOTrialResult(
                trial_id=self.trial_counter,
                timestamp=datetime.now(),
                parameters=params,
                cost_score=metrics.total_cost_score,
                service_score=metrics.service_quality_score,
                combined_objective=combined_objective,
                feasibility_rate=metrics.operational.feasibility_rate,
                total_cost=metrics.cost.total_cost,
                on_time_rate=metrics.sla.on_time_rate,
                violations=sum(metrics.compliance.violations.values()),
                solve_time_seconds=solution.solve_time_seconds
            )
        else:
            # Infeasible solution (violates hard constraints)
            result = BOTrialResult(
                trial_id=self.trial_counter,
                timestamp=datetime.now(),
                parameters=params,
                cost_score=1.0,
                service_score=0.0,
                combined_objective=float('inf'),  # Large penalty for infeasibility
                feasibility_rate=0.0,
                total_cost=float('inf'),
                on_time_rate=0.0,
                violations=999,  # Flag as violating constraints
                solve_time_seconds=solution.solve_time_seconds if solution else 0.0
            )
        
        # Print progress
        print(f"\nTrial {self.trial_counter}:")
        print(f"  Cost weight: {cost_weight_norm:.2f}, Service weight: {service_weight_norm:.2f}")
        print(f"  Combined objective: {result.combined_objective:.4f}")
        print(f"  Feasibility rate: {result.feasibility_rate:.2%}")
        if result.violations > 0:
            print(f"  ⚠️ Violations detected: {result.violations}")
        
        return result
    
    def _save_results(self, study):
        """
        Save optimization results to disk.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save trial history
        trials_df = pd.DataFrame([t.to_dict() for t in self.trial_results])
        trials_df.to_csv(self.results_dir / f"trials_{timestamp}.csv", index=False)
        
        # Save best parameters
        if self.best_parameters:
            with open(self.results_dir / f"best_params_{timestamp}.json", 'w') as f:
                json.dump(self.best_parameters, f, indent=2)
        
        # Save Optuna study
        import pickle
        study_path = self.results_dir / f"study_{timestamp}.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        
        print(f"\n📁 Results saved to {self.results_dir}")
    
    def plot_optimization_progress(self):
        """
        Plot optimization progress over trials.
        """
        if not self.trial_results:
            print("No trials to plot")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        trials = list(range(1, len(self.trial_results) + 1))
        objectives = [r.combined_objective if r.combined_objective != float('inf') else None 
                     for r in self.trial_results]
        feasibility = [r.feasibility_rate for r in self.trial_results]
        costs = [r.total_cost if r.total_cost != float('inf') else None 
                for r in self.trial_results]
        on_time = [r.on_time_rate for r in self.trial_results]
        
        # Plot 1: Objective over trials
        valid_objectives = [(i, obj) for i, obj in enumerate(objectives) if obj is not None]
        if valid_objectives:
            valid_trials, valid_objs = zip(*valid_objectives)
            valid_trials = [t + 1 for t in valid_trials]
            axes[0, 0].plot(valid_trials, valid_objs, 'b-', alpha=0.6)
            axes[0, 0].scatter(valid_trials, valid_objs, 
                             c=[feasibility[i] for i, _ in valid_objectives], 
                             cmap='RdYlGn', s=30)
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('Combined Objective')
        axes[0, 0].set_title('Optimization Progress (Cost vs Service)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cost vs Service tradeoff
        valid_points = [(r.cost_score, r.service_score, i) 
                       for i, r in enumerate(self.trial_results) 
                       if r.feasibility_rate > 0]
        if valid_points:
            cost_scores, service_scores, indices = zip(*valid_points)
            axes[0, 1].scatter(cost_scores, service_scores,
                             c=[i+1 for i in indices], cmap='viridis', s=50)
        axes[0, 1].set_xlabel('Cost Score (lower is better)')
        axes[0, 1].set_ylabel('Service Score (higher is better)')
        axes[0, 1].set_title('Cost vs Service Tradeoff')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Feasibility rate
        axes[1, 0].plot(trials, feasibility, 'g-', linewidth=2)
        axes[1, 0].fill_between(trials, 0, feasibility, alpha=0.3, color='green')
        axes[1, 0].set_xlabel('Trial')
        axes[1, 0].set_ylabel('Feasibility Rate')
        axes[1, 0].set_title('Solution Feasibility (Compliance as Hard Constraint)')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Parameter evolution (only cost and service weights)
        cost_weights = [r.parameters.get('cost_weight', 0) for r in self.trial_results]
        service_weights = [r.parameters.get('service_weight', 0) for r in self.trial_results]
        
        # Normalize for display
        normalized_costs = []
        normalized_services = []
        for cw, sw in zip(cost_weights, service_weights):
            total = cw + sw
            normalized_costs.append(cw / total if total > 0 else 0)
            normalized_services.append(sw / total if total > 0 else 0)
        
        axes[1, 1].plot(trials, normalized_costs, label='Cost Weight', linewidth=2, color='red')
        axes[1, 1].plot(trials, normalized_services, label='Service Weight', linewidth=2, color='blue')
        axes[1, 1].set_xlabel('Trial')
        axes[1, 1].set_ylabel('Normalized Weight Value')
        axes[1, 1].set_title('Weight Evolution (Cost vs Service)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
        
        plt.suptitle('Bayesian Optimization Progress - Compliance as Hard Constraints', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'optimization_progress.png', dpi=150)
        plt.show()
        
        print(f"📊 Plot saved to {self.results_dir / 'optimization_progress.png'}")
    
    def get_recommendations(self) -> Dict:
        """
        Get recommendations based on optimization results.
        """
        if not self.trial_results:
            return {"error": "No optimization results available"}
        
        # Filter out infeasible solutions
        feasible_trials = [r for r in self.trial_results if r.feasibility_rate > 0 and r.violations == 0]
        
        if not feasible_trials:
            return {"error": "No feasible solutions found"}
        
        # Find best configurations for different objectives
        best_cost = min(feasible_trials, key=lambda x: x.cost_score)
        best_service = max(feasible_trials, key=lambda x: x.service_score)
        best_combined = min(feasible_trials, key=lambda x: x.combined_objective)
        
        recommendations = {
            "best_overall": {
                "parameters": best_combined.parameters,
                "metrics": {
                    "combined_objective": best_combined.combined_objective,
                    "feasibility_rate": best_combined.feasibility_rate,
                    "total_cost": best_combined.total_cost,
                    "on_time_rate": best_combined.on_time_rate,
                    "violations": best_combined.violations
                }
            },
            "best_for_cost": {
                "parameters": best_cost.parameters,
                "cost_score": best_cost.cost_score,
                "total_cost": best_cost.total_cost
            },
            "best_for_service": {
                "parameters": best_service.parameters,
                "service_score": best_service.service_score,
                "on_time_rate": best_service.on_time_rate
            },
            "feasible_trials": len(feasible_trials),
            "infeasible_trials": len(self.trial_results) - len(feasible_trials),
            "total_trials": len(self.trial_results)
        }
        
        return recommendations


# Example usage
def example_bayesian_optimization():
    """
    Example of running Bayesian Optimization with compliance as hard constraints.
    """
    from models.driver_state import DriverState
    
    # Create dummy driver states
    driver_states = {}
    for i in range(10):
        driver_states[f"driver_{i}"] = DriverState(
            driver_id=f"driver_{i}",
            route_id=f"route_{i}"
        )
    
    # Initialize CP-SAT optimizer
    cpsat_optimizer = CPSATOptimizer(driver_states)
    
    # Initialize BO tuner
    bo_tuner = BayesianOptimizationTuner(
        cpsat_optimizer=cpsat_optimizer,
        results_dir="bo_results"
    )
    
    # Define disrupted trips
    disrupted_trips = [
        {
            'id': f'TRIP_{i:03d}',
            'start_time': datetime(2024, 1, 15, 9 + i, 0),
            'end_time': datetime(2024, 1, 15, 13 + i, 0),
            'duration_minutes': 240,
            'start_location': 'Delhi_DC',
            'end_location': 'Mumbai_DC'
        }
        for i in range(5)
    ]
    
    print("="*60)
    print("BAYESIAN OPTIMIZATION FOR TRIP RESCHEDULING")
    print("Compliance enforced as HARD CONSTRAINTS")
    print("Optimizing: Cost vs Service Quality Tradeoff")
    print("="*60)
    
    # Run single-objective optimization
    print("\n📊 Running Bayesian optimization...")
    best_params = bo_tuner.optimize_single_objective(
        disrupted_trips,
        n_trials=20
    )
    
    print(f"\n✅ Best parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value:.3f}")
    
    # Get recommendations
    recommendations = bo_tuner.get_recommendations()
    print(f"\n📋 Recommendations:")
    print(f"  Feasible trials: {recommendations['feasible_trials']}/{recommendations['total_trials']}")
    print(f"  Best feasibility rate: {recommendations['best_overall']['metrics']['feasibility_rate']:.2%}")
    print(f"  Best total cost: ${recommendations['best_overall']['metrics']['total_cost']:.2f}")
    print(f"  Best on-time rate: {recommendations['best_overall']['metrics']['on_time_rate']:.2%}")
    print(f"  Compliance violations: {recommendations['best_overall']['metrics']['violations']}")
    
    # Plot results
    bo_tuner.plot_optimization_progress()
    
    return bo_tuner


if __name__ == "__main__":
    # Run example
    tuner = example_bayesian_optimization()