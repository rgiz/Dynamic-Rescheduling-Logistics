"""
Bayesian Optimization Layer for Dynamic Trip Rescheduling
=========================================================

Uses Bayesian Optimization to find optimal weight configurations for the
multi-objective CP-SAT model. Explores the Pareto frontier between:
- Cost minimization
- Service quality maximization  
- Compliance assurance

Learns from historical performance to improve over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
from pathlib import Path

# Bayesian Optimization libraries
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    print("Warning: scikit-optimize not installed. Install with: pip install scikit-optimize")
    SKOPT_AVAILABLE = False

# Alternative: Optuna (more modern, better for multi-objective)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Warning: Optuna not installed. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False

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
    
    # Objective values
    cost_score: float
    service_score: float
    compliance_score: float
    combined_objective: float
    
    # Solution details
    feasibility_rate: float
    total_cost: float
    on_time_rate: float
    violations: int
    
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
            'compliance_score': self.compliance_score,
            'combined_objective': self.combined_objective,
            'feasibility_rate': self.feasibility_rate,
            'total_cost': self.total_cost,
            'on_time_rate': self.on_time_rate,
            'violations': self.violations,
            'solve_time_seconds': self.solve_time_seconds
        }


@dataclass
class ParetoPoint:
    """
    A point on the Pareto frontier.
    """
    parameters: Dict[str, float]
    objectives: Dict[str, float]  # cost, service, compliance
    dominates: List[int] = field(default_factory=list)  # Trial IDs this point dominates
    dominated_by: List[int] = field(default_factory=list)  # Trial IDs that dominate this
    
    def is_pareto_optimal(self) -> bool:
        """Check if this point is on the Pareto frontier."""
        return len(self.dominated_by) == 0


class BayesianOptimizationTuner:
    """
    Bayesian Optimization tuner for CP-SAT model weights.
    Supports both single-objective and multi-objective optimization.
    """
    
    def __init__(self,
                 cpsat_optimizer: CPSATOptimizer,
                 optimization_backend: str = 'optuna',  # 'optuna' or 'skopt'
                 results_dir: Optional[Path] = None):
        """
        Initialize the BO tuner.
        
        Args:
            cpsat_optimizer: The CP-SAT optimizer to tune
            optimization_backend: Which BO library to use
            results_dir: Directory to save results (optional)
        """
        self.cpsat_optimizer = cpsat_optimizer
        self.optimization_backend = optimization_backend
        self.results_dir = Path(results_dir) if results_dir else Path("bo_results")
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Trial history
        self.trial_results: List[BOTrialResult] = []
        self.best_parameters: Optional[Dict[str, float]] = None
        self.pareto_frontier: List[ParetoPoint] = []
        
        # Current optimization context
        self.current_disrupted_trips = None
        self.trial_counter = 0
    
    def optimize_single_objective(self,
                                 disrupted_trips: List[Dict],
                                 n_trials: int = 50,
                                 objective: str = 'combined',
                                 parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict:
        """
        Run single-objective Bayesian Optimization.
        
        Args:
            disrupted_trips: Trips to optimize
            n_trials: Number of BO trials
            objective: Which objective to optimize ('combined', 'cost', 'service', 'compliance')
            parameter_bounds: Optional custom bounds for parameters
            
        Returns:
            Best parameters found
        """
        self.current_disrupted_trips = disrupted_trips
        
        # Set default parameter bounds if not provided
        if parameter_bounds is None:
            parameter_bounds = {
                'cost_weight': (0.1, 0.7),
                'service_weight': (0.1, 0.7),
                'compliance_weight': (0.1, 0.7),
                'max_cascade_depth': (1, 4),
                'max_deadhead_minutes': (60, 180),
                'max_delay_minutes': (30, 180)
            }
        
        if self.optimization_backend == 'optuna' and OPTUNA_AVAILABLE:
            return self._optimize_with_optuna(
                disrupted_trips, n_trials, objective, parameter_bounds
            )
        elif self.optimization_backend == 'skopt' and SKOPT_AVAILABLE:
            return self._optimize_with_skopt(
                disrupted_trips, n_trials, objective, parameter_bounds
            )
        else:
            print("No BO backend available. Using random search fallback.")
            return self._random_search(
                disrupted_trips, n_trials, objective, parameter_bounds
            )
    
    def optimize_multi_objective(self,
                                disrupted_trips: List[Dict],
                                n_trials: int = 100,
                                parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> List[ParetoPoint]:
        """
        Run multi-objective Bayesian Optimization to find Pareto frontier.
        
        Args:
            disrupted_trips: Trips to optimize
            n_trials: Number of BO trials
            parameter_bounds: Optional custom bounds for parameters
            
        Returns:
            List of Pareto-optimal points
        """
        self.current_disrupted_trips = disrupted_trips
        
        if parameter_bounds is None:
            parameter_bounds = {
                'cost_weight': (0.1, 0.7),
                'service_weight': (0.1, 0.7),
                'compliance_weight': (0.1, 0.7)
            }
        
        if OPTUNA_AVAILABLE:
            return self._optimize_multi_objective_optuna(
                disrupted_trips, n_trials, parameter_bounds
            )
        else:
            print("Multi-objective optimization requires Optuna. Using weighted single-objective.")
            return self._weighted_pareto_search(
                disrupted_trips, n_trials, parameter_bounds
            )
    
    def _optimize_with_optuna(self,
                             disrupted_trips: List[Dict],
                             n_trials: int,
                             objective: str,
                             parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict:
        """
        Optimize using Optuna.
        """
        import optuna
        
        def objective_function(trial):
            # Sample parameters
            params = {}
            for param_name, (low, high) in parameter_bounds.items():
                if 'weight' in param_name:
                    params[param_name] = trial.suggest_float(param_name, low, high)
                elif param_name == 'max_cascade_depth':
                    params[param_name] = trial.suggest_int(param_name, int(low), int(high))
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)
            
            # Normalize weights
            weight_keys = [k for k in params if 'weight' in k]
            if weight_keys:
                total_weight = sum(params[k] for k in weight_keys)
                for k in weight_keys:
                    params[k] /= total_weight
            
            # Evaluate with CP-SAT
            result = self._evaluate_parameters(params, disrupted_trips)
            
            # Store trial result
            self.trial_results.append(result)
            
            # Return objective to minimize
            if objective == 'combined':
                return result.combined_objective
            elif objective == 'cost':
                return result.cost_score
            elif objective == 'service':
                return 1 - result.service_score  # Maximize service
            else:  # compliance
                return 1 - result.compliance_score  # Maximize compliance
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            study_name=f"trip_rescheduling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Optimize
        study.optimize(objective_function, n_trials=n_trials)
        
        # Get best parameters
        self.best_parameters = study.best_params
        
        # Save results
        self._save_results(study)
        
        return self.best_parameters
    
    def _optimize_multi_objective_optuna(self,
                                        disrupted_trips: List[Dict],
                                        n_trials: int,
                                        parameter_bounds: Dict[str, Tuple[float, float]]) -> List[ParetoPoint]:
        """
        Multi-objective optimization with Optuna.
        """
        import optuna
        
        def objective_function(trial):
            # Sample parameters
            params = {}
            for param_name, (low, high) in parameter_bounds.items():
                params[param_name] = trial.suggest_float(param_name, low, high)
            
            # Normalize weights
            total_weight = sum(params.values())
            for k in params:
                params[k] /= total_weight
            
            # Evaluate with CP-SAT
            result = self._evaluate_parameters(params, disrupted_trips)
            
            # Store trial result
            self.trial_results.append(result)
            
            # Return multiple objectives (cost, service, compliance)
            return result.cost_score, 1 - result.service_score, 1 - result.compliance_score
        
        # Create multi-objective study
        study = optuna.create_study(
            directions=['minimize', 'minimize', 'minimize'],
            study_name=f"multi_objective_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Optimize
        study.optimize(objective_function, n_trials=n_trials)
        
        # Extract Pareto frontier
        self.pareto_frontier = self._extract_pareto_frontier(study)
        
        # Save results
        self._save_results(study)
        
        return self.pareto_frontier
    
    def _evaluate_parameters(self,
                            params: Dict[str, float],
                            disrupted_trips: List[Dict]) -> BOTrialResult:
        """
        Evaluate a parameter configuration using CP-SAT.
        """
        self.trial_counter += 1
        
        # Extract weight parameters
        objective_weights = {
            'cost_weight': params.get('cost_weight', 0.4),
            'service_weight': params.get('service_weight', 0.3),
            'compliance_weight': params.get('compliance_weight', 0.3)
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
        if solution.metrics:
            metrics = solution.metrics
            result = BOTrialResult(
                trial_id=self.trial_counter,
                timestamp=datetime.now(),
                parameters=params,
                cost_score=metrics.total_cost_score,
                service_score=metrics.service_quality_score,
                compliance_score=metrics.compliance_score,
                combined_objective=metrics.combined_objective,
                feasibility_rate=metrics.operational.feasibility_rate,
                total_cost=metrics.cost.total_cost,
                on_time_rate=metrics.sla.on_time_rate,
                violations=sum(metrics.compliance.violations.values()),
                solve_time_seconds=solution.solve_time_seconds
            )
        else:
            # Infeasible solution
            result = BOTrialResult(
                trial_id=self.trial_counter,
                timestamp=datetime.now(),
                parameters=params,
                cost_score=1.0,
                service_score=0.0,
                compliance_score=0.0,
                combined_objective=1.0,
                feasibility_rate=0.0,
                total_cost=float('inf'),
                on_time_rate=0.0,
                violations=999,
                solve_time_seconds=solution.solve_time_seconds
            )
        
        # Print progress
        print(f"\nTrial {self.trial_counter}:")
        print(f"  Parameters: {params}")
        print(f"  Combined objective: {result.combined_objective:.4f}")
        print(f"  Feasibility rate: {result.feasibility_rate:.2%}")
        
        return result
    
    def _extract_pareto_frontier(self, study) -> List[ParetoPoint]:
        """
        Extract Pareto-optimal points from study.
        """
        pareto_points = []
        
        if hasattr(study, 'best_trials'):
            # Optuna multi-objective study
            for trial in study.best_trials:
                point = ParetoPoint(
                    parameters=trial.params,
                    objectives={
                        'cost': trial.values[0],
                        'service': 1 - trial.values[1],  # Convert back to maximization
                        'compliance': 1 - trial.values[2]
                    }
                )
                pareto_points.append(point)
        
        return pareto_points
    
    def _random_search(self,
                      disrupted_trips: List[Dict],
                      n_trials: int,
                      objective: str,
                      parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict:
        """
        Fallback random search when BO libraries not available.
        """
        best_objective = float('inf')
        best_params = None
        
        for i in range(n_trials):
            # Sample random parameters
            params = {}
            for param_name, (low, high) in parameter_bounds.items():
                if 'max_cascade_depth' in param_name:
                    params[param_name] = np.random.randint(low, high + 1)
                else:
                    params[param_name] = np.random.uniform(low, high)
            
            # Normalize weights
            weight_keys = [k for k in params if 'weight' in k]
            if weight_keys:
                total_weight = sum(params[k] for k in weight_keys)
                for k in weight_keys:
                    params[k] /= total_weight
            
            # Evaluate
            result = self._evaluate_parameters(params, disrupted_trips)
            
            # Update best
            if result.combined_objective < best_objective:
                best_objective = result.combined_objective
                best_params = params
                print(f"  New best: {best_objective:.4f}")
        
        self.best_parameters = best_params
        return best_params
    
    def _weighted_pareto_search(self,
                               disrupted_trips: List[Dict],
                               n_trials: int,
                               parameter_bounds: Dict[str, Tuple[float, float]]) -> List[ParetoPoint]:
        """
        Find Pareto frontier using weighted sum method.
        """
        pareto_points = []
        
        # Generate weight combinations
        n_weights = int(np.sqrt(n_trials))
        cost_weights = np.linspace(0.1, 0.8, n_weights)
        
        for cost_weight in cost_weights:
            remaining = 1.0 - cost_weight
            service_weight = np.random.uniform(0.1, remaining - 0.1)
            compliance_weight = remaining - service_weight
            
            params = {
                'cost_weight': cost_weight,
                'service_weight': service_weight,
                'compliance_weight': compliance_weight
            }
            
            # Evaluate
            result = self._evaluate_parameters(params, disrupted_trips)
            
            point = ParetoPoint(
                parameters=params,
                objectives={
                    'cost': result.cost_score,
                    'service': result.service_score,
                    'compliance': result.compliance_score
                }
            )
            pareto_points.append(point)
        
        # Filter for Pareto optimality
        self.pareto_frontier = self._filter_pareto_optimal(pareto_points)
        return self.pareto_frontier
    
    def _filter_pareto_optimal(self, points: List[ParetoPoint]) -> List[ParetoPoint]:
        """
        Filter points to keep only Pareto-optimal ones.
        """
        n_points = len(points)
        
        for i in range(n_points):
            for j in range(n_points):
                if i == j:
                    continue
                
                # Check if point j dominates point i
                dominates = all(
                    points[j].objectives[obj] <= points[i].objectives[obj]
                    for obj in ['cost', 'service', 'compliance']
                ) and any(
                    points[j].objectives[obj] < points[i].objectives[obj]
                    for obj in ['cost', 'service', 'compliance']
                )
                
                if dominates:
                    points[i].dominated_by.append(j)
                    points[j].dominates.append(i)
        
        # Return only non-dominated points
        return [p for p in points if p.is_pareto_optimal()]
    
    def _save_results(self, study=None):
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
        
        # Save Pareto frontier if multi-objective
        if self.pareto_frontier:
            pareto_data = []
            for point in self.pareto_frontier:
                pareto_data.append({
                    **point.parameters,
                    **{f"obj_{k}": v for k, v in point.objectives.items()}
                })
            pareto_df = pd.DataFrame(pareto_data)
            pareto_df.to_csv(self.results_dir / f"pareto_{timestamp}.csv", index=False)
        
        # Save Optuna study if available
        if study and OPTUNA_AVAILABLE:
            import optuna
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
        objectives = [r.combined_objective for r in self.trial_results]
        feasibility = [r.feasibility_rate for r in self.trial_results]
        costs = [r.total_cost for r in self.trial_results]
        on_time = [r.on_time_rate for r in self.trial_results]
        
        # Plot 1: Objective over trials
        axes[0, 0].plot(trials, objectives, 'b-', alpha=0.6)
        axes[0, 0].scatter(trials, objectives, c=feasibility, cmap='RdYlGn', s=30)
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('Combined Objective')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cost vs Service tradeoff
        axes[0, 1].scatter([r.cost_score for r in self.trial_results],
                          [r.service_score for r in self.trial_results],
                          c=trials, cmap='viridis', s=50)
        axes[0, 1].set_xlabel('Cost Score')
        axes[0, 1].set_ylabel('Service Score')
        axes[0, 1].set_title('Cost vs Service Tradeoff')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Feasibility rate
        axes[1, 0].plot(trials, feasibility, 'g-', linewidth=2)
        axes[1, 0].fill_between(trials, 0, feasibility, alpha=0.3, color='green')
        axes[1, 0].set_xlabel('Trial')
        axes[1, 0].set_ylabel('Feasibility Rate')
        axes[1, 0].set_title('Solution Feasibility')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Parameter evolution (for weights)
        weight_params = ['cost_weight', 'service_weight', 'compliance_weight']
        for param in weight_params:
            values = [r.parameters.get(param, 0) for r in self.trial_results]
            axes[1, 1].plot(trials, values, label=param, linewidth=2)
        axes[1, 1].set_xlabel('Trial')
        axes[1, 1].set_ylabel('Weight Value')
        axes[1, 1].set_title('Parameter Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
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
        
        # Find best configurations for different objectives
        best_cost = min(self.trial_results, key=lambda x: x.cost_score)
        best_service = max(self.trial_results, key=lambda x: x.service_score)
        best_compliance = max(self.trial_results, key=lambda x: x.compliance_score)
        best_combined = min(self.trial_results, key=lambda x: x.combined_objective)
        
        recommendations = {
            "best_overall": {
                "parameters": best_combined.parameters,
                "metrics": {
                    "combined_objective": best_combined.combined_objective,
                    "feasibility_rate": best_combined.feasibility_rate,
                    "total_cost": best_combined.total_cost,
                    "on_time_rate": best_combined.on_time_rate
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
            "best_for_compliance": {
                "parameters": best_compliance.parameters,
                "compliance_score": best_compliance.compliance_score,
                "violations": best_compliance.violations
            },
            "pareto_optimal_count": len(self.pareto_frontier) if self.pareto_frontier else 0,
            "total_trials": len(self.trial_results)
        }
        
        return recommendations


# Example usage
def example_bayesian_optimization():
    """
    Example of running Bayesian Optimization.
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
        optimization_backend='optuna' if OPTUNA_AVAILABLE else 'skopt',
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
    print("="*60)
    
    # Run single-objective optimization
    print("\n📊 Running single-objective optimization...")
    best_params = bo_tuner.optimize_single_objective(
        disrupted_trips,
        n_trials=20,
        objective='combined'
    )
    
    print(f"\n✅ Best parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value:.3f}")
    
    # Get recommendations
    recommendations = bo_tuner.get_recommendations()
    print(f"\n📋 Recommendations:")
    print(f"  Best feasibility rate: {recommendations['best_overall']['metrics']['feasibility_rate']:.2%}")
    print(f"  Best total cost: ${recommendations['best_overall']['metrics']['total_cost']:.2f}")
    print(f"  Best on-time rate: {recommendations['best_overall']['metrics']['on_time_rate']:.2%}")
    
    # Plot results
    bo_tuner.plot_optimization_progress()
    
    return bo_tuner


if __name__ == "__main__":
    # Run example
    tuner = example_bayesian_optimization()