#!/usr/bin/env python3
"""
Automated Hyperparameter Optimization for CME Prediction System
Implements advanced optimization algorithms using scikit-optimize
"""

import numpy as np
import pandas as pd
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import Real, Integer, Categorical
from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
from skopt.utils import use_named_args
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from datetime import datetime
import logging
from dataclasses import dataclass
import pickle
import concurrent.futures
from tqdm import tqdm


@dataclass
class OptimizationResult:
    """Data class for optimization results"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    total_evaluations: int
    total_time: float


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization for neural networks and data processing"""
    
    def __init__(self, n_jobs: int = 1, random_state: int = 42):
        """
        Initialize hyperparameter optimizer
        
        Args:
            n_jobs: Number of parallel jobs for optimization
            random_state: Random state for reproducibility
        """
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.optimization_history = []
        
    def define_search_space(self, optimization_target: str = 'neural_network') -> List:
        """
        Define search space for different optimization targets
        
        Args:
            optimization_target: Target to optimize ('neural_network', 'feature_engineering', 'data_filtering')
            
        Returns:
            List of optimization dimensions
        """
        if optimization_target == 'neural_network':
            return [
                Integer(10, 200, name='hidden_layer_1_size'),
                Integer(5, 100, name='hidden_layer_2_size'),
                Integer(2, 50, name='hidden_layer_3_size'),
                Real(0.0001, 0.1, prior='log-uniform', name='learning_rate'),
                Real(0.0, 0.5, name='dropout_rate'),
                Real(0.8, 0.99, name='momentum'),
                Integer(50, 1000, name='epochs'),
                Integer(16, 128, name='batch_size'),
                Real(1e-6, 1e-2, prior='log-uniform', name='weight_decay'),
                Categorical(['relu', 'tanh', 'sigmoid', 'leaky_relu'], name='activation_function'),
                Categorical(['adam', 'sgd', 'rmsprop'], name='optimizer'),
                Real(0.1, 10.0, name='gradient_clip_threshold'),
                Categorical([True, False], name='batch_normalization'),
                Real(0.7, 0.95, name='train_test_split')
            ]
            
        elif optimization_target == 'feature_engineering':
            return [
                Integer(3, 50, name='rolling_window_size'),
                Integer(1, 10, name='polynomial_degree'),
                Real(0.01, 1.0, name='correlation_threshold'),
                Integer(2, 20, name='fourier_components'),
                Real(0.1, 5.0, name='outlier_threshold'),
                Integer(5, 100, name='statistical_window'),
                Categorical([True, False], name='use_gradients'),
                Categorical([True, False], name='use_interactions'),
                Categorical([True, False], name='use_spectral_features'),
                Real(0.1, 2.0, name='smoothing_factor'),
                Integer(1, 5, name='differencing_order'),
                Real(0.0, 0.5, name='noise_threshold')
            ]
            
        elif optimization_target == 'data_filtering':
            return [
                Integer(3, 31, name='savgol_window'),
                Integer(1, 5, name='savgol_polyorder'),
                Real(0.5, 5.0, name='gaussian_sigma'),
                Integer(3, 21, name='median_filter_size'),
                Real(0.01, 0.5, name='butterworth_cutoff'),
                Integer(1, 6, name='butterworth_order'),
                Real(0.1, 0.9, name='adaptive_alpha'),
                Real(0.05, 0.5, name='adaptive_beta'),
                Real(1.0, 5.0, name='robust_threshold'),
                Real(0.5, 0.99, name='quality_threshold'),
                Categorical(['linear', 'cubic', 'spline'], name='interpolation_method'),
                Real(0.01, 0.2, name='anomaly_contamination')
            ]
        else:
            raise ValueError("optimization_target must be 'neural_network', 'feature_engineering', or 'data_filtering'")
    
    def optimize_neural_network(self, 
                               objective_function: Callable,
                               n_calls: int = 100,
                               optimizer: str = 'gp',
                               acquisition_func: str = 'EI') -> OptimizationResult:
        """
        Optimize neural network hyperparameters
        
        Args:
            objective_function: Function to minimize (should return validation loss)
            n_calls: Number of optimization calls
            optimizer: Optimizer type ('gp', 'forest', 'gbrt')
            acquisition_func: Acquisition function ('EI', 'PI', 'LCB')
            
        Returns:
            OptimizationResult with best parameters and optimization history
        """
        self.logger.info(f"Starting neural network hyperparameter optimization with {n_calls} calls")
        
        dimensions = self.define_search_space('neural_network')
        start_time = time.time()
        
        # Convert acquisition function
        acquisition_functions = {
            'EI': gaussian_ei,
            'PI': gaussian_pi,  
            'LCB': gaussian_lcb
        }
        acq_func = acquisition_functions.get(acquisition_func, gaussian_ei)
        
        # Wrapper function to track optimization history
        @use_named_args(dimensions)
        def objective_wrapper(**params):
            try:
                # Validate parameters
                params = self._validate_neural_network_params(params)
                
                # Evaluate objective function
                score = objective_function(params)
                
                # Track history
                evaluation = {
                    'parameters': params.copy(),
                    'score': float(score),
                    'timestamp': datetime.now().isoformat(),
                    'evaluation_number': len(self.optimization_history) + 1
                }
                self.optimization_history.append(evaluation)
                
                self.logger.debug(f"Evaluation {len(self.optimization_history)}: Score = {score:.6f}")
                return score
                
            except Exception as e:
                self.logger.error(f"Error in objective function: {e}")
                return 1.0  # Return high loss for failed evaluations
        
        # Choose optimization algorithm
        if optimizer == 'gp':
            result = gp_minimize(
                func=objective_wrapper,
                dimensions=dimensions,
                n_calls=n_calls,
                n_initial_points=min(20, n_calls//4),
                acquisition_func=acq_func,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        elif optimizer == 'forest':
            result = forest_minimize(
                func=objective_wrapper,
                dimensions=dimensions,
                n_calls=n_calls,
                n_initial_points=min(20, n_calls//4),
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        elif optimizer == 'gbrt':
            result = gbrt_minimize(
                func=objective_wrapper,
                dimensions=dimensions,
                n_calls=n_calls,
                n_initial_points=min(20, n_calls//4),
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        else:
            raise ValueError("optimizer must be 'gp', 'forest', or 'gbrt'")
        
        # Extract best parameters
        best_params = {}
        for i, dim in enumerate(dimensions):
            best_params[dim.name] = result.x[i]
        
        # Validate best parameters
        best_params = self._validate_neural_network_params(best_params)
        
        total_time = time.time() - start_time
        
        # Analyze convergence
        convergence_info = self._analyze_convergence(self.optimization_history)
        
        optimization_result = OptimizationResult(
            best_params=best_params,
            best_score=float(result.fun),
            optimization_history=self.optimization_history.copy(),
            convergence_info=convergence_info,
            total_evaluations=len(result.func_vals),
            total_time=total_time
        )
        
        self.logger.info(f"Neural network optimization completed in {total_time:.2f}s")
        self.logger.info(f"Best score: {result.fun:.6f}")
        
        return optimization_result
    
    def _validate_neural_network_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and adjust neural network parameters"""
        validated_params = params.copy()
        
        # Ensure layer sizes make sense
        if validated_params['hidden_layer_2_size'] > validated_params['hidden_layer_1_size']:
            validated_params['hidden_layer_2_size'] = validated_params['hidden_layer_1_size'] // 2
        
        if validated_params['hidden_layer_3_size'] > validated_params['hidden_layer_2_size']:
            validated_params['hidden_layer_3_size'] = validated_params['hidden_layer_2_size'] // 2
        
        # Ensure batch size is reasonable for epochs
        if validated_params['batch_size'] > 64 and validated_params['epochs'] < 100:
            validated_params['epochs'] = max(100, validated_params['epochs'])
        
        # Adjust learning rate based on optimizer
        if validated_params['optimizer'] == 'sgd':
            validated_params['learning_rate'] = max(validated_params['learning_rate'], 0.001)
        
        return validated_params
    
    def optimize_feature_engineering(self,
                                   objective_function: Callable,
                                   n_calls: int = 50) -> OptimizationResult:
        """
        Optimize feature engineering parameters
        
        Args:
            objective_function: Function to minimize
            n_calls: Number of optimization calls
            
        Returns:
            OptimizationResult with best feature engineering parameters
        """
        self.logger.info(f"Starting feature engineering optimization with {n_calls} calls")
        
        dimensions = self.define_search_space('feature_engineering')
        start_time = time.time()
        
        @use_named_args(dimensions)
        def objective_wrapper(**params):
            try:
                score = objective_function(params)
                
                evaluation = {
                    'parameters': params.copy(),
                    'score': float(score),
                    'timestamp': datetime.now().isoformat(),
                    'evaluation_number': len(self.optimization_history) + 1
                }
                self.optimization_history.append(evaluation)
                
                return score
                
            except Exception as e:
                self.logger.error(f"Error in feature engineering objective: {e}")
                return 1.0
        
        result = gp_minimize(
            func=objective_wrapper,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        best_params = {}
        for i, dim in enumerate(dimensions):
            best_params[dim.name] = result.x[i]
        
        total_time = time.time() - start_time
        convergence_info = self._analyze_convergence(self.optimization_history)
        
        return OptimizationResult(
            best_params=best_params,
            best_score=float(result.fun),
            optimization_history=self.optimization_history.copy(),
            convergence_info=convergence_info,
            total_evaluations=len(result.func_vals),
            total_time=total_time
        )
    
    def multi_objective_optimization(self,
                                   objective_functions: Dict[str, Callable],
                                   weights: Dict[str, float] = None,
                                   n_calls: int = 100) -> OptimizationResult:
        """
        Multi-objective optimization with weighted objectives
        
        Args:
            objective_functions: Dictionary of objective functions
            weights: Weights for each objective
            n_calls: Number of optimization calls
            
        Returns:
            OptimizationResult for multi-objective optimization
        """
        if weights is None:
            weights = {name: 1.0/len(objective_functions) for name in objective_functions.keys()}
        
        dimensions = self.define_search_space('neural_network')
        
        @use_named_args(dimensions)
        def multi_objective_wrapper(**params):
            try:
                total_score = 0.0
                individual_scores = {}
                
                for name, obj_func in objective_functions.items():
                    score = obj_func(params)
                    individual_scores[name] = score
                    total_score += weights[name] * score
                
                evaluation = {
                    'parameters': params.copy(),
                    'total_score': float(total_score),
                    'individual_scores': individual_scores,
                    'timestamp': datetime.now().isoformat(),
                    'evaluation_number': len(self.optimization_history) + 1
                }
                self.optimization_history.append(evaluation)
                
                return total_score
                
            except Exception as e:
                self.logger.error(f"Error in multi-objective function: {e}")
                return 1.0
        
        start_time = time.time()
        
        result = gp_minimize(
            func=multi_objective_wrapper,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        best_params = {}
        for i, dim in enumerate(dimensions):
            best_params[dim.name] = result.x[i]
        
        total_time = time.time() - start_time
        convergence_info = self._analyze_convergence(self.optimization_history)
        
        return OptimizationResult(
            best_params=best_params,
            best_score=float(result.fun),
            optimization_history=self.optimization_history.copy(),
            convergence_info=convergence_info,
            total_evaluations=len(result.func_vals),
            total_time=total_time
        )
    
    def bayesian_optimization_with_constraints(self,
                                             objective_function: Callable,
                                             constraint_functions: List[Callable],
                                             n_calls: int = 100) -> OptimizationResult:
        """
        Bayesian optimization with constraints
        
        Args:
            objective_function: Main objective to minimize
            constraint_functions: List of constraint functions (should return <= 0)
            n_calls: Number of optimization calls
            
        Returns:
            OptimizationResult for constrained optimization
        """
        dimensions = self.define_search_space('neural_network')
        
        @use_named_args(dimensions)
        def constrained_objective(**params):
            try:
                # Check constraints
                for i, constraint_func in enumerate(constraint_functions):
                    constraint_value = constraint_func(params)
                    if constraint_value > 0:  # Constraint violated
                        penalty = 1.0 + 10.0 * constraint_value
                        return penalty
                
                # Evaluate objective if constraints are satisfied
                score = objective_function(params)
                
                evaluation = {
                    'parameters': params.copy(),
                    'score': float(score),
                    'constraints_satisfied': True,
                    'timestamp': datetime.now().isoformat(),
                    'evaluation_number': len(self.optimization_history) + 1
                }
                self.optimization_history.append(evaluation)
                
                return score
                
            except Exception as e:
                self.logger.error(f"Error in constrained objective: {e}")
                return 1.0
        
        start_time = time.time()
        
        result = gp_minimize(
            func=constrained_objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        best_params = {}
        for i, dim in enumerate(dimensions):
            best_params[dim.name] = result.x[i]
        
        total_time = time.time() - start_time
        convergence_info = self._analyze_convergence(self.optimization_history)
        
        return OptimizationResult(
            best_params=best_params,
            best_score=float(result.fun),
            optimization_history=self.optimization_history.copy(),
            convergence_info=convergence_info,
            total_evaluations=len(result.func_vals),
            total_time=total_time
        )
    
    def adaptive_optimization(self,
                            objective_function: Callable,
                            initial_n_calls: int = 50,
                            max_n_calls: int = 200,
                            improvement_threshold: float = 0.01,
                            patience: int = 10) -> OptimizationResult:
        """
        Adaptive optimization that stops early if no improvement
        
        Args:
            objective_function: Function to minimize
            initial_n_calls: Initial number of calls
            max_n_calls: Maximum number of calls
            improvement_threshold: Minimum improvement threshold
            patience: Number of calls without improvement before stopping
            
        Returns:
            OptimizationResult for adaptive optimization
        """
        dimensions = self.define_search_space('neural_network')
        best_score = float('inf')
        no_improvement_count = 0
        call_count = 0
        
        @use_named_args(dimensions)
        def adaptive_objective(**params):
            nonlocal best_score, no_improvement_count, call_count
            
            try:
                score = objective_function(params)
                call_count += 1
                
                if score < best_score - improvement_threshold:
                    best_score = score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                evaluation = {
                    'parameters': params.copy(),
                    'score': float(score),
                    'best_score_so_far': float(best_score),
                    'no_improvement_count': no_improvement_count,
                    'timestamp': datetime.now().isoformat(),
                    'evaluation_number': len(self.optimization_history) + 1
                }
                self.optimization_history.append(evaluation)
                
                # Early stopping condition
                if call_count >= initial_n_calls and no_improvement_count >= patience:
                    self.logger.info(f"Early stopping at call {call_count} due to no improvement")
                
                return score
                
            except Exception as e:
                self.logger.error(f"Error in adaptive objective: {e}")
                return 1.0
        
        start_time = time.time()
        
        # Start with smaller number of calls, can be extended if needed
        current_n_calls = min(initial_n_calls, max_n_calls)
        
        result = gp_minimize(
            func=adaptive_objective,
            dimensions=dimensions,
            n_calls=current_n_calls,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Continue optimization if we haven't hit early stopping and haven't reached max calls
        while call_count < max_n_calls and no_improvement_count < patience:
            additional_calls = min(20, max_n_calls - call_count)
            
            additional_result = gp_minimize(
                func=adaptive_objective,
                dimensions=dimensions,
                n_calls=additional_calls,
                x0=result.x_iters,
                y0=result.func_vals,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            
            # Update result with additional calls
            result.x_iters.extend(additional_result.x_iters)
            result.func_vals.extend(additional_result.func_vals)
            
            if additional_result.fun < result.fun:
                result.x = additional_result.x
                result.fun = additional_result.fun
        
        best_params = {}
        for i, dim in enumerate(dimensions):
            best_params[dim.name] = result.x[i]
        
        total_time = time.time() - start_time
        convergence_info = self._analyze_convergence(self.optimization_history)
        convergence_info['early_stopped'] = call_count < max_n_calls
        convergence_info['total_calls'] = call_count
        
        return OptimizationResult(
            best_params=best_params,
            best_score=float(result.fun),
            optimization_history=self.optimization_history.copy(),
            convergence_info=convergence_info,
            total_evaluations=call_count,
            total_time=total_time
        )
    
    def _analyze_convergence(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze optimization convergence"""
        if not history:
            return {}
        
        scores = [eval_data['score'] if 'score' in eval_data else eval_data.get('total_score', 1.0) 
                 for eval_data in history]
        
        # Calculate best score progression
        best_scores = []
        current_best = float('inf')
        for score in scores:
            if score < current_best:
                current_best = score
            best_scores.append(current_best)
        
        # Calculate improvement rate
        improvements = []
        for i in range(1, len(best_scores)):
            if best_scores[i-1] > 0:
                improvement = (best_scores[i-1] - best_scores[i]) / best_scores[i-1]
                improvements.append(improvement)
        
        # Convergence metrics
        convergence_info = {
            'total_evaluations': len(scores),
            'best_score': float(min(scores)),
            'worst_score': float(max(scores)),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'total_improvement': float(scores[0] - min(scores)) if len(scores) > 0 else 0.0,
            'final_improvement_rate': float(improvements[-10:]) if len(improvements) >= 10 else 0.0,
            'convergence_detected': len(improvements) > 10 and np.mean(improvements[-10:]) < 0.001,
            'best_scores_progression': best_scores,
            'evaluation_timestamps': [eval_data['timestamp'] for eval_data in history]
        }
        
        return convergence_info
    
    def save_optimization_results(self, 
                                result: OptimizationResult, 
                                filepath: str):
        """Save optimization results to file"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(result, f)
            self.logger.info(f"Optimization results saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save optimization results: {e}")
    
    def load_optimization_results(self, filepath: str) -> OptimizationResult:
        """Load optimization results from file"""
        try:
            with open(filepath, 'rb') as f:
                result = pickle.load(f)
            self.logger.info(f"Optimization results loaded from {filepath}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to load optimization results: {e}")
            raise
    
    def compare_optimization_results(self, 
                                   results: List[OptimizationResult]) -> Dict[str, Any]:
        """Compare multiple optimization results"""
        comparison = {
            'summary': {
                'total_results': len(results),
                'best_overall_score': min(r.best_score for r in results),
                'worst_overall_score': max(r.best_score for r in results),
                'mean_best_score': np.mean([r.best_score for r in results]),
                'std_best_score': np.std([r.best_score for r in results])
            },
            'individual_results': []
        }
        
        for i, result in enumerate(results):
            comparison['individual_results'].append({
                'result_index': i,
                'best_score': result.best_score,
                'total_evaluations': result.total_evaluations,
                'total_time': result.total_time,
                'convergence_detected': result.convergence_info.get('convergence_detected', False),
                'total_improvement': result.convergence_info.get('total_improvement', 0.0),
                'best_parameters': result.best_params
            })
        
        # Find best result
        best_result_idx = np.argmin([r.best_score for r in results])
        comparison['best_result_index'] = int(best_result_idx)
        comparison['best_result'] = comparison['individual_results'][best_result_idx]
        
        return comparison