#!/usr/bin/env python3
"""
Advanced Batch Processing System for CME Prediction
Handles large-scale data processing with parallel execution and progress tracking
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import queue
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable, Iterator, Union
import json
import logging
from dataclasses import dataclass, asdict
from tqdm import tqdm
import psutil
import os


@dataclass
class BatchJob:
    """Data class for batch job configuration"""
    job_id: str
    job_type: str
    input_data: Any
    parameters: Dict[str, Any]
    priority: int = 1
    max_workers: int = None
    chunk_size: int = 1000
    timeout: Optional[float] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = 'pending'
    progress: float = 0.0
    results: Any = None
    error_message: str = None


@dataclass
class BatchResult:
    """Data class for batch processing results"""
    job_id: str
    status: str
    results: Any
    processing_time: float
    items_processed: int
    errors_encountered: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]


class BatchProcessor:
    """Advanced batch processing system with parallel execution and monitoring"""
    
    def __init__(self, max_workers: int = None, max_memory_gb: float = 4.0):
        """
        Initialize batch processor
        
        Args:
            max_workers: Maximum number of worker threads/processes
            max_memory_gb: Maximum memory usage in GB
        """
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.max_memory_gb = max_memory_gb
        self.job_queue = queue.PriorityQueue()
        self.active_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: Dict[str, BatchResult] = {}
        self.logger = logging.getLogger(__name__)
        
        self._shutdown_event = threading.Event()
        self._worker_threads = []
        self._start_worker_threads()
    
    def _start_worker_threads(self):
        """Start worker threads for batch processing"""
        for i in range(min(4, self.max_workers)):  # Start with 4 dispatcher threads
            worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"BatchWorker-{i}",
                daemon=True
            )
            worker_thread.start()
            self._worker_threads.append(worker_thread)
        
        self.logger.info(f"Started {len(self._worker_threads)} batch worker threads")
    
    def _worker_loop(self):
        """Main worker loop for processing batch jobs"""
        while not self._shutdown_event.is_set():
            try:
                # Get job from queue with timeout
                priority, job = self.job_queue.get(timeout=1.0)
                
                if job.status == 'pending':
                    job.status = 'running'
                    job.started_at = datetime.now()
                    self.active_jobs[job.job_id] = job
                    
                    self.logger.info(f"Starting batch job {job.job_id} ({job.job_type})")
                    
                    try:
                        # Process the job
                        result = self._process_single_job(job)
                        
                        # Mark as completed
                        job.status = 'completed'
                        job.completed_at = datetime.now()
                        job.results = result.results
                        job.progress = 100.0
                        
                        self.completed_jobs[job.job_id] = result
                        
                        if job.job_id in self.active_jobs:
                            del self.active_jobs[job.job_id]
                        
                        self.logger.info(f"Completed batch job {job.job_id}")
                        
                    except Exception as e:
                        job.status = 'failed'
                        job.error_message = str(e)
                        job.completed_at = datetime.now()
                        
                        if job.job_id in self.active_jobs:
                            del self.active_jobs[job.job_id]
                        
                        self.logger.error(f"Batch job {job.job_id} failed: {e}")
                
                self.job_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
    
    def submit_job(self, 
                   job_type: str,
                   input_data: Any,
                   parameters: Dict[str, Any] = None,
                   priority: int = 1,
                   max_workers: int = None,
                   chunk_size: int = 1000) -> str:
        """
        Submit a batch job for processing
        
        Args:
            job_type: Type of batch job
            input_data: Input data for processing
            parameters: Job parameters
            priority: Job priority (lower values = higher priority)
            max_workers: Maximum workers for this job
            chunk_size: Chunk size for data processing
            
        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())
        
        job = BatchJob(
            job_id=job_id,
            job_type=job_type,
            input_data=input_data,
            parameters=parameters or {},
            priority=priority,
            max_workers=max_workers or self.max_workers,
            chunk_size=chunk_size,
            created_at=datetime.now(),
            status='pending'
        )
        
        # Add to priority queue (lower priority values are processed first)
        self.job_queue.put((priority, job))
        
        self.logger.info(f"Submitted batch job {job_id} ({job_type}) with priority {priority}")
        return job_id
    
    def _process_single_job(self, job: BatchJob) -> BatchResult:
        """Process a single batch job"""
        start_time = time.time()
        errors = []
        performance_metrics = {}
        
        try:
            if job.job_type == 'data_preprocessing':
                results = self._process_data_preprocessing(job, errors)
            elif job.job_type == 'feature_engineering':
                results = self._process_feature_engineering(job, errors)
            elif job.job_type == 'model_training':
                results = self._process_model_training(job, errors)
            elif job.job_type == 'prediction_batch':
                results = self._process_prediction_batch(job, errors)
            elif job.job_type == 'statistical_analysis':
                results = self._process_statistical_analysis(job, errors)
            elif job.job_type == 'hyperparameter_optimization':
                results = self._process_hyperparameter_optimization(job, errors)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
            
            processing_time = time.time() - start_time
            
            # Calculate performance metrics
            performance_metrics = {
                'processing_time_seconds': processing_time,
                'items_per_second': len(job.input_data) / processing_time if processing_time > 0 else 0,
                'memory_peak_mb': self._get_memory_usage(),
                'cpu_cores_used': job.max_workers,
                'errors_count': len(errors),
                'success_rate': 1.0 - (len(errors) / max(1, len(job.input_data))) if hasattr(job.input_data, '__len__') else 1.0
            }
            
            return BatchResult(
                job_id=job.job_id,
                status='completed',
                results=results,
                processing_time=processing_time,
                items_processed=len(job.input_data) if hasattr(job.input_data, '__len__') else 1,
                errors_encountered=errors,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return BatchResult(
                job_id=job.job_id,
                status='failed',
                results=None,
                processing_time=processing_time,
                items_processed=0,
                errors_encountered=[{'error': str(e), 'timestamp': datetime.now().isoformat()}],
                performance_metrics={'processing_time_seconds': processing_time}
            )
    
    def _process_data_preprocessing(self, job: BatchJob, errors: List) -> Dict[str, Any]:
        """Process data preprocessing batch job"""
        from data_filters import AdvancedDataFilter
        
        data = job.input_data
        params = job.parameters
        filter_obj = AdvancedDataFilter()
        
        results = {
            'processed_data': {},
            'filter_performance': {},
            'data_quality_metrics': {}
        }
        
        if isinstance(data, pd.DataFrame):
            # Process each column
            total_columns = len(data.columns)
            
            with ThreadPoolExecutor(max_workers=job.max_workers) as executor:
                future_to_column = {}
                
                for column in data.select_dtypes(include=[np.number]).columns:
                    filter_params = params.get(column, {})
                    
                    future = executor.submit(
                        self._process_single_column,
                        data[column].values,
                        filter_params,
                        filter_obj
                    )
                    future_to_column[future] = column
                
                # Collect results with progress tracking
                for i, future in enumerate(as_completed(future_to_column)):
                    column = future_to_column[future]
                    job.progress = (i + 1) / total_columns * 100
                    
                    try:
                        filtered_data, performance = future.result()
                        results['processed_data'][column] = filtered_data.tolist()
                        results['filter_performance'][column] = performance
                    except Exception as e:
                        errors.append({
                            'column': column,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
        
        return results
    
    def _process_single_column(self, 
                              data: np.ndarray, 
                              filter_params: Dict[str, Any],
                              filter_obj) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single data column with filtering"""
        filter_type = filter_params.get('type', 'adaptive')
        
        if filter_type == 'adaptive':
            filtered_data = filter_obj.adaptive_filter(
                data,
                window_size=filter_params.get('window_size', 10),
                threshold_factor=filter_params.get('threshold_factor', 2.0)
            )
        elif filter_type == 'robust':
            filtered_data = filter_obj.robust_filter(
                data,
                window_size=filter_params.get('window_size', 5),
                filter_type=filter_params.get('robust_method', 'median')
            )
        elif filter_type == 'frequency':
            filtered_data = filter_obj.frequency_domain_filter(
                data,
                sampling_rate=filter_params.get('sampling_rate', 1.0),
                filter_type=filter_params.get('filter_method', 'lowpass'),
                cutoff_freq=filter_params.get('cutoff_freq', 0.1)
            )
        else:
            filtered_data = data.copy()
        
        # Calculate performance metrics
        performance = filter_obj.filter_evaluation(data, filtered_data)
        
        return filtered_data, performance
    
    def _process_feature_engineering(self, job: BatchJob, errors: List) -> Dict[str, Any]:
        """Process feature engineering batch job"""
        from feature_engineering import FeatureEngineer
        
        data = job.input_data
        params = job.parameters
        
        engineer = FeatureEngineer()
        
        # Process in chunks for large datasets
        if isinstance(data, pd.DataFrame) and len(data) > job.chunk_size:
            chunks = [data[i:i+job.chunk_size] for i in range(0, len(data), job.chunk_size)]
            total_chunks = len(chunks)
            
            processed_chunks = []
            
            with ProcessPoolExecutor(max_workers=job.max_workers) as executor:
                future_to_chunk = {
                    executor.submit(engineer.engineer_features, chunk, **params): i
                    for i, chunk in enumerate(chunks)
                }
                
                for i, future in enumerate(as_completed(future_to_chunk)):
                    job.progress = (i + 1) / total_chunks * 100
                    
                    try:
                        chunk_result = future.result()
                        processed_chunks.append(chunk_result)
                    except Exception as e:
                        chunk_idx = future_to_chunk[future]
                        errors.append({
                            'chunk_index': chunk_idx,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Combine processed chunks
            if processed_chunks:
                results = pd.concat(processed_chunks, ignore_index=True)
            else:
                results = pd.DataFrame()
        else:
            results = engineer.engineer_features(data, **params)
        
        return {
            'engineered_features': results.to_dict() if isinstance(results, pd.DataFrame) else results,
            'feature_names': list(results.columns) if isinstance(results, pd.DataFrame) else [],
            'total_features': len(results.columns) if isinstance(results, pd.DataFrame) else 0
        }
    
    def _process_model_training(self, job: BatchJob, errors: List) -> Dict[str, Any]:
        """Process model training batch job"""
        from neural_net import NeuralNetwork
        
        training_data = job.input_data
        params = job.parameters
        
        # Create neural network
        network = NeuralNetwork(
            input_size=params.get('input_size', 50),
            hidden_layers=params.get('hidden_layers', [64, 32, 16]),
            output_size=params.get('output_size', 1),
            learning_rate=params.get('learning_rate', 0.001),
            activation=params.get('activation', 'relu')
        )
        
        X_train = training_data.get('X_train')
        y_train = training_data.get('y_train')
        X_val = training_data.get('X_val')
        y_val = training_data.get('y_val')
        
        # Training with progress tracking
        epochs = params.get('epochs', 100)
        batch_size = params.get('batch_size', 32)
        
        training_history = []
        
        for epoch in range(epochs):
            # Train for one epoch
            epoch_loss = network.train_epoch(X_train, y_train, batch_size)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_predictions = network.predict(X_val)
                val_loss = network.calculate_loss(y_val, val_predictions)
                val_accuracy = network.calculate_accuracy(y_val, val_predictions)
            else:
                val_loss = None
                val_accuracy = None
            
            training_history.append({
                'epoch': epoch + 1,
                'training_loss': float(epoch_loss),
                'validation_loss': float(val_loss) if val_loss is not None else None,
                'validation_accuracy': float(val_accuracy) if val_accuracy is not None else None
            })
            
            job.progress = (epoch + 1) / epochs * 100
        
        return {
            'trained_model': network.get_weights(),
            'training_history': training_history,
            'final_training_loss': float(epoch_loss),
            'final_validation_accuracy': float(val_accuracy) if val_accuracy is not None else None,
            'model_parameters': params
        }
    
    def _process_prediction_batch(self, job: BatchJob, errors: List) -> Dict[str, Any]:
        """Process batch prediction job"""
        from neural_net import NeuralNetwork
        
        data = job.input_data
        params = job.parameters
        
        # Load pre-trained model
        model_weights = params.get('model_weights')
        model_config = params.get('model_config', {})
        
        network = NeuralNetwork(**model_config)
        if model_weights:
            network.set_weights(model_weights)
        
        # Process predictions in batches
        X = data.get('X')
        batch_size = params.get('batch_size', 1000)
        
        if len(X) > batch_size:
            # Process in chunks
            n_chunks = len(X) // batch_size + (1 if len(X) % batch_size else 0)
            predictions = []
            
            for i in range(n_chunks):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X))
                
                chunk_X = X[start_idx:end_idx]
                chunk_predictions = network.predict(chunk_X)
                predictions.extend(chunk_predictions.tolist())
                
                job.progress = (i + 1) / n_chunks * 100
        else:
            predictions = network.predict(X).tolist()
            job.progress = 100.0
        
        return {
            'predictions': predictions,
            'total_predictions': len(predictions),
            'model_version': params.get('model_version', 'v1.0'),
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def _process_statistical_analysis(self, job: BatchJob, errors: List) -> Dict[str, Any]:
        """Process statistical analysis batch job"""
        from statistical_analysis import AdvancedStatisticalAnalyzer
        
        data = job.input_data
        params = job.parameters
        analyzer = AdvancedStatisticalAnalyzer()
        
        analysis_results = {}
        analysis_types = params.get('analysis_types', ['correlation', 'distribution'])
        
        total_analyses = len(analysis_types)
        
        for i, analysis_type in enumerate(analysis_types):
            try:
                if analysis_type == 'correlation':
                    result = analyzer.correlation_analysis(
                        data, 
                        method=params.get('correlation_method', 'pearson')
                    )
                elif analysis_type == 'distribution':
                    result = analyzer.distribution_analysis(data)
                elif analysis_type == 'temporal_patterns':
                    result = analyzer.analyze_temporal_patterns(
                        data, 
                        timestamp_col=params.get('timestamp_col', 'timestamp')
                    )
                elif analysis_type == 'outlier_detection':
                    columns = data.select_dtypes(include=[np.number]).columns
                    result = {}
                    for col in columns:
                        result[col] = analyzer.outlier_detection(
                            data[col],
                            methods=params.get('outlier_methods', ['zscore', 'iqr'])
                        )
                else:
                    result = {'error': f'Unknown analysis type: {analysis_type}'}
                
                analysis_results[analysis_type] = result
                job.progress = (i + 1) / total_analyses * 100
                
            except Exception as e:
                errors.append({
                    'analysis_type': analysis_type,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                analysis_results[analysis_type] = {'error': str(e)}
        
        return analysis_results
    
    def _process_hyperparameter_optimization(self, job: BatchJob, errors: List) -> Dict[str, Any]:
        """Process hyperparameter optimization batch job"""
        from hyperparameter_optimizer import HyperparameterOptimizer
        
        params = job.parameters
        optimizer = HyperparameterOptimizer(n_jobs=job.max_workers)
        
        objective_function = params.get('objective_function')
        n_calls = params.get('n_calls', 50)
        optimization_type = params.get('optimization_type', 'neural_network')
        
        try:
            if optimization_type == 'neural_network':
                result = optimizer.optimize_neural_network(
                    objective_function=objective_function,
                    n_calls=n_calls,
                    optimizer=params.get('optimizer', 'gp')
                )
            elif optimization_type == 'feature_engineering':
                result = optimizer.optimize_feature_engineering(
                    objective_function=objective_function,
                    n_calls=n_calls
                )
            else:
                raise ValueError(f"Unknown optimization type: {optimization_type}")
            
            job.progress = 100.0
            
            return {
                'best_parameters': result.best_params,
                'best_score': result.best_score,
                'total_evaluations': result.total_evaluations,
                'optimization_time': result.total_time,
                'convergence_info': result.convergence_info
            }
            
        except Exception as e:
            errors.append({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a batch job"""
        # Check active jobs
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                'job_id': job_id,
                'status': job.status,
                'progress': job.progress,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'error_message': job.error_message
            }
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            result = self.completed_jobs[job_id]
            return {
                'job_id': job_id,
                'status': result.status,
                'progress': 100.0,
                'processing_time': result.processing_time,
                'items_processed': result.items_processed,
                'errors_count': len(result.errors_encountered),
                'performance_metrics': result.performance_metrics
            }
        
        # Job not found
        return {
            'job_id': job_id,
            'status': 'not_found',
            'error': 'Job not found'
        }
    
    def get_job_results(self, job_id: str) -> Any:
        """Get results of a completed batch job"""
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id].results
        else:
            raise ValueError(f"Job {job_id} not found or not completed")
    
    def list_jobs(self, status_filter: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """List all jobs with optional status filtering"""
        jobs = {
            'active': [],
            'completed': [],
            'total_active': len(self.active_jobs),
            'total_completed': len(self.completed_jobs)
        }
        
        # Active jobs
        for job_id, job in self.active_jobs.items():
            if status_filter is None or job.status == status_filter:
                jobs['active'].append({
                    'job_id': job_id,
                    'job_type': job.job_type,
                    'status': job.status,
                    'progress': job.progress,
                    'created_at': job.created_at.isoformat(),
                    'started_at': job.started_at.isoformat() if job.started_at else None
                })
        
        # Completed jobs
        for job_id, result in self.completed_jobs.items():
            if status_filter is None or result.status == status_filter:
                jobs['completed'].append({
                    'job_id': job_id,
                    'status': result.status,
                    'processing_time': result.processing_time,
                    'items_processed': result.items_processed,
                    'errors_count': len(result.errors_encountered)
                })
        
        return jobs
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = 'cancelled'
            job.completed_at = datetime.now()
            
            # Move to completed jobs
            result = BatchResult(
                job_id=job_id,
                status='cancelled',
                results=None,
                processing_time=0.0,
                items_processed=0,
                errors_encountered=[],
                performance_metrics={}
            )
            
            self.completed_jobs[job_id] = result
            del self.active_jobs[job_id]
            
            self.logger.info(f"Cancelled job {job_id}")
            return True
        
        return False
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'active_jobs_count': len(self.active_jobs),
            'completed_jobs_count': len(self.completed_jobs),
            'worker_threads_count': len(self._worker_threads),
            'queue_size': self.job_queue.qsize(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def shutdown(self):
        """Shutdown the batch processor"""
        self.logger.info("Shutting down batch processor...")
        
        self._shutdown_event.set()
        
        # Wait for worker threads to finish
        for thread in self._worker_threads:
            thread.join(timeout=5.0)
        
        self.logger.info("Batch processor shut down complete")
    
    def __del__(self):
        """Destructor to ensure clean shutdown"""
        if not self._shutdown_event.is_set():
            self.shutdown()