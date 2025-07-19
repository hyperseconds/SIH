#!/usr/bin/env python3
"""
Utility Functions and Classes for Synapse Horizon CME Prediction System
Contains logging, helper functions, and common utilities
"""

import os
import sys
import logging
import time
import json
import pickle
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import threading
import queue


class Logger:
    """Advanced logging system for the application"""
    
    def __init__(self, name: str = "SynapseHorizon", log_level: str = "INFO"):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.log_level = log_level
        self.log_queue = queue.Queue()
        self.log_handlers = []
        
        # Configure logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        self.log_handlers.append(console_handler)
        
        # File handler (if logs directory exists)
        try:
            os.makedirs("logs", exist_ok=True)
            log_filename = f"logs/{self.name.lower()}_{datetime.now().strftime('%Y%m%d')}.log"
            
            file_handler = logging.FileHandler(log_filename, encoding='utf-8')
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
            self.log_handlers.append(file_handler)
            
        except Exception as e:
            self.logger.warning(f"Could not create file handler: {e}")
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message"""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message"""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message"""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message"""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, *args, **kwargs)
    
    def log_function_call(self, func_name: str, args: tuple = None, kwargs: dict = None):
        """Log function call with parameters"""
        args_str = f"args={args}" if args else ""
        kwargs_str = f"kwargs={kwargs}" if kwargs else ""
        params = ", ".join(filter(None, [args_str, kwargs_str]))
        self.debug(f"Called {func_name}({params})")
    
    def log_execution_time(self, func_name: str, execution_time: float):
        """Log function execution time"""
        self.info(f"{func_name} completed in {execution_time:.4f} seconds")
    
    def log_memory_usage(self):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self.debug(f"Memory usage: {memory_mb:.2f} MB")
        except ImportError:
            self.debug("Memory monitoring not available (psutil not installed)")
        except Exception as e:
            self.debug(f"Could not get memory usage: {e}")


class PerformanceTimer:
    """Context manager for measuring execution time"""
    
    def __init__(self, description: str = "Operation", logger: Logger = None):
        """
        Initialize timer
        
        Args:
            description: Description of the operation being timed
            logger: Logger instance for output
        """
        self.description = description
        self.logger = logger or Logger()
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing"""
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log result"""
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"{self.description} completed in {execution_time:.4f} seconds")
        else:
            self.logger.error(f"{self.description} failed after {execution_time:.4f} seconds")
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Get elapsed time"""
        if self.start_time is None:
            return None
        
        end_time = self.end_time or time.time()
        return end_time - self.start_time


def timing_decorator(description: str = None, logger: Logger = None):
    """Decorator for timing function execution"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            desc = description or f"{func.__name__}"
            log = logger or Logger()
            
            with PerformanceTimer(desc, log):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def exception_handler(logger: Logger = None, reraise: bool = True):
    """Decorator for exception handling and logging"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or Logger()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.exception(f"Exception in {func.__name__}: {str(e)}")
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator


class DataValidator:
    """Utility class for data validation"""
    
    @staticmethod
    def validate_swis_record(record: Dict[str, Any]) -> List[str]:
        """
        Validate SWIS data record
        
        Args:
            record: SWIS data record
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Required fields
        required_fields = ['timestamp', 'flux', 'density', 'temperature', 'speed']
        for field in required_fields:
            if field not in record:
                errors.append(f"Missing required field: {field}")
            elif record[field] is None:
                errors.append(f"Field {field} cannot be None")
        
        # Numeric validations
        numeric_fields = {
            'flux': (0, 1e12),
            'density': (0, 1000),
            'temperature': (0, 1e8),
            'speed': (0, 2000),
            'pressure': (0, 1000),
            'magnetic_field': (0, 1000)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in record and record[field] is not None:
                try:
                    value = float(record[field])
                    if not (min_val <= value <= max_val):
                        errors.append(f"Field {field} value {value} outside expected range [{min_val}, {max_val}]")
                    if value < 0:
                        errors.append(f"Field {field} cannot be negative")
                except (ValueError, TypeError):
                    errors.append(f"Field {field} must be numeric")
        
        return errors
    
    @staticmethod
    def validate_cme_event(event: Dict[str, Any]) -> List[str]:
        """
        Validate CME event record
        
        Args:
            event: CME event record
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Required fields
        if 'timestamp' not in event:
            errors.append("Missing required field: timestamp")
        
        # Numeric validations
        numeric_fields = {
            'angular_width': (0, 360),
            'speed': (0, 3000),
            'acceleration': (-1000, 1000)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in event and event[field] is not None:
                try:
                    value = float(event[field])
                    if not (min_val <= value <= max_val):
                        errors.append(f"Field {field} value {value} outside expected range [{min_val}, {max_val}]")
                except (ValueError, TypeError):
                    errors.append(f"Field {field} must be numeric")
        
        return errors
    
    @staticmethod
    def validate_feature_matrix(features: Union[list, tuple], 
                               feature_names: List[str] = None) -> List[str]:
        """
        Validate feature matrix
        
        Args:
            features: Feature matrix (numpy array or list)
            feature_names: List of feature names
            
        Returns:
            List of validation errors
        """
        import numpy as np
        
        errors = []
        
        # Convert to numpy array if needed
        try:
            features = np.array(features)
        except Exception as e:
            errors.append(f"Cannot convert features to numpy array: {e}")
            return errors
        
        # Check dimensions
        if features.ndim != 2:
            errors.append(f"Feature matrix must be 2D, got {features.ndim}D")
        
        # Check for NaN or infinite values
        if np.isnan(features).any():
            nan_count = np.isnan(features).sum()
            errors.append(f"Feature matrix contains {nan_count} NaN values")
        
        if np.isinf(features).any():
            inf_count = np.isinf(features).sum()
            errors.append(f"Feature matrix contains {inf_count} infinite values")
        
        # Check feature names consistency
        if feature_names is not None:
            if len(feature_names) != features.shape[1]:
                errors.append(f"Feature names count ({len(feature_names)}) doesn't match feature matrix columns ({features.shape[1]})")
        
        return errors


class FileManager:
    """Utility class for file operations"""
    
    @staticmethod
    def ensure_directory(path: str) -> bool:
        """
        Ensure directory exists, create if necessary
        
        Args:
            path: Directory path
            
        Returns:
            True if directory exists or was created successfully
        """
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            Logger().error(f"Failed to create directory {path}: {e}")
            return False
    
    @staticmethod
    def safe_save_json(data: Any, filepath: str, backup: bool = True) -> bool:
        """
        Safely save data to JSON file with backup
        
        Args:
            data: Data to save
            filepath: Target file path
            backup: Whether to create backup of existing file
            
        Returns:
            True if save was successful
        """
        try:
            # Create backup if file exists
            if backup and os.path.exists(filepath):
                backup_path = f"{filepath}.backup.{int(time.time())}"
                try:
                    os.rename(filepath, backup_path)
                except Exception as e:
                    Logger().warning(f"Could not create backup: {e}")
            
            # Ensure directory exists
            directory = os.path.dirname(filepath)
            if directory:
                FileManager.ensure_directory(directory)
            
            # Write to temporary file first
            temp_filepath = f"{filepath}.tmp"
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            # Move temporary file to target
            os.rename(temp_filepath, filepath)
            
            Logger().debug(f"Successfully saved JSON to {filepath}")
            return True
            
        except Exception as e:
            Logger().error(f"Failed to save JSON to {filepath}: {e}")
            return False
    
    @staticmethod
    def safe_load_json(filepath: str, default: Any = None) -> Any:
        """
        Safely load JSON file with error handling
        
        Args:
            filepath: File path to load
            default: Default value if loading fails
            
        Returns:
            Loaded data or default value
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            Logger().debug(f"Successfully loaded JSON from {filepath}")
            return data
            
        except FileNotFoundError:
            Logger().warning(f"JSON file not found: {filepath}")
            return default
        except json.JSONDecodeError as e:
            Logger().error(f"JSON decode error in {filepath}: {e}")
            return default
        except Exception as e:
            Logger().error(f"Failed to load JSON from {filepath}: {e}")
            return default
    
    @staticmethod
    def safe_save_pickle(data: Any, filepath: str, backup: bool = True) -> bool:
        """
        Safely save data to pickle file
        
        Args:
            data: Data to save
            filepath: Target file path
            backup: Whether to create backup of existing file
            
        Returns:
            True if save was successful
        """
        try:
            # Create backup if file exists
            if backup and os.path.exists(filepath):
                backup_path = f"{filepath}.backup.{int(time.time())}"
                try:
                    os.rename(filepath, backup_path)
                except Exception as e:
                    Logger().warning(f"Could not create backup: {e}")
            
            # Ensure directory exists
            directory = os.path.dirname(filepath)
            if directory:
                FileManager.ensure_directory(directory)
            
            # Write to temporary file first
            temp_filepath = f"{filepath}.tmp"
            with open(temp_filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Move temporary file to target
            os.rename(temp_filepath, filepath)
            
            Logger().debug(f"Successfully saved pickle to {filepath}")
            return True
            
        except Exception as e:
            Logger().error(f"Failed to save pickle to {filepath}: {e}")
            return False
    
    @staticmethod
    def safe_load_pickle(filepath: str, default: Any = None) -> Any:
        """
        Safely load pickle file
        
        Args:
            filepath: File path to load
            default: Default value if loading fails
            
        Returns:
            Loaded data or default value
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            Logger().debug(f"Successfully loaded pickle from {filepath}")
            return data
            
        except FileNotFoundError:
            Logger().warning(f"Pickle file not found: {filepath}")
            return default
        except Exception as e:
            Logger().error(f"Failed to load pickle from {filepath}: {e}")
            return default
    
    @staticmethod
    def get_file_size(filepath: str) -> Optional[int]:
        """Get file size in bytes"""
        try:
            return os.path.getsize(filepath)
        except Exception:
            return None
    
    @staticmethod
    def cleanup_old_files(directory: str, pattern: str = "*", max_age_days: int = 30) -> int:
        """
        Clean up old files in directory
        
        Args:
            directory: Directory to clean
            pattern: File pattern to match
            max_age_days: Maximum age in days
            
        Returns:
            Number of files deleted
        """
        import glob
        
        if not os.path.exists(directory):
            return 0
        
        deleted_count = 0
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        try:
            file_pattern = os.path.join(directory, pattern)
            for filepath in glob.glob(file_pattern):
                try:
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        deleted_count += 1
                        Logger().debug(f"Deleted old file: {filepath}")
                except Exception as e:
                    Logger().warning(f"Could not delete {filepath}: {e}")
                    
        except Exception as e:
            Logger().error(f"Error during cleanup: {e}")
        
        return deleted_count


class MemoryMonitor:
    """Monitor memory usage of the application"""
    
    def __init__(self):
        """Initialize memory monitor"""
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.logger = Logger()
    
    def get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            return None
        except Exception:
            return None
    
    def update_peak(self) -> Optional[float]:
        """Update peak memory usage"""
        current = self.get_memory_usage()
        if current is not None:
            if current > self.peak_memory:
                self.peak_memory = current
        return current
    
    def log_memory_status(self):
        """Log current memory status"""
        current = self.update_peak()
        if current is not None:
            self.logger.info(f"Memory: Current={current:.2f}MB, Peak={self.peak_memory:.2f}MB, "
                           f"Delta={current - self.initial_memory:.2f}MB")


class ConfigurationManager:
    """Manage application configuration"""
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize configuration manager
        
        Args:
            config_file: Configuration file path
        """
        self.config_file = config_file
        self.config = {}
        self.logger = Logger()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        self.config = FileManager.safe_load_json(self.config_file, {})
        self.logger.info(f"Loaded configuration from {self.config_file}")
    
    def save_config(self):
        """Save configuration to file"""
        success = FileManager.safe_save_json(self.config, self.config_file)
        if success:
            self.logger.info(f"Saved configuration to {self.config_file}")
        return success
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        self.config.update(updates)


class ThreadSafeCounter:
    """Thread-safe counter utility"""
    
    def __init__(self, initial_value: int = 0):
        """Initialize counter"""
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value"""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Decrement counter and return new value"""
        with self._lock:
            self._value -= amount
            return self._value
    
    def get(self) -> int:
        """Get current value"""
        with self._lock:
            return self._value
    
    def set(self, value: int) -> int:
        """Set value"""
        with self._lock:
            self._value = value
            return self._value
    
    def reset(self) -> int:
        """Reset to zero"""
        return self.set(0)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 0:
        return "0s"
    
    units = [
        ("d", 86400),  # days
        ("h", 3600),   # hours
        ("m", 60),     # minutes
        ("s", 1),      # seconds
    ]
    
    result = []
    for unit_name, unit_seconds in units:
        if seconds >= unit_seconds:
            unit_count = int(seconds // unit_seconds)
            result.append(f"{unit_count}{unit_name}")
            seconds = seconds % unit_seconds
    
    if not result:
        return "0s"
    
    return " ".join(result)


def retry_on_exception(max_retries: int = 3, delay: float = 1.0, 
                      exceptions: tuple = (Exception,)):
    """Decorator for retrying function calls on exception"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = Logger()
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise
                    else:
                        logger.warning(f"Function {func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
            
        return wrapper
    return decorator


def deprecated(reason: str = ""):
    """Decorator to mark functions as deprecated"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import warnings
            message = f"Function {func.__name__} is deprecated"
            if reason:
                message += f": {reason}"
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


class SingletonMeta(type):
    """Metaclass for singleton pattern"""
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    import platform
    
    info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
    }
    
    # Add memory info if available
    try:
        import psutil
        memory = psutil.virtual_memory()
        info['total_memory_gb'] = memory.total / (1024**3)
        info['available_memory_gb'] = memory.available / (1024**3)
        info['cpu_count'] = psutil.cpu_count()
    except ImportError:
        pass
    
    return info
