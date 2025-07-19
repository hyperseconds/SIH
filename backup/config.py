#!/usr/bin/env python3
"""
Configuration Module for Synapse Horizon CME Prediction System
Contains all configuration constants, settings, and parameters
"""

import os
from typing import Dict, List, Any
from datetime import timedelta

class Config:
    """Main configuration class for the application"""
    
    # Application Information
    APP_NAME = "Synapse Horizon"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Advanced CME Prediction System using Custom Neural Networks"
    
    # Database Configuration
    DATABASE_PATH = "synapse_horizon.db"
    DATABASE_BACKUP_DIR = "backups"
    DATABASE_TIMEOUT = 30  # seconds
    
    # Data Processing Configuration
    DEFAULT_WINDOW_SIZE = 24  # hours for rolling statistics
    DEFAULT_PREDICTION_HORIZON = 12  # hours ahead to predict
    TIME_ALIGNMENT_TOLERANCE = 6  # hours for timestamp alignment
    
    # Neural Network Configuration
    DEFAULT_HIDDEN_LAYERS = [64, 32, 16]
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_EPOCHS = 100
    DEFAULT_BATCH_SIZE = 32
    ACTIVATION_FUNCTION = 'relu'
    OUTPUT_ACTIVATION = 'sigmoid'
    
    # Feature Engineering Configuration
    ROLLING_WINDOW_SIZES = [6, 12, 24, 48]  # hours
    LAG_PERIODS = [1, 3, 6, 12, 24]  # hours
    ANOMALY_THRESHOLD = 3.0  # standard deviations
    
    # SWIS Parameter Configuration
    SWIS_PARAMETERS = {
        'flux': {
            'unit': 'counts/s',
            'expected_range': (0, 1e10),
            'log_scale': True,
            'critical_threshold': 1e8
        },
        'density': {
            'unit': 'cm⁻³',
            'expected_range': (0, 100),
            'log_scale': False,
            'critical_threshold': 50
        },
        'temperature': {
            'unit': 'K',
            'expected_range': (0, 1e7),
            'log_scale': True,
            'critical_threshold': 1e6
        },
        'speed': {
            'unit': 'km/s',
            'expected_range': (200, 1000),
            'log_scale': False,
            'critical_threshold': 700
        },
        'pressure': {
            'unit': 'nPa',
            'expected_range': (0, 100),
            'log_scale': False,
            'critical_threshold': 20
        },
        'magnetic_field': {
            'unit': 'nT',
            'expected_range': (0, 100),
            'log_scale': False,
            'critical_threshold': 50
        }
    }
    
    # CME Event Configuration
    CME_CLASSIFICATION = {
        'halo': {
            'angular_width_min': 360,
            'description': 'Full halo CME'
        },
        'partial_halo': {
            'angular_width_min': 270,
            'angular_width_max': 359,
            'description': 'Partial halo CME'
        },
        'wide': {
            'angular_width_min': 120,
            'angular_width_max': 269,
            'description': 'Wide CME'
        },
        'normal': {
            'angular_width_min': 30,
            'angular_width_max': 119,
            'description': 'Normal CME'
        },
        'narrow': {
            'angular_width_min': 0,
            'angular_width_max': 29,
            'description': 'Narrow CME'
        }
    }
    
    # Risk Level Configuration
    RISK_LEVELS = {
        'low': {
            'threshold_min': 0.0,
            'threshold_max': 0.3,
            'color': '#6BCF7F',
            'description': 'Low probability of CME occurrence',
            'recommended_action': 'Continue normal operations'
        },
        'medium': {
            'threshold_min': 0.3,
            'threshold_max': 0.7,
            'color': '#FF8C42',
            'description': 'Moderate probability of CME occurrence',
            'recommended_action': 'Monitor conditions closely'
        },
        'high': {
            'threshold_min': 0.7,
            'threshold_max': 1.0,
            'color': '#FF3838',
            'description': 'High probability of CME occurrence',
            'recommended_action': 'Prepare for potential space weather impacts'
        }
    }
    
    # Alert System Configuration
    ALERT_THRESHOLDS = {
        'cme_probability': 0.7,
        'parameter_anomaly': 3.0,  # standard deviations
        'data_gap': 4,  # hours
        'system_error': True
    }
    
    ALERT_COOLDOWN = 3600  # seconds (1 hour)
    
    # GUI Configuration
    GUI_THEME = 'dark'
    GUI_COLORS = {
        'background': '#1e1e1e',
        'surface': '#2d2d2d',
        'primary': '#4CAF50',
        'secondary': '#2196F3',
        'error': '#F44336',
        'warning': '#FF9800',
        'success': '#4CAF50',
        'text_primary': '#ffffff',
        'text_secondary': '#cccccc'
    }
    
    GUI_FONTS = {
        'default': ('Arial', 10),
        'header': ('Arial', 16, 'bold'),
        'subheader': ('Arial', 12, 'bold'),
        'monospace': ('Courier New', 9)
    }
    
    WINDOW_SIZE = {
        'main': (1200, 800),
        'plot': (1000, 600),
        'dialog': (400, 300)
    }
    
    # Visualization Configuration
    PLOT_STYLE = 'dark'
    PLOT_COLORS = {
        'flux': '#FF6B6B',
        'density': '#4ECDC4',
        'temperature': '#45B7D1',
        'speed': '#96CEB4',
        'pressure': '#FFEAA7',
        'magnetic_field': '#DDA0DD',
        'cme_event': '#FF4757',
        'prediction': '#5F27CD',
        'anomaly': '#FF3838'
    }
    
    PLOT_SETTINGS = {
        'dpi': 100,
        'figure_size': (12, 8),
        'line_width': 1.5,
        'marker_size': 4,
        'grid_alpha': 0.3,
        'legend_alpha': 0.9
    }
    
    # File Paths and Directories
    DATA_DIR = "data"
    MODELS_DIR = "models"
    LOGS_DIR = "logs"
    EXPORTS_DIR = "exports"
    CONFIG_DIR = "config"
    
    DEFAULT_FILES = {
        'swis_data': 'swis_data.json',
        'cme_events': 'cactus_events.csv',
        'model_config': 'model_config.json',
        'feature_config': 'feature_config.json'
    }
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    LOG_FILE_BACKUP_COUNT = 5
    
    # Data Validation Configuration
    DATA_QUALITY_CHECKS = {
        'missing_values_threshold': 0.1,  # 10% missing values
        'outlier_threshold': 0.05,  # 5% outliers
        'time_gap_threshold': timedelta(hours=2),
        'minimum_samples': 100
    }
    
    # Model Training Configuration
    TRAINING_CONFIG = {
        'validation_split': 0.2,
        'test_split': 0.1,
        'cross_validation_folds': 5,
        'early_stopping_patience': 10,
        'learning_rate_decay': 0.95,
        'weight_decay': 1e-5,
        'dropout_rate': 0.1
    }
    
    # Performance Metrics
    PERFORMANCE_THRESHOLDS = {
        'minimum_accuracy': 0.7,
        'minimum_precision': 0.6,
        'minimum_recall': 0.6,
        'minimum_f1_score': 0.6,
        'maximum_false_positive_rate': 0.3
    }
    
    # Export Configuration
    EXPORT_FORMATS = {
        'csv': {
            'delimiter': ',',
            'encoding': 'utf-8',
            'date_format': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            'indent': 2,
            'ensure_ascii': False,
            'date_format': 'iso'
        },
        'excel': {
            'engine': 'openpyxl',
            'date_format': 'YYYY-MM-DD HH:MM:SS'
        }
    }
    
    # API Configuration (for future extensions)
    API_CONFIG = {
        'timeout': 30,
        'retry_attempts': 3,
        'retry_delay': 1,
        'rate_limit': 60  # requests per minute
    }
    
    # Space Weather Data Sources
    DATA_SOURCES = {
        'swis': {
            'name': 'Solar Wind Ion Spectrometer',
            'spacecraft': 'Aditya-L1',
            'parameters': ['flux', 'density', 'temperature', 'speed'],
            'cadence': '1 hour',
            'description': 'Solar wind particle measurements from L1 orbit'
        },
        'cactus': {
            'name': 'CACTUS CME Catalog',
            'source': 'Royal Observatory of Belgium',
            'parameters': ['timestamp', 'angular_width', 'speed', 'acceleration'],
            'description': 'Automated CME detection and tracking'
        }
    }
    
    # System Requirements
    SYSTEM_REQUIREMENTS = {
        'minimum_ram': '4 GB',
        'recommended_ram': '8 GB',
        'minimum_disk_space': '1 GB',
        'python_version': '3.7+',
        'dependencies': [
            'numpy>=1.19.0',
            'scipy>=1.5.0',
            'matplotlib>=3.3.0',
            'pandas>=1.1.0'
        ]
    }
    
    # Feature Importance Configuration
    FEATURE_IMPORTANCE_CONFIG = {
        'methods': ['correlation', 'mutual_info', 'permutation'],
        'top_features': 20,
        'correlation_threshold': 0.95,
        'variance_threshold': 0.01
    }
    
    # Model Ensemble Configuration
    ENSEMBLE_CONFIG = {
        'n_models': 5,
        'bootstrap_ratio': 0.8,
        'feature_sampling_ratio': 0.9,
        'voting_method': 'soft',
        'diversity_threshold': 0.1
    }
    
    # Time Series Configuration
    TIME_SERIES_CONFIG = {
        'sequence_length': 24,
        'prediction_steps': 12,
        'overlap_ratio': 0.5,
        'normalization_method': 'minmax',
        'trend_removal': True,
        'seasonal_decomposition': True
    }
    
    # Quality Assurance Configuration
    QA_CONFIG = {
        'unit_tests': True,
        'integration_tests': True,
        'performance_benchmarks': True,
        'code_coverage_threshold': 0.8,
        'documentation_coverage': 0.9
    }

    @classmethod
    def get_parameter_config(cls, parameter: str) -> Dict[str, Any]:
        """Get configuration for a specific SWIS parameter"""
        return cls.SWIS_PARAMETERS.get(parameter, {})
    
    @classmethod
    def get_risk_level(cls, probability: float) -> Dict[str, Any]:
        """Get risk level information for a given probability"""
        for level, config in cls.RISK_LEVELS.items():
            if config['threshold_min'] <= probability < config['threshold_max']:
                return {'level': level, **config}
        return {'level': 'unknown', 'description': 'Unknown risk level'}
    
    @classmethod
    def get_cme_classification(cls, angular_width: float) -> Dict[str, Any]:
        """Classify CME based on angular width"""
        for cme_type, config in cls.CME_CLASSIFICATION.items():
            min_width = config.get('angular_width_min', 0)
            max_width = config.get('angular_width_max', float('inf'))
            if min_width <= angular_width <= max_width:
                return {'type': cme_type, **config}
        return {'type': 'unknown', 'description': 'Unknown CME type'}
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.EXPORTS_DIR,
            cls.CONFIG_DIR,
            cls.DATABASE_BACKUP_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def validate_configuration(cls) -> List[str]:
        """Validate configuration settings"""
        issues = []
        
        # Check required directories
        try:
            cls.create_directories()
        except Exception as e:
            issues.append(f"Failed to create directories: {e}")
        
        # Validate neural network configuration
        if not cls.DEFAULT_HIDDEN_LAYERS:
            issues.append("DEFAULT_HIDDEN_LAYERS cannot be empty")
        
        if cls.DEFAULT_LEARNING_RATE <= 0:
            issues.append("DEFAULT_LEARNING_RATE must be positive")
        
        if cls.DEFAULT_EPOCHS <= 0:
            issues.append("DEFAULT_EPOCHS must be positive")
        
        # Validate risk thresholds
        risk_thresholds = [config['threshold_max'] for config in cls.RISK_LEVELS.values()]
        if not all(t1 <= t2 for t1, t2 in zip(risk_thresholds, risk_thresholds[1:])):
            issues.append("Risk level thresholds are not properly ordered")
        
        # Validate SWIS parameters
        for param, config in cls.SWIS_PARAMETERS.items():
            if 'expected_range' not in config:
                issues.append(f"SWIS parameter {param} missing expected_range")
            elif len(config['expected_range']) != 2:
                issues.append(f"SWIS parameter {param} expected_range must have 2 values")
        
        return issues

    @classmethod
    def load_from_file(cls, config_file: str):
        """Load configuration from file (for future extensions)"""
        # Placeholder for loading configuration from JSON/YAML file
        pass
    
    @classmethod
    def save_to_file(cls, config_file: str):
        """Save configuration to file (for future extensions)"""
        # Placeholder for saving configuration to JSON/YAML file
        pass


class DevConfig(Config):
    """Development configuration with debug settings"""
    LOG_LEVEL = 'DEBUG'
    DATABASE_PATH = "synapse_horizon_dev.db"
    GUI_THEME = 'light'
    
    # Reduced training parameters for faster development
    DEFAULT_EPOCHS = 20
    DEFAULT_BATCH_SIZE = 16


class TestConfig(Config):
    """Test configuration for unit tests"""
    DATABASE_PATH = ":memory:"  # In-memory database for tests
    LOG_LEVEL = 'WARNING'
    
    # Minimal training for tests
    DEFAULT_EPOCHS = 5
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_HIDDEN_LAYERS = [16, 8]


class ProductionConfig(Config):
    """Production configuration with optimized settings"""
    LOG_LEVEL = 'WARNING'
    
    # Optimized training parameters
    DEFAULT_EPOCHS = 200
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_HIDDEN_LAYERS = [128, 64, 32, 16]
    
    # Enhanced validation
    TRAINING_CONFIG = {
        **Config.TRAINING_CONFIG,
        'validation_split': 0.15,
        'cross_validation_folds': 10,
        'early_stopping_patience': 15
    }


def get_config(environment: str = 'production') -> Config:
    """Get configuration based on environment"""
    config_map = {
        'development': DevConfig,
        'dev': DevConfig,
        'test': TestConfig,
        'testing': TestConfig,
        'production': ProductionConfig,
        'prod': ProductionConfig
    }
    
    config_class = config_map.get(environment.lower(), Config)
    return config_class()


# Global configuration instance
config = get_config(os.getenv('ENVIRONMENT', 'production'))
