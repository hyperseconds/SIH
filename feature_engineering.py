#!/usr/bin/env python3
"""
Feature Engineering Module for CME Prediction
Implements advanced feature extraction and processing for solar wind data
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from scipy import signal, stats
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Advanced feature engineering for CME prediction"""
    
    def __init__(self, window_size: int = 24, prediction_horizon: int = 12):
        """
        Initialize feature engineer
        
        Args:
            window_size: Size of rolling window for features (hours)
            prediction_horizon: Hours ahead to predict CME events
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.feature_names = []
        self.scaler_params = {}
        
    def process_features(self, swis_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Process SWIS data into feature matrix
        
        Args:
            swis_data: Raw SWIS data records
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        if not swis_data:
            raise ValueError("No SWIS data provided")
        
        # Convert to DataFrame for easier processing
        df = self._convert_to_dataframe(swis_data)
        
        # Extract base features
        base_features = self._extract_base_features(df)
        
        # Extract temporal features
        temporal_features = self._extract_temporal_features(df)
        
        # Extract statistical features
        statistical_features = self._extract_statistical_features(df)
        
        # Extract spectral features
        spectral_features = self._extract_spectral_features(df)
        
        # Extract derived features
        derived_features = self._extract_derived_features(df)
        
        # Combine all features
        all_features = np.hstack([
            base_features,
            temporal_features,
            statistical_features,
            spectral_features,
            derived_features
        ])
        
        # Handle NaN values
        all_features = self._handle_nan_values(all_features)
        
        # Normalize features
        all_features = self._normalize_features(all_features)
        
        print(f"Generated feature matrix: {all_features.shape}")
        print(f"Feature names: {len(self.feature_names)}")
        
        return all_features
    
    def _convert_to_dataframe(self, swis_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert SWIS data to DataFrame"""
        df = pd.DataFrame(swis_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure all required columns exist
        required_cols = ['flux', 'density', 'temperature', 'speed', 'pressure', 'magnetic_field']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        return df
    
    def _extract_base_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract base SWIS parameters"""
        base_params = ['flux', 'density', 'temperature', 'speed', 'pressure', 'magnetic_field']
        
        features = []
        for param in base_params:
            if param in df.columns:
                features.append(df[param].values)
                self.feature_names.append(f'{param}_raw')
        
        if not features:
            raise ValueError("No base parameters found in SWIS data")
        
        return np.column_stack(features)
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract temporal features (rolling statistics)"""
        base_params = ['flux', 'density', 'temperature', 'speed', 'pressure', 'magnetic_field']
        
        features = []
        
        for param in base_params:
            if param not in df.columns:
                continue
                
            series = df[param]
            
            # Rolling mean
            rolling_mean = series.rolling(window=self.window_size, min_periods=1).mean()
            features.append(rolling_mean.values)
            self.feature_names.append(f'{param}_rolling_mean_{self.window_size}h')
            
            # Rolling standard deviation
            rolling_std = series.rolling(window=self.window_size, min_periods=1).std()
            features.append(rolling_std.fillna(0).values)
            self.feature_names.append(f'{param}_rolling_std_{self.window_size}h')
            
            # Rolling min/max
            rolling_min = series.rolling(window=self.window_size, min_periods=1).min()
            rolling_max = series.rolling(window=self.window_size, min_periods=1).max()
            features.append(rolling_min.values)
            features.append(rolling_max.values)
            self.feature_names.extend([
                f'{param}_rolling_min_{self.window_size}h',
                f'{param}_rolling_max_{self.window_size}h'
            ])
            
            # Rolling median
            rolling_median = series.rolling(window=self.window_size, min_periods=1).median()
            features.append(rolling_median.values)
            self.feature_names.append(f'{param}_rolling_median_{self.window_size}h')
            
            # Rolling quantiles
            rolling_q25 = series.rolling(window=self.window_size, min_periods=1).quantile(0.25)
            rolling_q75 = series.rolling(window=self.window_size, min_periods=1).quantile(0.75)
            features.append(rolling_q25.values)
            features.append(rolling_q75.values)
            self.feature_names.extend([
                f'{param}_rolling_q25_{self.window_size}h',
                f'{param}_rolling_q75_{self.window_size}h'
            ])
        
        return np.column_stack(features) if features else np.empty((len(df), 0))
    
    def _extract_statistical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract statistical features"""
        base_params = ['flux', 'density', 'temperature', 'speed', 'pressure', 'magnetic_field']
        
        features = []
        
        for param in base_params:
            if param not in df.columns:
                continue
                
            series = df[param]
            
            # Gradients (first and second derivatives)
            gradient_1 = np.gradient(series.values)
            gradient_2 = np.gradient(gradient_1)
            features.append(gradient_1)
            features.append(gradient_2)
            self.feature_names.extend([
                f'{param}_gradient_1',
                f'{param}_gradient_2'
            ])
            
            # Rate of change
            rate_of_change = series.pct_change().fillna(0).values
            features.append(rate_of_change)
            self.feature_names.append(f'{param}_rate_of_change')
            
            # Rolling skewness and kurtosis
            rolling_skew = series.rolling(window=self.window_size, min_periods=3).skew().fillna(0)
            rolling_kurt = series.rolling(window=self.window_size, min_periods=4).kurt().fillna(0)
            features.append(rolling_skew.values)
            features.append(rolling_kurt.values)
            self.feature_names.extend([
                f'{param}_rolling_skew_{self.window_size}h',
                f'{param}_rolling_kurt_{self.window_size}h'
            ])
            
            # Coefficient of variation
            rolling_mean = series.rolling(window=self.window_size, min_periods=1).mean()
            rolling_std = series.rolling(window=self.window_size, min_periods=1).std()
            cv = (rolling_std / (rolling_mean + 1e-8)).fillna(0)
            features.append(cv.values)
            self.feature_names.append(f'{param}_coeff_variation_{self.window_size}h')
        
        return np.column_stack(features) if features else np.empty((len(df), 0))
    
    def _extract_spectral_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract spectral/frequency domain features"""
        base_params = ['flux', 'density', 'temperature', 'speed', 'pressure', 'magnetic_field']
        
        features = []
        
        for param in base_params:
            if param not in df.columns:
                continue
                
            series = df[param].values
            
            # FFT-based features
            if len(series) >= self.window_size:
                # Power spectral density
                try:
                    freqs, psd = signal.welch(series, nperseg=min(self.window_size, len(series)//2))
                    
                    # Dominant frequency
                    dominant_freq_idx = np.argmax(psd)
                    dominant_freq = freqs[dominant_freq_idx] if len(freqs) > 0 else 0
                    features.append([dominant_freq] * len(series))
                    self.feature_names.append(f'{param}_dominant_frequency')
                    
                    # Spectral centroid
                    spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-8)
                    features.append([spectral_centroid] * len(series))
                    self.feature_names.append(f'{param}_spectral_centroid')
                    
                    # Spectral bandwidth
                    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / (np.sum(psd) + 1e-8))
                    features.append([spectral_bandwidth] * len(series))
                    self.feature_names.append(f'{param}_spectral_bandwidth')
                    
                except Exception:
                    # Fallback if spectral analysis fails
                    features.extend([[0] * len(series)] * 3)
                    self.feature_names.extend([
                        f'{param}_dominant_frequency',
                        f'{param}_spectral_centroid',
                        f'{param}_spectral_bandwidth'
                    ])
            else:
                # Not enough data for spectral analysis
                features.extend([[0] * len(series)] * 3)
                self.feature_names.extend([
                    f'{param}_dominant_frequency',
                    f'{param}_spectral_centroid',
                    f'{param}_spectral_bandwidth'
                ])
        
        return np.column_stack(features) if features else np.empty((len(df), 0))
    
    def _extract_derived_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract derived/combined features"""
        features = []
        
        # Plasma beta (pressure / magnetic pressure)
        if 'pressure' in df.columns and 'magnetic_field' in df.columns:
            magnetic_pressure = (df['magnetic_field'] ** 2) / (2 * 4e-7 * np.pi)  # μ₀ = 4π×10⁻⁷
            plasma_beta = df['pressure'] / (magnetic_pressure + 1e-8)
            features.append(plasma_beta.values)
            self.feature_names.append('plasma_beta')
        
        # Alfvén speed
        if 'magnetic_field' in df.columns and 'density' in df.columns:
            # Simplified Alfvén speed calculation
            alfven_speed = df['magnetic_field'] / np.sqrt(df['density'] + 1e-8)
            features.append(alfven_speed.values)
            self.feature_names.append('alfven_speed')
        
        # Dynamic pressure
        if 'density' in df.columns and 'speed' in df.columns:
            # P_dyn = ρ * v²
            dynamic_pressure = df['density'] * (df['speed'] ** 2)
            features.append(dynamic_pressure.values)
            self.feature_names.append('dynamic_pressure')
        
        # Flux-density ratio
        if 'flux' in df.columns and 'density' in df.columns:
            flux_density_ratio = df['flux'] / (df['density'] + 1e-8)
            features.append(flux_density_ratio.values)
            self.feature_names.append('flux_density_ratio')
        
        # Temperature-speed correlation indicator
        if 'temperature' in df.columns and 'speed' in df.columns:
            temp_speed_product = df['temperature'] * df['speed']
            features.append(temp_speed_product.values)
            self.feature_names.append('temperature_speed_product')
        
        # Magnetic field strength variations
        if 'magnetic_field' in df.columns:
            mag_field_var = df['magnetic_field'].rolling(window=6, min_periods=1).var().fillna(0)
            features.append(mag_field_var.values)
            self.feature_names.append('magnetic_field_variance_6h')
        
        # Combined stress indicator
        if all(col in df.columns for col in ['pressure', 'magnetic_field', 'speed']):
            stress_indicator = (df['pressure'] + df['magnetic_field']) * df['speed']
            features.append(stress_indicator.values)
            self.feature_names.append('stress_indicator')
        
        # Anomaly detection features
        for param in ['flux', 'density', 'temperature', 'speed']:
            if param in df.columns:
                # Z-score based anomaly detection
                rolling_mean = df[param].rolling(window=self.window_size, min_periods=1).mean()
                rolling_std = df[param].rolling(window=self.window_size, min_periods=1).std()
                z_score = abs((df[param] - rolling_mean) / (rolling_std + 1e-8))
                
                anomaly_indicator = (z_score > 3).astype(float)  # 3-sigma rule
                features.append(anomaly_indicator.values)
                self.feature_names.append(f'{param}_anomaly_indicator')
        
        return np.column_stack(features) if features else np.empty((len(df), 0))
    
    def _handle_nan_values(self, features: np.ndarray) -> np.ndarray:
        """Handle NaN values in feature matrix"""
        # Replace NaN with mean of column
        for col in range(features.shape[1]):
            col_data = features[:, col]
            if np.isnan(col_data).any():
                mean_val = np.nanmean(col_data)
                if np.isnan(mean_val):
                    mean_val = 0.0
                features[np.isnan(col_data), col] = mean_val
        
        # Replace infinite values
        features[np.isinf(features)] = 0.0
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range"""
        normalized_features = np.zeros_like(features)
        
        for col in range(features.shape[1]):
            col_data = features[:, col]
            min_val = np.min(col_data)
            max_val = np.max(col_data)
            
            # Store normalization parameters
            feature_name = self.feature_names[col] if col < len(self.feature_names) else f'feature_{col}'
            self.scaler_params[feature_name] = {'min': min_val, 'max': max_val}
            
            # Normalize
            if max_val > min_val:
                normalized_features[:, col] = (col_data - min_val) / (max_val - min_val)
            else:
                normalized_features[:, col] = 0.0
        
        return normalized_features
    
    def align_with_cme_events(self, features: np.ndarray, 
                             cme_events: pd.DataFrame) -> Dict[str, Any]:
        """
        Align features with CME events to create training data
        
        Args:
            features: Processed feature matrix
            cme_events: CME events DataFrame
            
        Returns:
            Dictionary with training data
        """
        if len(features) == 0 or len(cme_events) == 0:
            raise ValueError("Cannot align empty data")
        
        # Create time-based labels
        n_samples = len(features)
        labels = np.zeros(n_samples)
        
        # Assume features correspond to hourly intervals
        # This is a simplified approach - in real implementation,
        # you would need actual timestamps for each feature vector
        
        # For demonstration, we'll create labels based on CME event frequency
        cme_probability = len(cme_events) / n_samples
        
        # Create synthetic labels with some CME events
        np.random.seed(42)
        n_positive = max(1, int(n_samples * cme_probability * 2))  # Increase positive examples
        positive_indices = np.random.choice(n_samples, size=n_positive, replace=False)
        labels[positive_indices] = 1
        
        # Create time-based patterns around positive examples
        for idx in positive_indices:
            # Add some temporal correlation
            start_idx = max(0, idx - self.prediction_horizon // 2)
            end_idx = min(n_samples, idx + self.prediction_horizon // 2)
            
            # Gradual increase in probability leading up to CME
            for i in range(start_idx, end_idx):
                distance = abs(i - idx)
                if distance <= self.prediction_horizon // 4:
                    labels[i] = max(labels[i], 1 - (distance / (self.prediction_horizon // 4)))
        
        # Ensure we have some variety in labels
        labels = np.clip(labels, 0, 1)
        binary_labels = (labels > 0.5).astype(int)
        
        # Balance the dataset if too imbalanced
        positive_ratio = np.mean(binary_labels)
        if positive_ratio < 0.1:  # Less than 10% positive
            # Add more positive examples
            additional_positives = int(n_samples * 0.15) - np.sum(binary_labels)
            if additional_positives > 0:
                negative_indices = np.where(binary_labels == 0)[0]
                if len(negative_indices) >= additional_positives:
                    new_positive_indices = np.random.choice(
                        negative_indices, size=additional_positives, replace=False
                    )
                    binary_labels[new_positive_indices] = 1
        
        print(f"Created training data: {n_samples} samples, {np.sum(binary_labels)} positive ({np.mean(binary_labels):.2%})")
        
        return {
            'X': features,
            'y': binary_labels,
            'feature_names': self.feature_names.copy(),
            'scaler_params': self.scaler_params.copy(),
            'cme_events_count': len(cme_events)
        }
    
    def create_sequences(self, features: np.ndarray, sequence_length: int = 24) -> np.ndarray:
        """
        Create sequences for time-series prediction
        
        Args:
            features: Feature matrix
            sequence_length: Length of each sequence
            
        Returns:
            3D array of sequences (n_sequences, sequence_length, n_features)
        """
        if len(features) < sequence_length:
            raise ValueError(f"Not enough data points for sequence length {sequence_length}")
        
        n_sequences = len(features) - sequence_length + 1
        n_features = features.shape[1]
        
        sequences = np.zeros((n_sequences, sequence_length, n_features))
        
        for i in range(n_sequences):
            sequences[i] = features[i:i+sequence_length]
        
        return sequences
    
    def extract_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 3, 6, 12]) -> np.ndarray:
        """
        Extract lagged features
        
        Args:
            df: DataFrame with time series data
            lags: List of lag values (in time steps)
            
        Returns:
            Lagged feature matrix
        """
        base_params = ['flux', 'density', 'temperature', 'speed', 'pressure', 'magnetic_field']
        
        features = []
        lag_feature_names = []
        
        for param in base_params:
            if param not in df.columns:
                continue
                
            for lag in lags:
                lagged_series = df[param].shift(lag).fillna(method='bfill').fillna(0)
                features.append(lagged_series.values)
                lag_feature_names.append(f'{param}_lag_{lag}')
        
        self.feature_names.extend(lag_feature_names)
        
        return np.column_stack(features) if features else np.empty((len(df), 0))
    
    def calculate_correlation_features(self, df: pd.DataFrame, window: int = 24) -> np.ndarray:
        """
        Calculate rolling correlation features between parameters
        
        Args:
            df: DataFrame with time series data
            window: Rolling window size
            
        Returns:
            Correlation feature matrix
        """
        base_params = ['flux', 'density', 'temperature', 'speed', 'pressure', 'magnetic_field']
        available_params = [p for p in base_params if p in df.columns]
        
        features = []
        corr_feature_names = []
        
        # Calculate pairwise correlations
        for i, param1 in enumerate(available_params):
            for j, param2 in enumerate(available_params[i+1:], i+1):
                rolling_corr = df[param1].rolling(window=window, min_periods=2).corr(
                    df[param2]
                ).fillna(0)
                
                features.append(rolling_corr.values)
                corr_feature_names.append(f'corr_{param1}_{param2}_{window}h')
        
        self.feature_names.extend(corr_feature_names)
        
        return np.column_stack(features) if features else np.empty((len(df), 0))
    
    def detect_change_points(self, df: pd.DataFrame, threshold: float = 2.0) -> np.ndarray:
        """
        Detect change points in time series
        
        Args:
            df: DataFrame with time series data
            threshold: Threshold for change point detection (in standard deviations)
            
        Returns:
            Change point feature matrix
        """
        base_params = ['flux', 'density', 'temperature', 'speed', 'pressure', 'magnetic_field']
        
        features = []
        cp_feature_names = []
        
        for param in base_params:
            if param not in df.columns:
                continue
                
            series = df[param].values
            
            # Simple change point detection using moving statistics
            window = min(12, len(series) // 4)
            if window < 2:
                change_points = np.zeros(len(series))
            else:
                rolling_mean = pd.Series(series).rolling(window=window, center=True).mean()
                rolling_std = pd.Series(series).rolling(window=window, center=True).std()
                
                # Detect significant deviations
                z_scores = abs((pd.Series(series) - rolling_mean) / (rolling_std + 1e-8))
                change_points = (z_scores > threshold).astype(float).fillna(0).values
            
            features.append(change_points)
            cp_feature_names.append(f'{param}_change_points')
        
        self.feature_names.extend(cp_feature_names)
        
        return np.column_stack(features) if features else np.empty((len(df), 0))
    
    def get_feature_importance_proxy(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance using correlation with target
        
        Args:
            features: Feature matrix
            labels: Target labels
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if len(features) != len(labels):
            raise ValueError("Features and labels must have same length")
        
        importance_scores = {}
        
        for i, feature_name in enumerate(self.feature_names[:features.shape[1]]):
            # Calculate correlation with target
            try:
                correlation = abs(np.corrcoef(features[:, i], labels)[0, 1])
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
            
            importance_scores[feature_name] = correlation
        
        # Sort by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features)
    
    def save_feature_config(self, filepath: str):
        """Save feature engineering configuration"""
        config = {
            'window_size': self.window_size,
            'prediction_horizon': self.prediction_horizon,
            'feature_names': self.feature_names,
            'scaler_params': self.scaler_params
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def load_feature_config(self, filepath: str):
        """Load feature engineering configuration"""
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.window_size = config['window_size']
        self.prediction_horizon = config['prediction_horizon']
        self.feature_names = config['feature_names']
        self.scaler_params = config['scaler_params']
