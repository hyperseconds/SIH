#!/usr/bin/env python3
"""
Advanced Data Filtering and Smoothing Algorithms for CME Prediction System
Implements sophisticated signal processing techniques using SciPy
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import savgol_filter, butter, filtfilt, wiener, medfilt
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
import logging


class AdvancedDataFilter:
    """Advanced data filtering and smoothing algorithms for solar wind data"""
    
    def __init__(self):
        """Initialize the data filter"""
        self.logger = logging.getLogger(__name__)
        
    def adaptive_filter(self, data: np.ndarray, 
                       window_size: int = 10, 
                       threshold_factor: float = 2.0) -> np.ndarray:
        """
        Adaptive filter that adjusts smoothing based on local data variability
        
        Args:
            data: Input data array
            window_size: Base window size for analysis
            threshold_factor: Factor for determining when to apply stronger filtering
            
        Returns:
            Filtered data array
        """
        filtered_data = np.copy(data)
        n = len(data)
        
        for i in range(window_size, n - window_size):
            # Calculate local variability
            local_window = data[i-window_size:i+window_size+1]
            local_std = np.std(local_window)
            local_mean = np.mean(local_window)
            
            # Calculate global statistics for comparison
            global_std = np.std(data)
            
            # Adaptive filtering based on local variability
            if local_std > threshold_factor * global_std:
                # High variability - apply stronger smoothing
                kernel_size = min(window_size * 2, 21)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                start_idx = max(0, i - kernel_size // 2)
                end_idx = min(n, i + kernel_size // 2 + 1)
                
                # Use Savitzky-Golay filter for smooth regions
                if end_idx - start_idx >= kernel_size:
                    filtered_data[i] = savgol_filter(
                        data[start_idx:end_idx], 
                        kernel_size, 
                        polyorder=min(3, kernel_size-1)
                    )[i - start_idx]
                else:
                    filtered_data[i] = np.mean(local_window)
            else:
                # Low variability - minimal filtering
                filtered_data[i] = np.mean(data[max(0, i-2):min(n, i+3)])
        
        return filtered_data
    
    def multi_scale_filter(self, data: np.ndarray, 
                          scales: List[int] = None) -> Dict[str, np.ndarray]:
        """
        Multi-scale filtering for different frequency components
        
        Args:
            data: Input data array
            scales: List of scale factors for different frequency bands
            
        Returns:
            Dictionary with filtered data at different scales
        """
        if scales is None:
            scales = [3, 7, 15, 31]  # Different time scales
        
        results = {'original': data}
        
        for scale in scales:
            # Ensure odd window size
            window = scale if scale % 2 == 1 else scale + 1
            
            # Apply different filters at each scale
            gaussian_filtered = gaussian_filter1d(data, sigma=scale/3)
            savgol_filtered = savgol_filter(data, window, polyorder=min(3, window-1))
            median_filtered = median_filter(data, size=window)
            
            results[f'gaussian_scale_{scale}'] = gaussian_filtered
            results[f'savgol_scale_{scale}'] = savgol_filtered
            results[f'median_scale_{scale}'] = median_filtered
            
            # Combine filters for robust smoothing
            combined = (gaussian_filtered + savgol_filtered + median_filtered) / 3
            results[f'combined_scale_{scale}'] = combined
        
        return results
    
    def frequency_domain_filter(self, data: np.ndarray, 
                               sampling_rate: float = 1.0,
                               filter_type: str = 'lowpass',
                               cutoff_freq: float = 0.1,
                               filter_order: int = 4) -> np.ndarray:
        """
        Frequency domain filtering using Butterworth filters
        
        Args:
            data: Input data array
            sampling_rate: Sampling rate in Hz
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
            cutoff_freq: Cutoff frequency (or [low, high] for bandpass/bandstop)
            filter_order: Filter order
            
        Returns:
            Filtered data array
        """
        nyquist_freq = sampling_rate / 2
        
        if filter_type in ['lowpass', 'highpass']:
            normalized_cutoff = cutoff_freq / nyquist_freq
            b, a = butter(filter_order, normalized_cutoff, btype=filter_type)
        elif filter_type in ['bandpass', 'bandstop']:
            if isinstance(cutoff_freq, (list, tuple)) and len(cutoff_freq) == 2:
                low_freq, high_freq = cutoff_freq
                normalized_freqs = [low_freq / nyquist_freq, high_freq / nyquist_freq]
                b, a = butter(filter_order, normalized_freqs, btype=filter_type)
            else:
                raise ValueError("Bandpass/bandstop filters require [low_freq, high_freq]")
        else:
            raise ValueError("Filter type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        # Apply zero-phase filtering
        filtered_data = filtfilt(b, a, data)
        
        return filtered_data
    
    def wavelet_denoising(self, data: np.ndarray, 
                         wavelet: str = 'db4',
                         levels: int = None,
                         threshold_mode: str = 'soft') -> np.ndarray:
        """
        Wavelet denoising using approximation coefficients
        Simple implementation without PyWavelets
        
        Args:
            data: Input data array
            wavelet: Wavelet type (simplified implementation)
            levels: Number of decomposition levels
            threshold_mode: Thresholding mode
            
        Returns:
            Denoised data array
        """
        # Simple wavelet-like denoising using multi-resolution analysis
        if levels is None:
            levels = int(np.log2(len(data))) - 2
            levels = max(1, min(levels, 6))
        
        # Start with the original data
        approximation = data.copy()
        
        for level in range(levels):
            # Simple averaging for approximation
            kernel_size = 2 ** (level + 1)
            
            # Downsample and smooth
            smoothed = gaussian_filter1d(approximation, sigma=kernel_size/4)
            
            # Calculate detail coefficients (difference)
            detail = approximation - smoothed
            
            # Soft thresholding
            threshold = np.std(detail) * np.sqrt(2 * np.log(len(detail)))
            
            if threshold_mode == 'soft':
                detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
            elif threshold_mode == 'hard':
                detail = detail * (np.abs(detail) > threshold)
            
            # Update approximation
            approximation = smoothed + detail
        
        return approximation
    
    def robust_filter(self, data: np.ndarray, 
                     window_size: int = 5,
                     filter_type: str = 'median') -> np.ndarray:
        """
        Robust filtering methods resistant to outliers
        
        Args:
            data: Input data array
            window_size: Window size for filtering
            filter_type: Type of robust filter ('median', 'trimmed_mean', 'winsorized')
            
        Returns:
            Robustly filtered data array
        """
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
        
        filtered_data = np.copy(data)
        half_window = window_size // 2
        
        for i in range(half_window, len(data) - half_window):
            window = data[i - half_window:i + half_window + 1]
            
            if filter_type == 'median':
                filtered_data[i] = np.median(window)
            elif filter_type == 'trimmed_mean':
                # Remove 20% of extreme values
                trim_count = int(0.1 * len(window))
                if trim_count > 0:
                    sorted_window = np.sort(window)
                    trimmed_window = sorted_window[trim_count:-trim_count]
                    filtered_data[i] = np.mean(trimmed_window)
                else:
                    filtered_data[i] = np.mean(window)
            elif filter_type == 'winsorized':
                # Replace extreme values with percentiles
                p5, p95 = np.percentile(window, [5, 95])
                winsorized_window = np.clip(window, p5, p95)
                filtered_data[i] = np.mean(winsorized_window)
            else:
                raise ValueError("Filter type must be 'median', 'trimmed_mean', or 'winsorized'")
        
        return filtered_data
    
    def edge_preserving_filter(self, data: np.ndarray, 
                              threshold: float = None,
                              window_size: int = 5) -> np.ndarray:
        """
        Edge-preserving filter that maintains sharp transitions
        
        Args:
            data: Input data array
            threshold: Edge detection threshold
            window_size: Window size for local analysis
            
        Returns:
            Edge-preserving filtered data array
        """
        if threshold is None:
            threshold = 2.0 * np.std(data)
        
        filtered_data = np.copy(data)
        half_window = window_size // 2
        
        # Calculate gradients
        gradients = np.gradient(data)
        
        for i in range(half_window, len(data) - half_window):
            if abs(gradients[i]) > threshold:
                # Edge detected - minimal filtering
                filtered_data[i] = data[i]
            else:
                # Smooth region - apply filtering
                window = data[i - half_window:i + half_window + 1]
                filtered_data[i] = np.mean(window)
        
        return filtered_data
    
    def quality_based_filter(self, data: np.ndarray, 
                           quality_scores: np.ndarray,
                           min_quality: float = 0.5,
                           interpolation_method: str = 'linear') -> np.ndarray:
        """
        Filter data based on quality scores
        
        Args:
            data: Input data array
            quality_scores: Quality scores for each data point
            min_quality: Minimum quality threshold
            interpolation_method: Method for interpolating filtered data
            
        Returns:
            Quality-filtered data array
        """
        # Identify high-quality data points
        valid_mask = quality_scores >= min_quality
        valid_indices = np.where(valid_mask)[0]
        valid_data = data[valid_mask]
        
        if len(valid_indices) < 2:
            self.logger.warning("Insufficient high-quality data points for filtering")
            return data
        
        # Interpolate missing or low-quality data
        if interpolation_method == 'linear':
            interpolator = interp1d(valid_indices, valid_data, 
                                  kind='linear', fill_value='extrapolate')
            filtered_data = interpolator(np.arange(len(data)))
        elif interpolation_method == 'cubic':
            if len(valid_indices) >= 4:
                interpolator = interp1d(valid_indices, valid_data, 
                                      kind='cubic', fill_value='extrapolate')
                filtered_data = interpolator(np.arange(len(data)))
            else:
                # Fall back to linear if insufficient points for cubic
                interpolator = interp1d(valid_indices, valid_data, 
                                      kind='linear', fill_value='extrapolate')
                filtered_data = interpolator(np.arange(len(data)))
        elif interpolation_method == 'spline':
            try:
                spline = UnivariateSpline(valid_indices, valid_data, s=0)
                filtered_data = spline(np.arange(len(data)))
            except:
                # Fall back to linear interpolation
                interpolator = interp1d(valid_indices, valid_data, 
                                      kind='linear', fill_value='extrapolate')
                filtered_data = interpolator(np.arange(len(data)))
        else:
            raise ValueError("Interpolation method must be 'linear', 'cubic', or 'spline'")
        
        # Keep high-quality original data points
        filtered_data[valid_mask] = data[valid_mask]
        
        return filtered_data
    
    def anomaly_filter(self, data: np.ndarray, 
                      method: str = 'isolation',
                      contamination: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter anomalies from data using statistical methods
        
        Args:
            data: Input data array
            method: Anomaly detection method ('isolation', 'zscore', 'iqr')
            contamination: Expected fraction of anomalies
            
        Returns:
            Tuple of (filtered_data, anomaly_mask)
        """
        anomaly_mask = np.zeros(len(data), dtype=bool)
        
        if method == 'zscore':
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            threshold = 3.0
            anomaly_mask = z_scores > threshold
            
        elif method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomaly_mask = (data < lower_bound) | (data > upper_bound)
            
        elif method == 'isolation':
            # Simple isolation-based anomaly detection
            # Calculate local density for each point
            window_size = max(5, int(len(data) * 0.05))
            isolation_scores = np.zeros(len(data))
            
            for i in range(len(data)):
                # Find k-nearest neighbors
                distances = np.abs(data - data[i])
                k_nearest = np.partition(distances, window_size)[:window_size]
                isolation_scores[i] = np.mean(k_nearest)
            
            threshold = np.percentile(isolation_scores, (1 - contamination) * 100)
            anomaly_mask = isolation_scores > threshold
        
        else:
            raise ValueError("Method must be 'isolation', 'zscore', or 'iqr'")
        
        # Create filtered data by interpolating anomalies
        filtered_data = data.copy()
        if np.any(anomaly_mask):
            normal_indices = np.where(~anomaly_mask)[0]
            anomaly_indices = np.where(anomaly_mask)[0]
            
            if len(normal_indices) >= 2:
                interpolator = interp1d(normal_indices, data[normal_indices], 
                                      kind='linear', fill_value='extrapolate')
                filtered_data[anomaly_mask] = interpolator(anomaly_indices)
        
        return filtered_data, anomaly_mask
    
    def adaptive_smoothing(self, data: np.ndarray, 
                          alpha: float = 0.3,
                          beta: float = 0.3) -> np.ndarray:
        """
        Adaptive smoothing using exponential smoothing with trend
        
        Args:
            data: Input data array
            alpha: Smoothing parameter for level
            beta: Smoothing parameter for trend
            
        Returns:
            Adaptively smoothed data array
        """
        if len(data) < 2:
            return data
        
        smoothed_data = np.zeros_like(data)
        
        # Initialize
        level = data[0]
        trend = data[1] - data[0] if len(data) > 1 else 0
        smoothed_data[0] = level
        
        for i in range(1, len(data)):
            # Update level and trend
            prev_level = level
            level = alpha * data[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            
            smoothed_data[i] = level + trend
        
        return smoothed_data
    
    def multi_parameter_filter(self, dataframe: pd.DataFrame, 
                              filter_params: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply different filters to multiple parameters simultaneously
        
        Args:
            dataframe: DataFrame with multiple parameters
            filter_params: Dictionary specifying filters for each parameter
            
        Returns:
            DataFrame with filtered parameters
        """
        filtered_df = dataframe.copy()
        
        for column in dataframe.columns:
            if column in filter_params:
                params = filter_params[column]
                filter_type = params.get('type', 'savgol')
                
                data = dataframe[column].values
                
                if filter_type == 'savgol':
                    window = params.get('window', 11)
                    polyorder = params.get('polyorder', 3)
                    if window % 2 == 0:
                        window += 1
                    filtered_df[column] = savgol_filter(data, window, polyorder)
                    
                elif filter_type == 'gaussian':
                    sigma = params.get('sigma', 1.0)
                    filtered_df[column] = gaussian_filter1d(data, sigma)
                    
                elif filter_type == 'median':
                    window = params.get('window', 5)
                    filtered_df[column] = medfilt(data, kernel_size=window)
                    
                elif filter_type == 'butterworth':
                    cutoff = params.get('cutoff', 0.1)
                    order = params.get('order', 4)
                    sampling_rate = params.get('sampling_rate', 1.0)
                    filtered_df[column] = self.frequency_domain_filter(
                        data, sampling_rate, 'lowpass', cutoff, order
                    )
                    
                elif filter_type == 'adaptive':
                    window = params.get('window', 10)
                    threshold = params.get('threshold', 2.0)
                    filtered_df[column] = self.adaptive_filter(data, window, threshold)
                    
                elif filter_type == 'robust':
                    window = params.get('window', 5)
                    method = params.get('method', 'median')
                    filtered_df[column] = self.robust_filter(data, window, method)
        
        return filtered_df
    
    def filter_evaluation(self, original: np.ndarray, 
                         filtered: np.ndarray) -> Dict[str, float]:
        """
        Evaluate filter performance using various metrics
        
        Args:
            original: Original data
            filtered: Filtered data
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Signal-to-noise ratio improvement
        original_noise = np.std(np.diff(original))
        filtered_noise = np.std(np.diff(filtered))
        snr_improvement = 20 * np.log10(original_noise / filtered_noise) if filtered_noise > 0 else 0
        
        # Mean squared error
        mse = np.mean((original - filtered) ** 2)
        
        # Smoothness metric (second derivative)
        original_smoothness = np.mean(np.abs(np.diff(original, 2)))
        filtered_smoothness = np.mean(np.abs(np.diff(filtered, 2)))
        smoothness_improvement = (original_smoothness - filtered_smoothness) / original_smoothness if original_smoothness > 0 else 0
        
        # Correlation with original
        correlation = np.corrcoef(original, filtered)[0, 1] if len(original) > 1 else 1.0
        
        # Edge preservation (gradient correlation)
        original_gradients = np.gradient(original)
        filtered_gradients = np.gradient(filtered)
        edge_preservation = np.corrcoef(original_gradients, filtered_gradients)[0, 1] if len(original) > 1 else 1.0
        
        return {
            'snr_improvement_db': snr_improvement,
            'mean_squared_error': mse,
            'smoothness_improvement': smoothness_improvement,
            'correlation_with_original': correlation,
            'edge_preservation': edge_preservation,
            'rms_error': np.sqrt(mse),
            'max_absolute_error': np.max(np.abs(original - filtered))
        }