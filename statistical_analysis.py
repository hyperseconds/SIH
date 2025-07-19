#!/usr/bin/env python3
"""
Advanced Statistical Analysis Tools for CME Prediction System
Implements sophisticated statistical methods using SciPy
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq, fftshift
from scipy.ndimage import gaussian_filter1d, median_filter
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from datetime import datetime, timedelta
import logging


class AdvancedStatisticalAnalyzer:
    """Advanced statistical analysis tools using SciPy for CME prediction"""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer
        
        Args:
            confidence_level: Confidence level for statistical tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.logger = logging.getLogger(__name__)
        
    def analyze_temporal_patterns(self, data: pd.DataFrame, timestamp_col: str = 'timestamp') -> Dict[str, Any]:
        """
        Analyze temporal patterns in solar wind data
        
        Args:
            data: DataFrame with solar wind measurements
            timestamp_col: Name of timestamp column
            
        Returns:
            Dictionary with temporal pattern analysis results
        """
        results = {}
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
            data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        
        # Extract time components
        data = data.copy()
        data['hour'] = data[timestamp_col].dt.hour
        data['day_of_week'] = data[timestamp_col].dt.dayofweek
        data['month'] = data[timestamp_col].dt.month
        
        # Analyze hourly patterns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['hour', 'day_of_week', 'month']]
        
        results['hourly_patterns'] = {}
        for col in numeric_cols:
            hourly_stats = data.groupby('hour')[col].agg(['mean', 'std', 'count'])
            
            # Test for significant hourly variation using ANOVA
            hourly_groups = [group[col].values for name, group in data.groupby('hour')]
            f_stat, p_value = stats.f_oneway(*hourly_groups)
            
            results['hourly_patterns'][col] = {
                'statistics': hourly_stats.to_dict(),
                'anova_f_statistic': f_stat,
                'anova_p_value': p_value,
                'significant_variation': p_value < self.alpha
            }
        
        # Weekly patterns
        results['weekly_patterns'] = {}
        for col in numeric_cols:
            weekly_stats = data.groupby('day_of_week')[col].agg(['mean', 'std', 'count'])
            
            weekly_groups = [group[col].values for name, group in data.groupby('day_of_week')]
            f_stat, p_value = stats.f_oneway(*weekly_groups)
            
            results['weekly_patterns'][col] = {
                'statistics': weekly_stats.to_dict(),
                'anova_f_statistic': f_stat,
                'anova_p_value': p_value,
                'significant_variation': p_value < self.alpha
            }
        
        # Monthly patterns
        results['monthly_patterns'] = {}
        for col in numeric_cols:
            monthly_stats = data.groupby('month')[col].agg(['mean', 'std', 'count'])
            
            monthly_groups = [group[col].values for name, group in data.groupby('month')]
            f_stat, p_value = stats.f_oneway(*monthly_groups)
            
            results['monthly_patterns'][col] = {
                'statistics': monthly_stats.to_dict(),
                'anova_f_statistic': f_stat,
                'anova_p_value': p_value,
                'significant_variation': p_value < self.alpha
            }
        
        self.logger.info("Temporal pattern analysis completed")
        return results
    
    def correlation_analysis(self, data: pd.DataFrame, method: str = 'pearson') -> Dict[str, Any]:
        """
        Comprehensive correlation analysis with significance testing
        
        Args:
            data: DataFrame with numeric data
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary with correlation results and significance tests
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        if method == 'pearson':
            corr_matrix = numeric_data.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = numeric_data.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = numeric_data.corr(method='kendall')
        else:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
        
        # Calculate p-values for correlations
        n = len(numeric_data)
        pvalue_matrix = pd.DataFrame(np.zeros_like(corr_matrix), 
                                   index=corr_matrix.index, 
                                   columns=corr_matrix.columns)
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i != j:
                    if method == 'pearson':
                        _, p_val = stats.pearsonr(numeric_data[col1], numeric_data[col2])
                    elif method == 'spearman':
                        _, p_val = stats.spearmanr(numeric_data[col1], numeric_data[col2])
                    elif method == 'kendall':
                        _, p_val = stats.kendalltau(numeric_data[col1], numeric_data[col2])
                    
                    pvalue_matrix.iloc[i, j] = p_val
        
        # Find significant correlations
        significant_mask = pvalue_matrix < self.alpha
        significant_correlations = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j and significant_mask.iloc[i, j]:
                    significant_correlations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': corr_matrix.iloc[i, j],
                        'p_value': pvalue_matrix.iloc[i, j],
                        'strength': self._interpret_correlation_strength(abs(corr_matrix.iloc[i, j]))
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'p_value_matrix': pvalue_matrix.to_dict(),
            'significant_correlations': significant_correlations,
            'method': method,
            'sample_size': n,
            'confidence_level': self.confidence_level
        }
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.9:
            return "Very Strong"
        elif abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.5:
            return "Moderate"
        elif abs_corr >= 0.3:
            return "Weak"
        else:
            return "Very Weak"
    
    def distribution_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze distributions and test for normality
        
        Args:
            data: DataFrame with numeric data
            
        Returns:
            Dictionary with distribution analysis results
        """
        results = {}
        numeric_data = data.select_dtypes(include=[np.number])
        
        for column in numeric_data.columns:
            series = numeric_data[column].dropna()
            
            # Basic statistics
            basic_stats = {
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'var': float(series.var()),
                'skewness': float(stats.skew(series)),
                'kurtosis': float(stats.kurtosis(series)),
                'min': float(series.min()),
                'max': float(series.max()),
                'range': float(series.max() - series.min()),
                'iqr': float(series.quantile(0.75) - series.quantile(0.25))
            }
            
            # Normality tests
            normality_tests = {}
            
            # Shapiro-Wilk test (best for n < 5000)
            if len(series) < 5000:
                shapiro_stat, shapiro_p = stats.shapiro(series)
                normality_tests['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > self.alpha
                }
            
            # D'Agostino's test
            try:
                dagostino_stat, dagostino_p = stats.normaltest(series)
                normality_tests['dagostino'] = {
                    'statistic': float(dagostino_stat),
                    'p_value': float(dagostino_p),
                    'is_normal': dagostino_p > self.alpha
                }
            except:
                pass
            
            # Kolmogorov-Smirnov test against normal distribution
            ks_stat, ks_p = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
            normality_tests['kolmogorov_smirnov'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_p),
                'is_normal': ks_p > self.alpha
            }
            
            # Anderson-Darling test
            ad_result = stats.anderson(series, dist='norm')
            normality_tests['anderson_darling'] = {
                'statistic': float(ad_result.statistic),
                'critical_values': ad_result.critical_values.tolist(),
                'significance_levels': ad_result.significance_level.tolist()
            }
            
            # Distribution fitting
            distributions_to_test = [
                ('normal', stats.norm),
                ('lognormal', stats.lognorm),
                ('exponential', stats.expon),
                ('gamma', stats.gamma),
                ('weibull', stats.weibull_min)
            ]
            
            best_fit = {'name': None, 'params': None, 'aic': np.inf, 'bic': np.inf}
            distribution_fits = {}
            
            for dist_name, distribution in distributions_to_test:
                try:
                    params = distribution.fit(series)
                    
                    # Calculate AIC and BIC
                    log_likelihood = np.sum(distribution.logpdf(series, *params))
                    k = len(params)  # number of parameters
                    n = len(series)
                    
                    aic = 2 * k - 2 * log_likelihood
                    bic = k * np.log(n) - 2 * log_likelihood
                    
                    # KS test for goodness of fit
                    ks_stat_fit, ks_p_fit = stats.kstest(series, 
                                                       lambda x: distribution.cdf(x, *params))
                    
                    distribution_fits[dist_name] = {
                        'parameters': params,
                        'aic': float(aic),
                        'bic': float(bic),
                        'log_likelihood': float(log_likelihood),
                        'ks_statistic': float(ks_stat_fit),
                        'ks_p_value': float(ks_p_fit),
                        'good_fit': ks_p_fit > self.alpha
                    }
                    
                    if aic < best_fit['aic']:
                        best_fit = {
                            'name': dist_name,
                            'params': params,
                            'aic': aic,
                            'bic': bic
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Could not fit {dist_name} distribution to {column}: {e}")
                    continue
            
            results[column] = {
                'basic_statistics': basic_stats,
                'normality_tests': normality_tests,
                'distribution_fits': distribution_fits,
                'best_fit_distribution': best_fit
            }
        
        self.logger.info(f"Distribution analysis completed for {len(results)} variables")
        return results
    
    def time_series_decomposition(self, data: pd.Series, period: int = None) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components
        
        Args:
            data: Time series data
            period: Seasonal period (auto-detected if None)
            
        Returns:
            Dictionary with decomposition results
        """
        from scipy.signal import periodogram
        
        # Remove NaN values
        clean_data = data.dropna()
        
        if len(clean_data) < 10:
            raise ValueError("Insufficient data for time series decomposition")
        
        # Auto-detect period using periodogram if not specified
        if period is None:
            frequencies, power = periodogram(clean_data.values)
            # Find the frequency with maximum power (excluding DC component)
            max_freq_idx = np.argmax(power[1:]) + 1
            period = int(1 / frequencies[max_freq_idx]) if frequencies[max_freq_idx] > 0 else len(clean_data) // 4
            period = max(2, min(period, len(clean_data) // 2))
        
        # Simple moving average for trend
        trend = clean_data.rolling(window=period, center=True).mean()
        
        # Detrended data
        detrended = clean_data - trend
        
        # Seasonal component (average of each period position)
        seasonal = pd.Series(index=clean_data.index, dtype=float)
        for i in range(len(clean_data)):
            period_position = i % period
            seasonal.iloc[i] = detrended.iloc[period_position::period].mean()
        
        # Residual
        residual = clean_data - trend - seasonal
        
        # Statistics
        total_variance = clean_data.var()
        trend_variance = trend.var()
        seasonal_variance = seasonal.var()
        residual_variance = residual.var()
        
        # Calculate R-squared for each component
        trend_r2 = 1 - (residual_variance + seasonal_variance) / total_variance if total_variance > 0 else 0
        seasonal_r2 = seasonal_variance / total_variance if total_variance > 0 else 0
        
        return {
            'original': clean_data.to_dict(),
            'trend': trend.to_dict(),
            'seasonal': seasonal.to_dict(),
            'residual': residual.to_dict(),
            'period': period,
            'variance_explained': {
                'trend': float(trend_r2),
                'seasonal': float(seasonal_r2),
                'residual': float(residual_variance / total_variance if total_variance > 0 else 0)
            },
            'statistics': {
                'total_variance': float(total_variance),
                'trend_variance': float(trend_variance),
                'seasonal_variance': float(seasonal_variance),
                'residual_variance': float(residual_variance)
            }
        }
    
    def spectral_analysis(self, data: pd.Series, sampling_rate: float = 1.0) -> Dict[str, Any]:
        """
        Perform spectral analysis using FFT
        
        Args:
            data: Time series data
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with spectral analysis results
        """
        clean_data = data.dropna().values
        n = len(clean_data)
        
        if n < 4:
            raise ValueError("Insufficient data for spectral analysis")
        
        # Apply window function to reduce spectral leakage
        window = np.hanning(n)
        windowed_data = clean_data * window
        
        # Compute FFT
        fft_result = fft(windowed_data)
        frequencies = fftfreq(n, d=1/sampling_rate)
        
        # Power spectral density
        power_spectrum = np.abs(fft_result)**2 / n
        
        # Only take positive frequencies
        positive_freq_mask = frequencies > 0
        positive_frequencies = frequencies[positive_freq_mask]
        positive_power = power_spectrum[positive_freq_mask]
        
        # Find peaks in power spectrum
        peak_indices, peak_properties = find_peaks(positive_power, height=np.std(positive_power))
        
        dominant_frequencies = []
        for idx in peak_indices:
            dominant_frequencies.append({
                'frequency': float(positive_frequencies[idx]),
                'power': float(positive_power[idx]),
                'period': float(1 / positive_frequencies[idx]) if positive_frequencies[idx] > 0 else np.inf
            })
        
        # Sort by power (descending)
        dominant_frequencies.sort(key=lambda x: x['power'], reverse=True)
        
        return {
            'frequencies': positive_frequencies.tolist(),
            'power_spectrum': positive_power.tolist(),
            'dominant_frequencies': dominant_frequencies[:10],  # Top 10
            'total_power': float(np.sum(positive_power)),
            'peak_frequency': float(positive_frequencies[np.argmax(positive_power)]),
            'bandwidth': float(positive_frequencies[-1] - positive_frequencies[0]),
            'sampling_rate': sampling_rate
        }
    
    def change_point_detection(self, data: pd.Series, method: str = 'cusum') -> Dict[str, Any]:
        """
        Detect change points in time series data
        
        Args:
            data: Time series data
            method: Detection method ('cusum', 'variance')
            
        Returns:
            Dictionary with change point detection results
        """
        clean_data = data.dropna().values
        n = len(clean_data)
        
        if n < 10:
            raise ValueError("Insufficient data for change point detection")
        
        change_points = []
        
        if method == 'cusum':
            # CUSUM change point detection
            mean_val = np.mean(clean_data)
            std_val = np.std(clean_data)
            
            # Standardize data
            standardized = (clean_data - mean_val) / std_val
            
            # CUSUM statistics
            threshold = 3.0  # Detection threshold
            cusum_pos = np.zeros(n)
            cusum_neg = np.zeros(n)
            
            for i in range(1, n):
                cusum_pos[i] = max(0, cusum_pos[i-1] + standardized[i] - 0.5)
                cusum_neg[i] = max(0, cusum_neg[i-1] - standardized[i] - 0.5)
                
                if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                    change_points.append({
                        'index': i,
                        'timestamp': data.index[i] if hasattr(data.index, '__getitem__') else i,
                        'cusum_positive': float(cusum_pos[i]),
                        'cusum_negative': float(cusum_neg[i]),
                        'type': 'mean_shift'
                    })
        
        elif method == 'variance':
            # Variance change point detection
            window_size = max(5, n // 10)
            variance_ratio_threshold = 2.0
            
            for i in range(window_size, n - window_size):
                # Calculate variance before and after point i
                var_before = np.var(clean_data[i-window_size:i])
                var_after = np.var(clean_data[i:i+window_size])
                
                if var_before > 0 and var_after > 0:
                    var_ratio = max(var_before, var_after) / min(var_before, var_after)
                    
                    if var_ratio > variance_ratio_threshold:
                        change_points.append({
                            'index': i,
                            'timestamp': data.index[i] if hasattr(data.index, '__getitem__') else i,
                            'variance_before': float(var_before),
                            'variance_after': float(var_after),
                            'variance_ratio': float(var_ratio),
                            'type': 'variance_change'
                        })
        
        return {
            'change_points': change_points,
            'method': method,
            'total_change_points': len(change_points),
            'data_length': n
        }
    
    def outlier_detection(self, data: pd.Series, methods: List[str] = None) -> Dict[str, Any]:
        """
        Detect outliers using multiple statistical methods
        
        Args:
            data: Series data
            methods: List of methods to use ('zscore', 'iqr', 'isolation', 'modified_zscore')
            
        Returns:
            Dictionary with outlier detection results
        """
        if methods is None:
            methods = ['zscore', 'iqr', 'modified_zscore']
        
        clean_data = data.dropna()
        results = {}
        
        for method in methods:
            outliers = []
            
            if method == 'zscore':
                # Z-score method
                z_scores = np.abs(stats.zscore(clean_data))
                threshold = 3.0
                outlier_mask = z_scores > threshold
                
                for idx in clean_data.index[outlier_mask]:
                    outliers.append({
                        'index': idx,
                        'value': float(clean_data[idx]),
                        'z_score': float(z_scores[clean_data.index.get_loc(idx)]),
                        'threshold': threshold
                    })
            
            elif method == 'iqr':
                # Interquartile range method
                Q1 = clean_data.quantile(0.25)
                Q3 = clean_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (clean_data < lower_bound) | (clean_data > upper_bound)
                
                for idx in clean_data.index[outlier_mask]:
                    outliers.append({
                        'index': idx,
                        'value': float(clean_data[idx]),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'iqr': float(IQR)
                    })
            
            elif method == 'modified_zscore':
                # Modified Z-score using median
                median = np.median(clean_data)
                mad = np.median(np.abs(clean_data - median))  # Median absolute deviation
                modified_z_scores = 0.6745 * (clean_data - median) / mad if mad > 0 else np.zeros_like(clean_data)
                threshold = 3.5
                outlier_mask = np.abs(modified_z_scores) > threshold
                
                for idx in clean_data.index[outlier_mask]:
                    outliers.append({
                        'index': idx,
                        'value': float(clean_data[idx]),
                        'modified_z_score': float(modified_z_scores[clean_data.index.get_loc(idx)]),
                        'threshold': threshold
                    })
            
            results[method] = {
                'outliers': outliers,
                'count': len(outliers),
                'percentage': (len(outliers) / len(clean_data)) * 100
            }
        
        # Consensus outliers (detected by multiple methods)
        all_outlier_indices = set()
        for method_result in results.values():
            all_outlier_indices.update([o['index'] for o in method_result['outliers']])
        
        consensus_outliers = []
        for idx in all_outlier_indices:
            detection_count = sum(1 for method_result in results.values() 
                                if idx in [o['index'] for o in method_result['outliers']])
            if detection_count > 1:
                consensus_outliers.append({
                    'index': idx,
                    'value': float(clean_data[idx]),
                    'detected_by_methods': detection_count,
                    'methods': [method for method, method_result in results.items() 
                              if idx in [o['index'] for o in method_result['outliers']]]
                })
        
        results['consensus'] = {
            'outliers': consensus_outliers,
            'count': len(consensus_outliers)
        }
        
        self.logger.info(f"Outlier detection completed using methods: {methods}")
        return results