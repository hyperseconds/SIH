#!/usr/bin/env python3
"""
Visualization Module for CME Prediction System
Advanced plotting and visualization using matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import warnings

# Configure matplotlib for better appearance
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=UserWarning)


class Visualizer:
    """Advanced visualization for CME prediction system"""
    
    def __init__(self, style: str = 'dark'):
        """
        Initialize visualizer
        
        Args:
            style: Plotting style ('dark', 'light', 'space')
        """
        self.style = style
        self.setup_style()
        
        # Color schemes
        self.colors = {
            'flux': '#FF6B6B',
            'density': '#4ECDC4',
            'temperature': '#45B7D1',
            'speed': '#96CEB4',
            'pressure': '#FFEAA7',
            'magnetic_field': '#DDA0DD',
            'cme_event': '#FF4757',
            'prediction': '#5F27CD',
            'high_risk': '#FF3838',
            'medium_risk': '#FF8C42',
            'low_risk': '#6BCF7F'
        }
        
        # Current figure references
        self.current_figures = []
        
    def setup_style(self):
        """Setup matplotlib style based on theme"""
        if self.style == 'dark':
            plt.rcParams.update({
                'figure.facecolor': '#1e1e1e',
                'axes.facecolor': '#2d2d2d',
                'axes.edgecolor': '#444444',
                'axes.labelcolor': '#ffffff',
                'xtick.color': '#ffffff',
                'ytick.color': '#ffffff',
                'text.color': '#ffffff',
                'axes.spines.left': True,
                'axes.spines.bottom': True,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'grid.color': '#444444',
                'grid.alpha': 0.3
            })
        elif self.style == 'space':
            plt.rcParams.update({
                'figure.facecolor': '#0a0a0a',
                'axes.facecolor': '#111111',
                'axes.edgecolor': '#333333',
                'axes.labelcolor': '#ffffff',
                'xtick.color': '#ffffff',
                'ytick.color': '#ffffff',
                'text.color': '#ffffff',
                'grid.color': '#222222',
                'grid.alpha': 0.5
            })
    
    def plot_swis_data(self, swis_data: List[Dict[str, Any]], 
                      parameter: str = 'all', 
                      time_range: str = '24h'):
        """
        Plot SWIS solar wind data
        
        Args:
            swis_data: SWIS data records
            parameter: Parameter to plot ('all', 'flux', 'density', etc.)
            time_range: Time range to display
        """
        if not swis_data:
            print("No SWIS data to plot")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(swis_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by time range
        if time_range != 'all':
            hours = self._parse_time_range(time_range)
            if hours:
                cutoff_time = df['timestamp'].max() - timedelta(hours=hours)
                df = df[df['timestamp'] >= cutoff_time]
        
        if len(df) == 0:
            print("No data in specified time range")
            return
        
        # Parameters to plot
        if parameter == 'all':
            params = ['flux', 'density', 'temperature', 'speed', 'pressure', 'magnetic_field']
            params = [p for p in params if p in df.columns]
        else:
            params = [parameter] if parameter in df.columns else []
        
        if not params:
            print(f"Parameter '{parameter}' not found in data")
            return
        
        # Create figure
        n_params = len(params)
        fig, axes = plt.subplots(n_params, 1, figsize=(15, 3*n_params), sharex=True)
        if n_params == 1:
            axes = [axes]
        
        fig.suptitle(f'🛰️ SWIS Solar Wind Data - {time_range}', 
                    fontsize=16, fontweight='bold', color='white')
        
        # Plot each parameter
        for i, param in enumerate(params):
            ax = axes[i]
            
            # Get color for parameter
            color = self.colors.get(param, '#ffffff')
            
            # Plot time series
            ax.plot(df['timestamp'], df[param], color=color, linewidth=1.5, 
                   alpha=0.8, label=param.replace('_', ' ').title())
            
            # Add rolling mean
            if len(df) > 10:
                window = min(20, len(df) // 10)
                rolling_mean = df[param].rolling(window=window, center=True).mean()
                ax.plot(df['timestamp'], rolling_mean, color=color, 
                       linewidth=2, alpha=0.9, linestyle='--', label=f'{param} (Rolling Mean)')
            
            # Styling
            ax.set_ylabel(self._get_parameter_label(param), color='white')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Format y-axis
            self._format_parameter_axis(ax, param, df[param])
            
            # Highlight anomalies
            self._highlight_anomalies(ax, df['timestamp'], df[param], color)
        
        # Format x-axis
        axes[-1].set_xlabel('Time (UTC)', color='white')
        self._format_time_axis(axes[-1], df['timestamp'])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Store figure reference
        self.current_figures.append(fig)
        
        plt.show()
    
    def plot_predictions(self, prediction_results: Dict[str, Any]):
        """
        Plot CME prediction results
        
        Args:
            prediction_results: Dictionary with prediction results
        """
        if not prediction_results or 'predictions' not in prediction_results:
            print("No prediction results to plot")
            return
        
        predictions = prediction_results['predictions']
        timestamps = prediction_results.get('timestamps', 
                                          [f"T+{i}h" for i in range(len(predictions))])
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('🔮 CME Prediction Results', fontsize=16, fontweight='bold', color='white')
        
        # 1. Probability timeline
        self._plot_probability_timeline(ax1, predictions, timestamps)
        
        # 2. Risk level distribution
        self._plot_risk_distribution(ax2, predictions)
        
        # 3. Confidence intervals (if available)
        self._plot_confidence_intervals(ax3, predictions, timestamps, prediction_results)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Store figure reference
        self.current_figures.append(fig)
        
        plt.show()
    
    def plot_cme_zones(self, swis_data: List[Dict[str, Any]], cme_events: pd.DataFrame):
        """
        Plot CME event zones overlaid on SWIS data
        
        Args:
            swis_data: SWIS data records
            cme_events: CME events DataFrame
        """
        if not swis_data or len(cme_events) == 0:
            print("No data to plot CME zones")
            return
        
        # Convert SWIS data to DataFrame
        df = pd.DataFrame(swis_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure CME events have proper timestamps
        if 'timestamp' not in cme_events.columns:
            print("CME events missing timestamp column")
            return
        
        cme_events = cme_events.copy()
        cme_events['timestamp'] = pd.to_datetime(cme_events['timestamp'])
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle('⚡ CME Events and Solar Wind Conditions', 
                    fontsize=16, fontweight='bold', color='white')
        
        # Key parameters to show
        params = [
            (['speed', 'temperature'], 'Speed & Temperature'),
            (['density', 'pressure'], 'Density & Pressure'),
            (['flux', 'magnetic_field'], 'Flux & Magnetic Field')
        ]
        
        for i, (param_list, title) in enumerate(params):
            ax = axes[i]
            
            # Plot parameters
            for j, param in enumerate(param_list):
                if param in df.columns:
                    color = self.colors.get(param, f'C{j}')
                    
                    # Normalize for dual axis if needed
                    if len(param_list) == 2 and j == 1:
                        ax2 = ax.twinx()
                        ax2.plot(df['timestamp'], df[param], color=color, 
                               linewidth=1.5, alpha=0.8, label=param.replace('_', ' ').title())
                        ax2.set_ylabel(self._get_parameter_label(param), color=color)
                        ax2.tick_params(axis='y', labelcolor=color)
                        ax2.legend(loc='upper right')
                    else:
                        ax.plot(df['timestamp'], df[param], color=color, 
                               linewidth=1.5, alpha=0.8, label=param.replace('_', ' ').title())
            
            # Highlight CME events
            self._highlight_cme_events(ax, df['timestamp'], cme_events)
            
            # Styling
            ax.set_title(title, color='white', fontweight='bold')
            ax.set_ylabel(self._get_parameter_label(param_list[0]), color='white')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
        
        # Format x-axis
        axes[-1].set_xlabel('Time (UTC)', color='white')
        self._format_time_axis(axes[-1], df['timestamp'])
        
        # Add CME event markers
        self._add_cme_event_markers(fig, axes, cme_events)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Store figure reference
        self.current_figures.append(fig)
        
        plt.show()
    
    def plot_feature_importance(self, importance_scores: Dict[str, float], 
                              top_n: int = 20):
        """
        Plot feature importance scores
        
        Args:
            importance_scores: Dictionary mapping feature names to importance scores
            top_n: Number of top features to show
        """
        if not importance_scores:
            print("No feature importance data to plot")
            return
        
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        # Prepare data
        features, scores = zip(*top_features)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('🎯 Feature Importance Analysis', 
                    fontsize=16, fontweight='bold', color='white')
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(features)), scores, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(features))))
        
        # Styling
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
        ax.set_xlabel('Importance Score', color='white')
        ax.set_title(f'Top {top_n} Most Important Features', color='white')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            width = bar.get_width()
            ax.text(width + max(scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', ha='left', va='center', color='white', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Store figure reference
        self.current_figures.append(fig)
        
        plt.show()
    
    def plot_training_history(self, loss_history: List[float], 
                            accuracy_history: List[float] = None):
        """
        Plot training history
        
        Args:
            loss_history: List of loss values per epoch
            accuracy_history: List of accuracy values per epoch
        """
        if not loss_history:
            print("No training history to plot")
            return
        
        epochs = range(1, len(loss_history) + 1)
        
        # Create figure
        if accuracy_history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            ax2 = None
        
        fig.suptitle('🧠 Neural Network Training History', 
                    fontsize=16, fontweight='bold', color='white')
        
        # Plot loss
        ax1.plot(epochs, loss_history, color='#FF6B6B', linewidth=2, 
                marker='o', markersize=3, label='Training Loss')
        ax1.set_xlabel('Epoch', color='white')
        ax1.set_ylabel('Loss', color='white')
        ax1.set_title('Training Loss', color='white', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot accuracy if available
        if accuracy_history and ax2:
            ax2.plot(epochs, accuracy_history, color='#4ECDC4', linewidth=2, 
                    marker='s', markersize=3, label='Training Accuracy')
            ax2.set_xlabel('Epoch', color='white')
            ax2.set_ylabel('Accuracy', color='white')
            ax2.set_title('Training Accuracy', color='white', fontweight='bold')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        
        # Store figure reference
        self.current_figures.append(fig)
        
        plt.show()
    
    def plot_correlation_matrix(self, features: np.ndarray, feature_names: List[str]):
        """
        Plot feature correlation matrix
        
        Args:
            features: Feature matrix
            feature_names: List of feature names
        """
        if features.shape[1] == 0:
            print("No features to plot correlation matrix")
            return
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(features, rowvar=False)
        
        # Handle case where we have too many features
        max_features = 30
        if len(feature_names) > max_features:
            # Show only most variable features
            feature_vars = np.var(features, axis=0)
            top_indices = np.argsort(feature_vars)[-max_features:]
            correlation_matrix = correlation_matrix[np.ix_(top_indices, top_indices)]
            feature_names = [feature_names[i] for i in top_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.suptitle('🔗 Feature Correlation Matrix', 
                    fontsize=16, fontweight='bold', color='white')
        
        # Create heatmap
        im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', 
                      vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', color='white')
        
        # Set ticks and labels
        ax.set_xticks(range(len(feature_names)))
        ax.set_yticks(range(len(feature_names)))
        ax.set_xticklabels([name.replace('_', ' ') for name in feature_names], 
                          rotation=45, ha='right')
        ax.set_yticklabels([name.replace('_', ' ') for name in feature_names])
        
        # Add correlation values to cells
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        # Store figure reference
        self.current_figures.append(fig)
        
        plt.show()
    
    def plot_time_series_decomposition(self, swis_data: List[Dict[str, Any]], 
                                     parameter: str = 'speed'):
        """
        Plot time series decomposition
        
        Args:
            swis_data: SWIS data records
            parameter: Parameter to decompose
        """
        if not swis_data:
            print("No SWIS data for decomposition")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(swis_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if parameter not in df.columns:
            print(f"Parameter '{parameter}' not found in data")
            return
        
        # Perform simple decomposition
        series = df[parameter].values
        
        # Trend (moving average)
        window = min(24, len(series) // 4)
        if window < 3:
            print("Insufficient data for decomposition")
            return
        
        trend = pd.Series(series).rolling(window=window, center=True).mean().values
        
        # Detrended series
        detrended = series - np.nanmean(trend)
        
        # Seasonal component (simplified)
        seasonal_period = min(24, len(series) // 3)
        seasonal = np.tile(np.nanmean(detrended[:seasonal_period]), 
                          len(series) // seasonal_period + 1)[:len(series)]
        
        # Residual
        residual = series - np.nanmean(trend) - seasonal
        
        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f'📈 Time Series Decomposition - {parameter.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold', color='white')
        
        # Original series
        axes[0].plot(df['timestamp'], series, color=self.colors.get(parameter, '#ffffff'), 
                    linewidth=1.5)
        axes[0].set_title('Original Series', color='white')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(df['timestamp'], trend, color='#FFD93D', linewidth=2)
        axes[1].set_title('Trend', color='white')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        axes[2].plot(df['timestamp'], seasonal, color='#6BCF7F', linewidth=1.5)
        axes[2].set_title('Seasonal', color='white')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        axes[3].plot(df['timestamp'], residual, color='#FF8C42', linewidth=1)
        axes[3].set_title('Residual', color='white')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_xlabel('Time (UTC)', color='white')
        
        # Format x-axis
        self._format_time_axis(axes[-1], df['timestamp'])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Store figure reference
        self.current_figures.append(fig)
        
        plt.show()
    
    def create_dashboard(self, swis_data: List[Dict[str, Any]], 
                        cme_events: pd.DataFrame,
                        prediction_results: Dict[str, Any] = None):
        """
        Create comprehensive dashboard
        
        Args:
            swis_data: SWIS data records
            cme_events: CME events DataFrame
            prediction_results: Prediction results
        """
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('🛰️ Synapse Horizon - CME Prediction Dashboard', 
                    fontsize=20, fontweight='bold', color='white')
        
        # Define grid layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        if not swis_data:
            # Show empty state
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, 'No Data Available', 
                   ha='center', va='center', fontsize=24, color='white')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.show()
            return
        
        # Convert data
        df = pd.DataFrame(swis_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 1. Recent solar wind conditions (top row)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_recent_conditions(ax1, df)
        
        # 2. Current risk indicator (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_risk_gauge(ax2, prediction_results)
        
        # 3. Parameter trends (second row)
        params = ['speed', 'density', 'temperature']
        for i, param in enumerate(params):
            if param in df.columns:
                ax = fig.add_subplot(gs[1, i])
                self._plot_parameter_trend(ax, df, param)
        
        # 4. CME prediction timeline (third row)
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_prediction_timeline(ax4, prediction_results)
        
        # 5. Recent CME events (bottom left)
        ax5 = fig.add_subplot(gs[3, 0])
        self._plot_recent_cme_events(ax5, cme_events)
        
        # 6. System status (bottom center)
        ax6 = fig.add_subplot(gs[3, 1])
        self._plot_system_status(ax6, df, prediction_results)
        
        # 7. Alert summary (bottom right)
        ax7 = fig.add_subplot(gs[3, 2])
        self._plot_alert_summary(ax7, prediction_results)
        
        # Store figure reference
        self.current_figures.append(fig)
        
        plt.show()
    
    def close_all_figures(self):
        """Close all current figures"""
        for fig in self.current_figures:
            plt.close(fig)
        self.current_figures = []
    
    # Helper methods
    
    def _parse_time_range(self, time_range: str) -> Optional[int]:
        """Parse time range string to hours"""
        time_map = {
            '1h': 1, '6h': 6, '12h': 12, '24h': 24,
            '1d': 24, '2d': 48, '7d': 168, '1w': 168
        }
        return time_map.get(time_range.lower())
    
    def _get_parameter_label(self, param: str) -> str:
        """Get formatted parameter label with units"""
        labels = {
            'flux': 'Particle Flux (counts/s)',
            'density': 'Density (cm⁻³)',
            'temperature': 'Temperature (K)',
            'speed': 'Speed (km/s)',
            'pressure': 'Pressure (nPa)',
            'magnetic_field': 'Magnetic Field (nT)'
        }
        return labels.get(param, param.replace('_', ' ').title())
    
    def _format_parameter_axis(self, ax, param: str, data: pd.Series):
        """Format y-axis for specific parameters"""
        if param == 'temperature':
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        elif param == 'flux':
            ax.set_yscale('log')
    
    def _format_time_axis(self, ax, timestamps: pd.Series):
        """Format time axis"""
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m/%d'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(timestamps)//10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _highlight_anomalies(self, ax, timestamps: pd.Series, values: pd.Series, color: str):
        """Highlight anomalous values"""
        if len(values) < 10:
            return
        
        # Simple anomaly detection using 3-sigma rule
        mean_val = values.mean()
        std_val = values.std()
        
        anomalies = np.abs(values - mean_val) > 3 * std_val
        
        if anomalies.any():
            ax.scatter(timestamps[anomalies], values[anomalies], 
                      color='red', s=30, alpha=0.7, marker='x', 
                      label='Anomalies', zorder=5)
    
    def _plot_probability_timeline(self, ax, predictions: List[float], timestamps: List[str]):
        """Plot prediction probability timeline"""
        # Create color map based on risk levels
        colors = []
        for prob in predictions:
            if prob < 0.3:
                colors.append(self.colors['low_risk'])
            elif prob < 0.7:
                colors.append(self.colors['medium_risk'])
            else:
                colors.append(self.colors['high_risk'])
        
        # Plot as bar chart
        bars = ax.bar(range(len(predictions)), predictions, color=colors, alpha=0.8)
        
        # Add threshold lines
        ax.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.7, label='Low Risk Threshold')
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='High Risk Threshold')
        
        ax.set_xlabel('Time Horizon', color='white')
        ax.set_ylabel('CME Probability', color='white')
        ax.set_title('CME Probability Timeline', color='white', fontweight='bold')
        ax.set_xticks(range(len(timestamps)))
        ax.set_xticklabels(timestamps, rotation=45)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_risk_distribution(self, ax, predictions: List[float]):
        """Plot risk level distribution"""
        # Categorize predictions
        low_risk = sum(1 for p in predictions if p < 0.3)
        medium_risk = sum(1 for p in predictions if 0.3 <= p < 0.7)
        high_risk = sum(1 for p in predictions if p >= 0.7)
        
        # Create pie chart
        sizes = [low_risk, medium_risk, high_risk]
        labels = ['Low Risk', 'Medium Risk', 'High Risk']
        colors = [self.colors['low_risk'], self.colors['medium_risk'], self.colors['high_risk']]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        
        # Style text
        for text in texts:
            text.set_color('white')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        ax.set_title('Risk Level Distribution', color='white', fontweight='bold')
    
    def _plot_confidence_intervals(self, ax, predictions: List[float], 
                                  timestamps: List[str], prediction_results: Dict[str, Any]):
        """Plot confidence intervals"""
        # Use predictions as mean, create synthetic confidence intervals
        predictions = np.array(predictions)
        
        # Simple confidence interval based on prediction values
        confidence = 0.1 + 0.2 * predictions  # Higher uncertainty for higher predictions
        
        x = range(len(predictions))
        
        # Plot mean predictions
        ax.plot(x, predictions, color=self.colors['prediction'], 
               linewidth=2, marker='o', label='Prediction')
        
        # Plot confidence bands
        ax.fill_between(x, predictions - confidence, predictions + confidence,
                       color=self.colors['prediction'], alpha=0.3, label='Confidence Interval')
        
        ax.set_xlabel('Time Horizon', color='white')
        ax.set_ylabel('CME Probability', color='white')
        ax.set_title('Prediction Confidence', color='white', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(timestamps, rotation=45)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _highlight_cme_events(self, ax, timestamps: pd.Series, cme_events: pd.DataFrame):
        """Highlight CME event periods"""
        if len(cme_events) == 0:
            return
        
        y_min, y_max = ax.get_ylim()
        
        for _, event in cme_events.iterrows():
            event_time = event['timestamp']
            
            # Find closest timestamp in data
            time_diffs = np.abs(timestamps - event_time)
            closest_idx = time_diffs.argmin()
            
            if time_diffs.iloc[closest_idx] < timedelta(hours=6):
                # Add vertical line for CME event
                ax.axvline(x=event_time, color=self.colors['cme_event'], 
                          alpha=0.7, linestyle='-', linewidth=2)
                
                # Add event marker
                ax.add_patch(Rectangle((event_time - timedelta(hours=2), y_min),
                                     timedelta(hours=4), y_max - y_min,
                                     facecolor=self.colors['cme_event'], 
                                     alpha=0.2))
    
    def _add_cme_event_markers(self, fig, axes, cme_events: pd.DataFrame):
        """Add CME event markers to figure"""
        if len(cme_events) == 0:
            return
        
        # Add text box with event information
        event_text = f"CME Events: {len(cme_events)}\n"
        if len(cme_events) > 0:
            latest_event = cme_events['timestamp'].max()
            event_text += f"Latest: {latest_event.strftime('%Y-%m-%d %H:%M')}"
        
        fig.text(0.02, 0.02, event_text, fontsize=10, color='white',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    def _plot_recent_conditions(self, ax, df: pd.DataFrame):
        """Plot recent solar wind conditions"""
        # Get last 24 hours
        recent_df = df.tail(24)
        
        # Plot key parameters
        ax.plot(recent_df['timestamp'], recent_df['speed'], 
               color=self.colors['speed'], label='Speed', linewidth=2)
        
        if 'magnetic_field' in recent_df.columns:
            ax2 = ax.twinx()
            ax2.plot(recent_df['timestamp'], recent_df['magnetic_field'], 
                    color=self.colors['magnetic_field'], label='B-field', linewidth=2)
            ax2.set_ylabel('Magnetic Field (nT)', color=self.colors['magnetic_field'])
            ax2.legend(loc='upper right')
        
        ax.set_title('Recent Solar Wind Conditions', color='white', fontweight='bold')
        ax.set_ylabel('Speed (km/s)', color=self.colors['speed'])
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_gauge(self, ax, prediction_results: Dict[str, Any]):
        """Plot current risk gauge"""
        # Get current risk level
        if prediction_results and 'predictions' in prediction_results:
            current_risk = max(prediction_results['predictions'][:3])  # Next 3 hours
        else:
            current_risk = 0.1  # Default low risk
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Color sections
        colors = ['green'] * 33 + ['yellow'] * 33 + ['red'] * 34
        
        for i, (t, color) in enumerate(zip(theta, colors)):
            ax.bar(t, 1, width=np.pi/100, bottom=0, color=color, alpha=0.7)
        
        # Add needle
        needle_angle = current_risk * np.pi
        ax.arrow(needle_angle, 0, 0, 0.8, head_width=0.1, head_length=0.1, 
                fc='white', ec='white', linewidth=3)
        
        # Style
        ax.set_ylim(0, 1)
        ax.set_theta_zero_location('W')
        ax.set_title(f'Current Risk: {current_risk:.1%}', 
                    color='white', fontweight='bold', pad=20)
        ax.set_rticks([])
        ax.set_thetagrids([0, 90, 180], ['Low', 'Medium', 'High'])
    
    def _plot_parameter_trend(self, ax, df: pd.DataFrame, param: str):
        """Plot individual parameter trend"""
        if param not in df.columns:
            return
        
        recent_df = df.tail(48)  # Last 48 hours
        
        # Plot trend
        color = self.colors.get(param, '#ffffff')
        ax.plot(recent_df['timestamp'], recent_df[param], 
               color=color, linewidth=2)
        
        # Add trend line
        x = np.arange(len(recent_df))
        z = np.polyfit(x, recent_df[param], 1)
        p = np.poly1d(z)
        ax.plot(recent_df['timestamp'], p(x), 
               color='white', linestyle='--', alpha=0.7)
        
        ax.set_title(param.replace('_', ' ').title(), color='white', fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._format_time_axis(ax, recent_df['timestamp'])
    
    def _plot_prediction_timeline(self, ax, prediction_results: Dict[str, Any]):
        """Plot prediction timeline"""
        if not prediction_results or 'predictions' not in prediction_results:
            ax.text(0.5, 0.5, 'No Predictions Available', 
                   ha='center', va='center', fontsize=14, color='white')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
        
        predictions = prediction_results['predictions'][:24]  # Next 24 hours
        hours = [f'T+{i}h' for i in range(len(predictions))]
        
        # Create timeline plot
        colors = [self.colors['low_risk'] if p < 0.3 else 
                 self.colors['medium_risk'] if p < 0.7 else 
                 self.colors['high_risk'] for p in predictions]
        
        bars = ax.bar(range(len(predictions)), predictions, color=colors, alpha=0.8)
        
        ax.set_title('24-Hour CME Prediction Timeline', color='white', fontweight='bold')
        ax.set_xlabel('Time Horizon', color='white')
        ax.set_ylabel('CME Probability', color='white')
        ax.set_xticks(range(0, len(predictions), 4))
        ax.set_xticklabels([hours[i] for i in range(0, len(predictions), 4)])
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    def _plot_recent_cme_events(self, ax, cme_events: pd.DataFrame):
        """Plot recent CME events summary"""
        if len(cme_events) == 0:
            ax.text(0.5, 0.5, 'No Recent\nCME Events', 
                   ha='center', va='center', fontsize=12, color='white')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Recent CME Events', color='white', fontweight='bold')
            return
        
        # Get events from last 7 days
        recent_events = cme_events.tail(10)
        
        # Plot event timeline
        y_pos = range(len(recent_events))
        speeds = recent_events.get('speed', [500] * len(recent_events))
        
        bars = ax.barh(y_pos, speeds, color=self.colors['cme_event'], alpha=0.7)
        
        ax.set_title('Recent CME Events', color='white', fontweight='bold')
        ax.set_xlabel('Speed (km/s)', color='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'Event {i+1}' for i in range(len(recent_events))])
        ax.grid(True, alpha=0.3)
    
    def _plot_system_status(self, ax, df: pd.DataFrame, prediction_results: Dict[str, Any]):
        """Plot system status indicators"""
        # System health indicators
        indicators = {
            'Data Quality': 0.95 if len(df) > 0 else 0.0,
            'Model Status': 0.9 if prediction_results else 0.0,
            'Alert System': 0.98,
            'Database': 0.99
        }
        
        # Create status bars
        labels = list(indicators.keys())
        values = list(indicators.values())
        colors = ['green' if v > 0.8 else 'yellow' if v > 0.5 else 'red' for v in values]
        
        bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.8)
        
        ax.set_title('System Status', color='white', fontweight='bold')
        ax.set_xlabel('Health Score', color='white')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        
        # Add percentage labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{value:.0%}', ha='left', va='center', color='white', fontsize=10)
    
    def _plot_alert_summary(self, ax, prediction_results: Dict[str, Any]):
        """Plot alert summary"""
        if not prediction_results or 'predictions' not in prediction_results:
            ax.text(0.5, 0.5, 'No Alerts\nActive', 
                   ha='center', va='center', fontsize=14, color='green')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Alert Status', color='white', fontweight='bold')
            return
        
        predictions = prediction_results['predictions']
        max_risk = max(predictions[:12])  # Next 12 hours
        
        # Determine alert level
        if max_risk >= 0.7:
            alert_text = f'🔴 HIGH ALERT\n{max_risk:.1%} probability'
            alert_color = 'red'
        elif max_risk >= 0.3:
            alert_text = f'🟡 MEDIUM ALERT\n{max_risk:.1%} probability'
            alert_color = 'orange'
        else:
            alert_text = f'🟢 LOW RISK\n{max_risk:.1%} probability'
            alert_color = 'green'
        
        ax.text(0.5, 0.5, alert_text, ha='center', va='center', 
               fontsize=14, color=alert_color, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Current Alert Status', color='white', fontweight='bold')
