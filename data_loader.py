#!/usr/bin/env python3
"""
Data Loading Module for SWIS Solar Wind Data and CME Events
Handles JSON and CSV data loading with timestamp alignment
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import os
import re


class DataLoader:
    """Data loader for SWIS and CME event data"""
    
    def __init__(self):
        """Initialize data loader"""
        self.swis_data = None
        self.cme_events = None
        self.data_cache = {}
        
    def load_swis_data(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load SWIS solar wind data from JSON file
        
        Args:
            filepath: Path to SWIS data JSON file
            
        Returns:
            List of SWIS data records
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Validate and process data
            processed_data = self._process_swis_data(data)
            self.swis_data = processed_data
            
            print(f"Loaded {len(processed_data)} SWIS records from {filepath}")
            return processed_data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"SWIS data file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in SWIS data file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading SWIS data: {e}")
    
    def load_cme_events(self, filepath: str) -> pd.DataFrame:
        """
        Load CME events data from CSV file
        
        Args:
            filepath: Path to CME events CSV file
            
        Returns:
            DataFrame with CME events
        """
        try:
            # Load CSV data
            df = pd.read_csv(filepath)
            
            # Validate and process data
            processed_df = self._process_cme_data(df)
            self.cme_events = processed_df
            
            print(f"Loaded {len(processed_df)} CME events from {filepath}")
            return processed_df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"CME events file not found: {filepath}")
        except pd.errors.EmptyDataError:
            raise ValueError("CME events file is empty")
        except Exception as e:
            raise RuntimeError(f"Error loading CME events: {e}")
    
    def _process_swis_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """
        Process and validate raw SWIS data
        
        Args:
            raw_data: Raw data from JSON file
            
        Returns:
            Processed SWIS data records
        """
        if isinstance(raw_data, dict) and 'data' in raw_data:
            records = raw_data['data']
        elif isinstance(raw_data, list):
            records = raw_data
        else:
            raise ValueError("Invalid SWIS data format")
        
        processed_records = []
        
        for i, record in enumerate(records):
            try:
                processed_record = self._validate_swis_record(record, i)
                if processed_record:
                    processed_records.append(processed_record)
            except Exception as e:
                print(f"Warning: Skipping invalid SWIS record {i}: {e}")
                continue
        
        if not processed_records:
            raise ValueError("No valid SWIS records found")
        
        # Sort by timestamp
        processed_records.sort(key=lambda x: x['timestamp'])
        
        return processed_records
    
    def _validate_swis_record(self, record: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """
        Validate and normalize a single SWIS record
        
        Args:
            record: Raw SWIS record
            index: Record index for error reporting
            
        Returns:
            Validated record or None if invalid
        """
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} is not a dictionary")
        
        # Required fields
        required_fields = ['timestamp']
        for field in required_fields:
            if field not in record:
                raise ValueError(f"Missing required field '{field}' in record {index}")
        
        # Process timestamp
        timestamp = self._parse_timestamp(record['timestamp'])
        if timestamp is None:
            raise ValueError(f"Invalid timestamp in record {index}")
        
        # Extract and validate numerical parameters
        processed_record = {
            'timestamp': timestamp,
            'flux': self._extract_numeric_value(record, 'flux', 0.0),
            'density': self._extract_numeric_value(record, 'density', 0.0),
            'temperature': self._extract_numeric_value(record, 'temperature', 0.0),
            'speed': self._extract_numeric_value(record, 'speed', 0.0),
            'pressure': self._extract_numeric_value(record, 'pressure', 0.0),
            'magnetic_field': self._extract_numeric_value(record, 'magnetic_field', 0.0)
        }
        
        # Validate ranges
        if not self._validate_swis_ranges(processed_record):
            print(f"Warning: SWIS record {index} has values outside expected ranges")
        
        return processed_record
    
    def _process_cme_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and validate CME events DataFrame
        
        Args:
            df: Raw CME events DataFrame
            
        Returns:
            Processed CME events DataFrame
        """
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Standardize column names
        processed_df.columns = [col.lower().strip() for col in processed_df.columns]
        
        # Required columns
        required_columns = ['timestamp']
        
        # Try to find timestamp column with various names
        timestamp_cols = ['timestamp', 'time', 'datetime', 'date', 'event_time']
        timestamp_col = None
        
        for col in timestamp_cols:
            if col in processed_df.columns:
                timestamp_col = col
                break
        
        if timestamp_col is None:
            raise ValueError("No timestamp column found in CME events data")
        
        # Process timestamps
        processed_df['timestamp'] = processed_df[timestamp_col].apply(self._parse_timestamp)
        
        # Remove records with invalid timestamps
        invalid_timestamps = processed_df['timestamp'].isna()
        if invalid_timestamps.any():
            print(f"Warning: Removing {invalid_timestamps.sum()} CME records with invalid timestamps")
            processed_df = processed_df[~invalid_timestamps]
        
        if len(processed_df) == 0:
            raise ValueError("No valid CME events after timestamp processing")
        
        # Add default columns if missing
        default_columns = {
            'angular_width': 360.0,  # Assume halo CME if not specified
            'speed': 500.0,          # Average CME speed
            'acceleration': 0.0,
            'mass': 1e15,           # Average CME mass
            'energy': 1e32          # Average CME energy
        }
        
        for col, default_value in default_columns.items():
            if col not in processed_df.columns:
                processed_df[col] = default_value
        
        # Sort by timestamp
        processed_df = processed_df.sort_values('timestamp').reset_index(drop=True)
        
        return processed_df
    
    def _parse_timestamp(self, timestamp_str: Any) -> Optional[datetime]:
        """
        Parse timestamp string into datetime object
        
        Args:
            timestamp_str: Timestamp string in various formats
            
        Returns:
            Parsed datetime or None if invalid
        """
        if pd.isna(timestamp_str):
            return None
        
        if isinstance(timestamp_str, datetime):
            return timestamp_str
        
        if not isinstance(timestamp_str, str):
            timestamp_str = str(timestamp_str)
        
        # Common timestamp formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y/%m/%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%Y/%m/%d'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str.strip(), fmt)
            except ValueError:
                continue
        
        # Try pandas timestamp parsing as fallback
        try:
            return pd.to_datetime(timestamp_str)
        except:
            pass
        
        return None
    
    def _extract_numeric_value(self, record: Dict[str, Any], key: str, default: float) -> float:
        """
        Extract numeric value from record with fallback to default
        
        Args:
            record: Data record
            key: Key to extract
            default: Default value if key missing or invalid
            
        Returns:
            Numeric value
        """
        if key not in record:
            return default
        
        value = record[key]
        
        if pd.isna(value):
            return default
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _validate_swis_ranges(self, record: Dict[str, Any]) -> bool:
        """
        Validate SWIS parameter ranges
        
        Args:
            record: SWIS record to validate
            
        Returns:
            True if all values are within expected ranges
        """
        ranges = {
            'flux': (0, 1e10),
            'density': (0, 100),
            'temperature': (0, 1e7),
            'speed': (200, 1000),
            'pressure': (0, 100),
            'magnetic_field': (0, 100)
        }
        
        for param, (min_val, max_val) in ranges.items():
            if param in record:
                value = record[param]
                if not (min_val <= value <= max_val):
                    return False
        
        return True
    
    def align_timestamps(self, swis_data: List[Dict[str, Any]], 
                        cme_events: pd.DataFrame, 
                        tolerance_hours: int = 6) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        Align SWIS data with CME events based on timestamps
        
        Args:
            swis_data: SWIS data records
            cme_events: CME events DataFrame
            tolerance_hours: Time tolerance for alignment in hours
            
        Returns:
            Tuple of aligned SWIS data and CME events
        """
        if not swis_data or len(cme_events) == 0:
            return swis_data, cme_events
        
        # Convert to timestamps
        swis_timestamps = [record['timestamp'] for record in swis_data]
        cme_timestamps = cme_events['timestamp'].tolist()
        
        # Find time range overlap
        swis_start = min(swis_timestamps)
        swis_end = max(swis_timestamps)
        cme_start = min(cme_timestamps)
        cme_end = max(cme_timestamps)
        
        # Determine overlap period
        overlap_start = max(swis_start, cme_start)
        overlap_end = min(swis_end, cme_end)
        
        if overlap_start >= overlap_end:
            print("Warning: No time overlap between SWIS data and CME events")
            return swis_data, cme_events
        
        # Filter data to overlap period with tolerance
        tolerance = timedelta(hours=tolerance_hours)
        filter_start = overlap_start - tolerance
        filter_end = overlap_end + tolerance
        
        # Filter SWIS data
        aligned_swis = [
            record for record in swis_data
            if filter_start <= record['timestamp'] <= filter_end
        ]
        
        # Filter CME events
        aligned_cme = cme_events[
            (cme_events['timestamp'] >= filter_start) &
            (cme_events['timestamp'] <= filter_end)
        ].copy()
        
        print(f"Aligned data: {len(aligned_swis)} SWIS records, {len(aligned_cme)} CME events")
        print(f"Time range: {filter_start} to {filter_end}")
        
        return aligned_swis, aligned_cme
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded data
        
        Returns:
            Dictionary with data statistics
        """
        stats = {}
        
        if self.swis_data:
            swis_stats = self._calculate_swis_statistics(self.swis_data)
            stats['swis'] = swis_stats
        
        if self.cme_events is not None:
            cme_stats = self._calculate_cme_statistics(self.cme_events)
            stats['cme'] = cme_stats
        
        return stats
    
    def _calculate_swis_statistics(self, swis_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for SWIS data"""
        if not swis_data:
            return {}
        
        # Extract numerical parameters
        params = ['flux', 'density', 'temperature', 'speed', 'pressure', 'magnetic_field']
        
        stats = {
            'total_records': len(swis_data),
            'time_range': {
                'start': swis_data[0]['timestamp'].isoformat(),
                'end': swis_data[-1]['timestamp'].isoformat()
            },
            'parameters': {}
        }
        
        for param in params:
            values = [record.get(param, 0) for record in swis_data]
            values = [v for v in values if v is not None and not pd.isna(v)]
            
            if values:
                stats['parameters'][param] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'valid_count': len(values)
                }
        
        return stats
    
    def _calculate_cme_statistics(self, cme_events: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for CME events"""
        if len(cme_events) == 0:
            return {}
        
        stats = {
            'total_events': len(cme_events),
            'time_range': {
                'start': cme_events['timestamp'].min().isoformat(),
                'end': cme_events['timestamp'].max().isoformat()
            }
        }
        
        # Statistics for numerical columns
        numerical_cols = cme_events.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in cme_events.columns:
                values = cme_events[col].dropna()
                if len(values) > 0:
                    stats[col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'valid_count': len(values)
                    }
        
        return stats
    
    def export_processed_data(self, swis_filepath: str, cme_filepath: str):
        """
        Export processed data to files
        
        Args:
            swis_filepath: Path to save processed SWIS data
            cme_filepath: Path to save processed CME events
        """
        if self.swis_data:
            # Convert timestamps to ISO format for JSON serialization
            swis_export = []
            for record in self.swis_data:
                export_record = record.copy()
                export_record['timestamp'] = record['timestamp'].isoformat()
                swis_export.append(export_record)
            
            with open(swis_filepath, 'w') as f:
                json.dump(swis_export, f, indent=2)
            
            print(f"Exported {len(swis_export)} SWIS records to {swis_filepath}")
        
        if self.cme_events is not None:
            # Convert timestamps for CSV export
            cme_export = self.cme_events.copy()
            cme_export['timestamp'] = cme_export['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            cme_export.to_csv(cme_filepath, index=False)
            print(f"Exported {len(cme_export)} CME events to {cme_filepath}")
    
    def validate_data_integrity(self) -> Dict[str, List[str]]:
        """
        Validate data integrity and return issues found
        
        Returns:
            Dictionary with validation issues
        """
        issues = {
            'swis': [],
            'cme': [],
            'alignment': []
        }
        
        # Validate SWIS data
        if self.swis_data:
            issues['swis'].extend(self._validate_swis_integrity(self.swis_data))
        
        # Validate CME events
        if self.cme_events is not None:
            issues['cme'].extend(self._validate_cme_integrity(self.cme_events))
        
        # Validate alignment
        if self.swis_data and self.cme_events is not None:
            issues['alignment'].extend(self._validate_alignment_integrity())
        
        return issues
    
    def _validate_swis_integrity(self, swis_data: List[Dict[str, Any]]) -> List[str]:
        """Validate SWIS data integrity"""
        issues = []
        
        if not swis_data:
            issues.append("No SWIS data loaded")
            return issues
        
        # Check for time gaps
        timestamps = [record['timestamp'] for record in swis_data]
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i-1]
            if gap.total_seconds() > 7200:  # 2 hours
                issues.append(f"Large time gap detected: {gap} between records {i-1} and {i}")
        
        # Check for duplicate timestamps
        timestamp_counts = {}
        for i, ts in enumerate(timestamps):
            if ts in timestamp_counts:
                issues.append(f"Duplicate timestamp found: {ts} (records {timestamp_counts[ts]} and {i})")
            else:
                timestamp_counts[ts] = i
        
        # Check for invalid values
        for i, record in enumerate(swis_data):
            for param, value in record.items():
                if param != 'timestamp' and (pd.isna(value) or value < 0):
                    issues.append(f"Invalid {param} value in record {i}: {value}")
        
        return issues
    
    def _validate_cme_integrity(self, cme_events: pd.DataFrame) -> List[str]:
        """Validate CME events integrity"""
        issues = []
        
        if len(cme_events) == 0:
            issues.append("No CME events loaded")
            return issues
        
        # Check for duplicate timestamps
        duplicate_timestamps = cme_events['timestamp'].duplicated()
        if duplicate_timestamps.any():
            issues.append(f"Found {duplicate_timestamps.sum()} duplicate CME event timestamps")
        
        # Check for reasonable CME parameters
        if 'angular_width' in cme_events.columns:
            invalid_width = (cme_events['angular_width'] < 0) | (cme_events['angular_width'] > 360)
            if invalid_width.any():
                issues.append(f"Found {invalid_width.sum()} CME events with invalid angular width")
        
        if 'speed' in cme_events.columns:
            invalid_speed = (cme_events['speed'] < 0) | (cme_events['speed'] > 3000)
            if invalid_speed.any():
                issues.append(f"Found {invalid_speed.sum()} CME events with unrealistic speed")
        
        return issues
    
    def _validate_alignment_integrity(self) -> List[str]:
        """Validate data alignment integrity"""
        issues = []
        
        if not self.swis_data or len(self.cme_events) == 0:
            issues.append("Cannot validate alignment: missing data")
            return issues
        
        # Check time range coverage
        swis_start = min(record['timestamp'] for record in self.swis_data)
        swis_end = max(record['timestamp'] for record in self.swis_data)
        cme_start = self.cme_events['timestamp'].min()
        cme_end = self.cme_events['timestamp'].max()
        
        if swis_end < cme_start or cme_end < swis_start:
            issues.append("No time overlap between SWIS data and CME events")
        elif swis_start > cme_start:
            issues.append(f"SWIS data starts after first CME event: {swis_start} > {cme_start}")
        elif swis_end < cme_end:
            issues.append(f"SWIS data ends before last CME event: {swis_end} < {cme_end}")
        
        return issues
