#!/usr/bin/env python3
"""
Advanced Database Management Module for CME Prediction System
PostgreSQL database operations using psycopg2 with advanced features
"""

import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import os
import threading
from contextlib import contextmanager
import logging


class DatabaseManager:
    """Advanced PostgreSQL database manager for CME prediction data"""
    
    def __init__(self, database_url: str = None):
        """
        Initialize database manager with PostgreSQL connection pool
        
        Args:
            database_url: PostgreSQL connection URL
        """
        

        self.database_url = database_url or os.getenv("DATABASE_URL", "postgresql://postgres:Priya2308@localhost:5432/solar_data")

        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.connection_pool = None
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self._init_connection_pool()
    
    def _init_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.connection_pool = SimpleConnectionPool(
                minconn=2,
                maxconn=20,
                dsn=self.database_url
            )
            self.logger.info("PostgreSQL connection pool initialized")
        except psycopg2.Error as e:
            raise RuntimeError(f"Failed to create PostgreSQL connection pool: {e}")
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with context manager"""
        connection = None
        try:
            connection = self.connection_pool.getconn()
            if connection:
                connection.autocommit = False
                yield connection
        except psycopg2.Error as e:
            if connection:
                connection.rollback()
            self.logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if connection:
                self.connection_pool.putconn(connection)
    
    def initialize_tables(self):
        """Create all necessary PostgreSQL database tables and schemas"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    # Create schema
                    cursor.execute("CREATE SCHEMA IF NOT EXISTS synapse_horizon")
                    
                    # SWIS data table with advanced features
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS synapse_horizon.swis_data (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMPTZ NOT NULL UNIQUE,
                            flux DOUBLE PRECISION DEFAULT 0.0,
                            density DOUBLE PRECISION DEFAULT 0.0,
                            temperature DOUBLE PRECISION DEFAULT 0.0,
                            speed DOUBLE PRECISION DEFAULT 0.0,
                            pressure DOUBLE PRECISION DEFAULT 0.0,
                            magnetic_field DOUBLE PRECISION DEFAULT 0.0,
                            data_quality_score DOUBLE PRECISION DEFAULT 1.0,
                            processing_flags INTEGER DEFAULT 0,
                            source_instrument VARCHAR(50) DEFAULT 'SWIS-ASPEX',
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    
                    # Create indices
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_swis_timestamp 
                        ON synapse_horizon.swis_data(timestamp)
                    """)
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_swis_quality 
                        ON synapse_horizon.swis_data(data_quality_score)
                    """)
                    
                    # CME events table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS synapse_horizon.cme_events (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMPTZ NOT NULL UNIQUE,
                            angular_width DOUBLE PRECISION DEFAULT 360.0,
                            speed DOUBLE PRECISION DEFAULT 500.0,
                            acceleration DOUBLE PRECISION DEFAULT 0.0,
                            mass DOUBLE PRECISION DEFAULT 1e15,
                            energy DOUBLE PRECISION DEFAULT 1e32,
                            source_location VARCHAR(50) DEFAULT 'Unknown',
                            event_type VARCHAR(30) DEFAULT 'Halo',
                            confidence_score DOUBLE PRECISION DEFAULT 0.8,
                            remarks TEXT,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_cme_timestamp 
                        ON synapse_horizon.cme_events(timestamp)
                    """)
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_cme_type 
                        ON synapse_horizon.cme_events(event_type)
                    """)
                    
                    # Predictions table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS synapse_horizon.cme_predictions (
                            id SERIAL PRIMARY KEY,
                            prediction_timestamp TIMESTAMPTZ NOT NULL,
                            target_timestamp TIMESTAMPTZ NOT NULL,
                            cme_probability DOUBLE PRECISION NOT NULL CHECK (cme_probability >= 0 AND cme_probability <= 1),
                            risk_level VARCHAR(20) NOT NULL,
                            confidence DOUBLE PRECISION DEFAULT 0.0,
                            model_version VARCHAR(50) DEFAULT 'v1.0',
                            model_type VARCHAR(50) DEFAULT 'neural_network',
                            feature_vector JSONB,
                            hyperparameters JSONB,
                            processing_time DOUBLE PRECISION,
                            batch_id UUID,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_pred_timestamp 
                        ON synapse_horizon.cme_predictions(prediction_timestamp)
                    """)
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_target_timestamp 
                        ON synapse_horizon.cme_predictions(target_timestamp)
                    """)
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_pred_batch 
                        ON synapse_horizon.cme_predictions(batch_id)
                    """)
                    
                    # Training history table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS synapse_horizon.training_history (
                            id SERIAL PRIMARY KEY,
                            training_start TIMESTAMPTZ NOT NULL,
                            training_end TIMESTAMPTZ,
                            epochs INTEGER DEFAULT 0,
                            final_loss DOUBLE PRECISION DEFAULT 0.0,
                            final_accuracy DOUBLE PRECISION DEFAULT 0.0,
                            validation_accuracy DOUBLE PRECISION DEFAULT 0.0,
                            model_architecture JSONB,
                            hyperparameters JSONB,
                            training_samples INTEGER DEFAULT 0,
                            validation_samples INTEGER DEFAULT 0,
                            optimization_method VARCHAR(50),
                            best_hyperparameters JSONB,
                            convergence_info JSONB,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    
                    # Feature importance table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS synapse_horizon.feature_importance (
                            id SERIAL PRIMARY KEY,
                            feature_name VARCHAR(100) NOT NULL,
                            importance_score DOUBLE PRECISION NOT NULL,
                            importance_rank INTEGER,
                            model_version VARCHAR(50) DEFAULT 'v1.0',
                            analysis_method VARCHAR(50) DEFAULT 'permutation',
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    
                    # Model performance metrics table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS synapse_horizon.model_performance (
                            id SERIAL PRIMARY KEY,
                            model_version VARCHAR(50) NOT NULL,
                            model_type VARCHAR(50) NOT NULL,
                            accuracy DOUBLE PRECISION,
                            precision_score DOUBLE PRECISION,
                            recall DOUBLE PRECISION,
                            f1_score DOUBLE PRECISION,
                            auc_roc DOUBLE PRECISION,
                            confusion_matrix JSONB,
                            cross_validation_scores JSONB,
                            test_set_size INTEGER,
                            evaluation_date TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    
                    # Statistical analysis results table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS synapse_horizon.statistical_analysis (
                            id SERIAL PRIMARY KEY,
                            analysis_type VARCHAR(50) NOT NULL,
                            parameter_name VARCHAR(50),
                            analysis_results JSONB NOT NULL,
                            significance_level DOUBLE PRECISION,
                            p_value DOUBLE PRECISION,
                            confidence_interval JSONB,
                            sample_size INTEGER,
                            analysis_date TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    
                    # Batch processing jobs table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS synapse_horizon.batch_jobs (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            job_type VARCHAR(50) NOT NULL,
                            job_status VARCHAR(20) DEFAULT 'pending',
                            input_parameters JSONB,
                            output_results JSONB,
                            progress_percentage DOUBLE PRECISION DEFAULT 0.0,
                            error_message TEXT,
                            started_at TIMESTAMPTZ,
                            completed_at TIMESTAMPTZ,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    
                    # System logs table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS synapse_horizon.system_logs (
                            id SERIAL PRIMARY KEY,
                            log_level VARCHAR(20) NOT NULL,
                            message TEXT NOT NULL,
                            module_name VARCHAR(50),
                            function_name VARCHAR(50),
                            line_number INTEGER,
                            extra_data JSONB,
                            timestamp TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    
                    # Create update triggers for timestamps
                    cursor.execute("""
                        CREATE OR REPLACE FUNCTION update_modified_column()
                        RETURNS TRIGGER AS $$
                        BEGIN
                            NEW.updated_at = NOW();
                            RETURN NEW;
                        END;
                        $$ language 'plpgsql';
                    """)
                    
                    # Apply update triggers
                    for table in ['swis_data', 'cme_events']:
                        cursor.execute(f"""
                            DROP TRIGGER IF EXISTS update_{table}_modtime ON synapse_horizon.{table};
                            CREATE TRIGGER update_{table}_modtime 
                            BEFORE UPDATE ON synapse_horizon.{table}
                            FOR EACH ROW EXECUTE FUNCTION update_modified_column();
                        """)
                    
                    conn.commit()
                    self.logger.info("PostgreSQL database schema initialized successfully")
                    
                except psycopg2.Error as e:
                    conn.rollback()
                    raise RuntimeError(f"Failed to initialize PostgreSQL tables: {e}")
    
    def save_swis_data(self, swis_data: List[Dict[str, Any]]) -> int:
        """Save SWIS data to PostgreSQL with conflict resolution"""
        if not swis_data:
            return 0
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    # Prepare data for bulk insert
                    insert_data = []
                    for record in swis_data:
                        timestamp = record['timestamp']
                        if isinstance(timestamp, str):
                            timestamp = pd.to_datetime(timestamp)
                        
                        insert_data.append((
                            timestamp,
                            record.get('flux', 0.0),
                            record.get('density', 0.0),
                            record.get('temperature', 0.0),
                            record.get('speed', 0.0),
                            record.get('pressure', 0.0),
                            record.get('magnetic_field', 0.0),
                            record.get('data_quality_score', 1.0),
                            record.get('processing_flags', 0),
                            record.get('source_instrument', 'SWIS-ASPEX')
                        ))
                    
                    # Use ON CONFLICT for upsert
                    cursor.executemany("""
                        INSERT INTO synapse_horizon.swis_data 
                        (timestamp, flux, density, temperature, speed, pressure, magnetic_field,
                         data_quality_score, processing_flags, source_instrument)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (timestamp) DO UPDATE SET
                            flux = EXCLUDED.flux,
                            density = EXCLUDED.density,
                            temperature = EXCLUDED.temperature,
                            speed = EXCLUDED.speed,
                            pressure = EXCLUDED.pressure,
                            magnetic_field = EXCLUDED.magnetic_field,
                            data_quality_score = EXCLUDED.data_quality_score,
                            processing_flags = EXCLUDED.processing_flags,
                            updated_at = NOW()
                    """, insert_data)
                    
                    rows_affected = cursor.rowcount
                    conn.commit()
                    
                    self.logger.info(f"Inserted/updated {rows_affected} SWIS records")
                    return rows_affected
                    
                except psycopg2.Error as e:
                    conn.rollback()
                    raise RuntimeError(f"Failed to save SWIS data: {e}")
    
    def save_cme_events(self, cme_events: pd.DataFrame) -> int:
        """Save CME events to PostgreSQL"""
        if len(cme_events) == 0:
            return 0
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    insert_data = []
                    for _, row in cme_events.iterrows():
                        timestamp = pd.to_datetime(row['timestamp'])
                        
                        insert_data.append((
                            timestamp,
                            row.get('angular_width', 360.0),
                            row.get('speed', 500.0),
                            row.get('acceleration', 0.0),
                            row.get('mass', 1e15),
                            row.get('energy', 1e32),
                            row.get('source_location', 'Unknown'),
                            row.get('event_type', 'Halo'),
                            row.get('confidence_score', 0.8),
                            row.get('remarks', '')
                        ))
                    
                    cursor.executemany("""
                        INSERT INTO synapse_horizon.cme_events 
                        (timestamp, angular_width, speed, acceleration, mass, energy, 
                         source_location, event_type, confidence_score, remarks)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (timestamp) DO UPDATE SET
                            angular_width = EXCLUDED.angular_width,
                            speed = EXCLUDED.speed,
                            acceleration = EXCLUDED.acceleration,
                            mass = EXCLUDED.mass,
                            energy = EXCLUDED.energy,
                            source_location = EXCLUDED.source_location,
                            event_type = EXCLUDED.event_type,
                            confidence_score = EXCLUDED.confidence_score,
                            remarks = EXCLUDED.remarks,
                            updated_at = NOW()
                    """, insert_data)
                    
                    rows_affected = cursor.rowcount
                    conn.commit()
                    
                    self.logger.info(f"Inserted/updated {rows_affected} CME event records")
                    return rows_affected
                    
                except psycopg2.Error as e:
                    conn.rollback()
                    raise RuntimeError(f"Failed to save CME events: {e}")
    
    def save_predictions_batch(self, prediction_results: Dict[str, Any], batch_id: str = None) -> int:
        """Save prediction results in batch with PostgreSQL-specific features"""
        if not prediction_results or 'predictions' not in prediction_results:
            return 0
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    current_time = datetime.now()
                    predictions = prediction_results['predictions']
                    batch_uuid = batch_id or f"batch_{current_time.strftime('%Y%m%d_%H%M%S')}"
                    
                    insert_data = []
                    for i, prob in enumerate(predictions):
                        target_time = current_time + timedelta(hours=i+1)
                        
                        # Determine risk level
                        if prob < 0.3:
                            risk_level = "Low"
                        elif prob < 0.7:
                            risk_level = "Medium"
                        else:
                            risk_level = "High"
                        
                        # Prepare feature vector and hyperparameters as JSONB
                        feature_vector = None
                        if 'features' in prediction_results and i < len(prediction_results['features']):
                            feature_vector = json.dumps(prediction_results['features'][i].tolist())
                        
                        hyperparams = json.dumps(prediction_results.get('hyperparameters', {}))
                        
                        insert_data.append((
                            current_time,
                            target_time,
                            float(prob),
                            risk_level,
                            float(prob),  # confidence
                            prediction_results.get('model_version', 'v1.0'),
                            prediction_results.get('model_type', 'neural_network'),
                            feature_vector,
                            hyperparams,
                            prediction_results.get('processing_time', 0.0),
                            batch_uuid
                        ))
                    
                    cursor.executemany("""
                        INSERT INTO synapse_horizon.cme_predictions 
                        (prediction_timestamp, target_timestamp, cme_probability, risk_level, 
                         confidence, model_version, model_type, feature_vector, hyperparameters,
                         processing_time, batch_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, insert_data)
                    
                    rows_inserted = cursor.rowcount
                    conn.commit()
                    
                    self.logger.info(f"Inserted {rows_inserted} predictions in batch {batch_uuid}")
                    return rows_inserted
                    
                except psycopg2.Error as e:
                    conn.rollback()
                    raise RuntimeError(f"Failed to save predictions: {e}")
    
    def get_swis_data_filtered(self, 
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              min_quality_score: float = 0.5,
                              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get filtered SWIS data with quality filtering"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                try:
                    query = """
                        SELECT * FROM synapse_horizon.swis_data 
                        WHERE data_quality_score >= %s
                    """
                    params = [min_quality_score]
                    
                    if start_time:
                        query += " AND timestamp >= %s"
                        params.append(start_time)
                    
                    if end_time:
                        query += " AND timestamp <= %s"
                        params.append(end_time)
                    
                    query += " ORDER BY timestamp"
                    
                    if limit:
                        query += " LIMIT %s"
                        params.append(limit)
                    
                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    
                    return [dict(row) for row in results]
                    
                except psycopg2.Error as e:
                    raise RuntimeError(f"Failed to retrieve SWIS data: {e}")
    
    def execute_statistical_analysis(self, analysis_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute statistical analysis queries and store results"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    # Example: correlation analysis
                    if analysis_type == 'correlation':
                        cursor.execute("""
                            SELECT 
                                corr(flux, density) as flux_density_corr,
                                corr(speed, temperature) as speed_temp_corr,
                                corr(pressure, magnetic_field) as pressure_mag_corr
                            FROM synapse_horizon.swis_data 
                            WHERE data_quality_score > %s
                        """, [parameters.get('min_quality', 0.5)])
                        
                        result = cursor.fetchone()
                        analysis_results = {
                            'flux_density_correlation': result[0],
                            'speed_temperature_correlation': result[1],
                            'pressure_magnetic_correlation': result[2]
                        }
                    
                    # Store results
                    cursor.execute("""
                        INSERT INTO synapse_horizon.statistical_analysis
                        (analysis_type, analysis_results, sample_size)
                        VALUES (%s, %s, %s)
                    """, [analysis_type, json.dumps(analysis_results), parameters.get('sample_size', 0)])
                    
                    conn.commit()
                    return analysis_results
                    
                except psycopg2.Error as e:
                    conn.rollback()
                    raise RuntimeError(f"Statistical analysis failed: {e}")
    
    def create_batch_job(self, job_type: str, parameters: Dict[str, Any]) -> str:
        """Create a new batch processing job"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute("""
                        INSERT INTO synapse_horizon.batch_jobs 
                        (job_type, input_parameters, started_at)
                        VALUES (%s, %s, %s)
                        RETURNING id
                    """, [job_type, json.dumps(parameters), datetime.now()])
                    
                    job_id = cursor.fetchone()[0]
                    conn.commit()
                    
                    self.logger.info(f"Created batch job {job_id} of type {job_type}")
                    return str(job_id)
                    
                except psycopg2.Error as e:
                    conn.rollback()
                    raise RuntimeError(f"Failed to create batch job: {e}")
    
    def update_batch_job_progress(self, job_id: str, progress: float, status: str = None):
        """Update batch job progress"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    if status:
                        cursor.execute("""
                            UPDATE synapse_horizon.batch_jobs 
                            SET progress_percentage = %s, job_status = %s
                            WHERE id = %s
                        """, [progress, status, job_id])
                    else:
                        cursor.execute("""
                            UPDATE synapse_horizon.batch_jobs 
                            SET progress_percentage = %s
                            WHERE id = %s
                        """, [progress, job_id])
                    
                    conn.commit()
                    
                except psycopg2.Error as e:
                    conn.rollback()
                    raise RuntimeError(f"Failed to update batch job progress: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    stats = {}
                    
                    # Table counts
                    tables = ['swis_data', 'cme_events', 'cme_predictions', 'training_history']
                    for table in tables:
                        cursor.execute(f"SELECT COUNT(*) FROM synapse_horizon.{table}")
                        stats[f'{table}_count'] = cursor.fetchone()[0]
                    
                    # Data quality statistics
                    cursor.execute("""
                        SELECT 
                            AVG(data_quality_score) as avg_quality,
                            MIN(data_quality_score) as min_quality,
                            MAX(data_quality_score) as max_quality,
                            COUNT(*) FILTER (WHERE data_quality_score > 0.8) as high_quality_count
                        FROM synapse_horizon.swis_data
                    """)
                    
                    quality_stats = cursor.fetchone()
                    stats.update({
                        'average_data_quality': quality_stats[0],
                        'min_data_quality': quality_stats[1],
                        'max_data_quality': quality_stats[2],
                        'high_quality_records': quality_stats[3]
                    })
                    
                    # Recent activity
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM synapse_horizon.cme_predictions 
                        WHERE created_at > NOW() - INTERVAL '24 hours'
                    """)
                    stats['recent_predictions'] = cursor.fetchone()[0]
                    
                    return stats
                    
                except psycopg2.Error as e:
                    raise RuntimeError(f"Failed to get database stats: {e}")
    
    def close(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.connection_pool = None
            self.logger.info("Database connection pool closed")