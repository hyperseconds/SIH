#!/usr/bin/env python3
"""
Synapse Horizon - CME Prediction Application
Main GUI application for predicting Halo Coronal Mass Ejections
using custom neural network and SWIS-ASPEX payload data.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import os
import sys

# Import custom modules
from neural_net import NeuralNetwork
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from database import DatabaseManager
from visualization import Visualizer
from config import Config
from utils import Logger

class SynapseHorizonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Synapse Horizon - CME Prediction System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize components
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.db_manager = DatabaseManager()
        self.visualizer = Visualizer()
        self.neural_net = None
        self.logger = Logger()
        
        # Data storage
        self.swis_data = None
        self.cme_events = None
        self.processed_features = None
        self.training_data = None
        self.prediction_results = None
        
        # Training state
        self.is_training = False
        self.training_thread = None
        
        self.setup_styles()
        self.create_widgets()
        self.setup_layout()
        
    def setup_styles(self):
        """Configure custom styles for the application"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabel', background='#1e1e1e', foreground='#ffffff')
        style.configure('TButton', background='#2d2d2d', foreground='#ffffff')
        style.configure('TEntry', background='#2d2d2d', foreground='#ffffff')
        style.configure('Treeview', background='#2d2d2d', foreground='#ffffff')
        style.configure('TProgressbar', background='#4CAF50')
        
        # Header style
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), 
                       background='#1e1e1e', foreground='#4CAF50')
        
        # Status style
        style.configure('Status.TLabel', font=('Arial', 10), 
                       background='#1e1e1e', foreground='#ffaa00')
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main header
        self.header_frame = ttk.Frame(self.root)
        self.header_label = ttk.Label(self.header_frame, text="🛰️ Synapse Horizon - CME Prediction System", 
                                     style='Header.TLabel')
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Data Loading Tab
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="📁 Data Loading")
        self.create_data_tab()
        
        # Training Tab
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="🧠 Neural Network Training")
        self.create_training_tab()
        
        # Prediction Tab
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text="🔮 Live Prediction")
        self.create_prediction_tab()
        
        # Visualization Tab
        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="📊 Visualization")
        self.create_visualization_tab()
        
        # Database Tab
        self.db_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.db_tab, text="💾 Database")
        self.create_database_tab()
        
        # Status bar
        self.status_frame = ttk.Frame(self.root)
        self.status_label = ttk.Label(self.status_frame, text="Ready", style='Status.TLabel')
        
    def create_data_tab(self):
        """Create data loading tab widgets"""
        # SWIS Data section
        swis_frame = ttk.LabelFrame(self.data_tab, text="SWIS Solar Wind Data", padding=10)
        
        ttk.Label(swis_frame, text="SWIS Data File:").grid(row=0, column=0, sticky='w', pady=5)
        self.swis_path_var = tk.StringVar(value="swis_data.json")
        self.swis_entry = ttk.Entry(swis_frame, textvariable=self.swis_path_var, width=50)
        self.swis_entry.grid(row=0, column=1, padx=5, pady=5)
        
        self.browse_swis_btn = ttk.Button(swis_frame, text="Browse", 
                                         command=self.browse_swis_file)
        self.browse_swis_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.load_swis_btn = ttk.Button(swis_frame, text="Load SWIS Data", 
                                       command=self.load_swis_data)
        self.load_swis_btn.grid(row=1, column=0, columnspan=3, pady=10)
        
        # CME Events section
        cme_frame = ttk.LabelFrame(self.data_tab, text="CACTUS CME Events", padding=10)
        
        ttk.Label(cme_frame, text="CME Events File:").grid(row=0, column=0, sticky='w', pady=5)
        self.cme_path_var = tk.StringVar(value="cactus_events.csv")
        self.cme_entry = ttk.Entry(cme_frame, textvariable=self.cme_path_var, width=50)
        self.cme_entry.grid(row=0, column=1, padx=5, pady=5)
        
        self.browse_cme_btn = ttk.Button(cme_frame, text="Browse", 
                                        command=self.browse_cme_file)
        self.browse_cme_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.load_cme_btn = ttk.Button(cme_frame, text="Load CME Events", 
                                      command=self.load_cme_data)
        self.load_cme_btn.grid(row=1, column=0, columnspan=3, pady=10)
        
        # Data preview
        preview_frame = ttk.LabelFrame(self.data_tab, text="Data Preview", padding=10)
        
        self.data_preview = scrolledtext.ScrolledText(preview_frame, height=15, width=80,
                                                     bg='#2d2d2d', fg='#ffffff')
        self.data_preview.pack(fill='both', expand=True)
        
        # Process data button
        self.process_btn = ttk.Button(self.data_tab, text="Process & Align Data", 
                                     command=self.process_data)
        
        # Layout
        swis_frame.pack(fill='x', padx=10, pady=5)
        cme_frame.pack(fill='x', padx=10, pady=5)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.process_btn.pack(pady=10)
        
    def create_training_tab(self):
        """Create neural network training tab widgets"""
        # Network configuration
        config_frame = ttk.LabelFrame(self.training_tab, text="Network Configuration", padding=10)
        
        ttk.Label(config_frame, text="Hidden Layers:").grid(row=0, column=0, sticky='w', pady=5)
        self.hidden_layers_var = tk.StringVar(value="64,32,16")
        ttk.Entry(config_frame, textvariable=self.hidden_layers_var, width=20).grid(row=0, column=1, padx=5)
        
        ttk.Label(config_frame, text="Learning Rate:").grid(row=1, column=0, sticky='w', pady=5)
        self.learning_rate_var = tk.StringVar(value="0.001")
        ttk.Entry(config_frame, textvariable=self.learning_rate_var, width=20).grid(row=1, column=1, padx=5)
        
        ttk.Label(config_frame, text="Epochs:").grid(row=2, column=0, sticky='w', pady=5)
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(config_frame, textvariable=self.epochs_var, width=20).grid(row=2, column=1, padx=5)
        
        ttk.Label(config_frame, text="Batch Size:").grid(row=3, column=0, sticky='w', pady=5)
        self.batch_size_var = tk.StringVar(value="32")
        ttk.Entry(config_frame, textvariable=self.batch_size_var, width=20).grid(row=3, column=1, padx=5)
        
        # Training controls
        control_frame = ttk.Frame(self.training_tab)
        
        self.train_btn = ttk.Button(control_frame, text="🚀 Start Training", 
                                   command=self.start_training)
        self.train_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Stop Training", 
                                  command=self.stop_training, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Training progress
        progress_frame = ttk.LabelFrame(self.training_tab, text="Training Progress", padding=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.pack(pady=5)
        
        self.epoch_label = ttk.Label(progress_frame, text="Epoch: 0/0")
        self.epoch_label.pack(pady=2)
        
        self.loss_label = ttk.Label(progress_frame, text="Loss: N/A")
        self.loss_label.pack(pady=2)
        
        self.accuracy_label = ttk.Label(progress_frame, text="Accuracy: N/A")
        self.accuracy_label.pack(pady=2)
        
        # Training log
        log_frame = ttk.LabelFrame(self.training_tab, text="Training Log", padding=10)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, height=10, width=80,
                                                     bg='#2d2d2d', fg='#ffffff')
        self.training_log.pack(fill='both', expand=True)
        
        # Layout
        config_frame.pack(fill='x', padx=10, pady=5)
        control_frame.pack(pady=10)
        progress_frame.pack(fill='x', padx=10, pady=5)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
    def create_prediction_tab(self):
        """Create live prediction tab widgets"""
        # Prediction controls
        control_frame = ttk.LabelFrame(self.prediction_tab, text="Prediction Controls", padding=10)
        
        self.predict_btn = ttk.Button(control_frame, text="🔮 Make Prediction", 
                                     command=self.make_prediction)
        self.predict_btn.pack(side='left', padx=5)
        
        self.live_prediction_var = tk.BooleanVar()
        self.live_check = ttk.Checkbutton(control_frame, text="Live Monitoring", 
                                         variable=self.live_prediction_var,
                                         command=self.toggle_live_prediction)
        self.live_check.pack(side='left', padx=10)
        
        # Prediction results
        results_frame = ttk.LabelFrame(self.prediction_tab, text="Prediction Results", padding=10)
        
        self.prediction_display = scrolledtext.ScrolledText(results_frame, height=15, width=80,
                                                           bg='#2d2d2d', fg='#ffffff')
        self.prediction_display.pack(fill='both', expand=True)
        
        # Alert system
        alert_frame = ttk.LabelFrame(self.prediction_tab, text="CME Alert System", padding=10)
        
        self.alert_threshold_var = tk.StringVar(value="0.7")
        ttk.Label(alert_frame, text="Alert Threshold:").pack(side='left')
        ttk.Entry(alert_frame, textvariable=self.alert_threshold_var, width=10).pack(side='left', padx=5)
        
        self.alert_status = ttk.Label(alert_frame, text="🟢 No Alert", style='Status.TLabel')
        self.alert_status.pack(side='right', padx=10)
        
        # Export controls
        export_frame = ttk.Frame(self.prediction_tab)
        
        self.export_csv_btn = ttk.Button(export_frame, text="Export to CSV", 
                                        command=self.export_predictions_csv)
        self.export_csv_btn.pack(side='left', padx=5)
        
        self.export_db_btn = ttk.Button(export_frame, text="Save to Database", 
                                       command=self.save_predictions_db)
        self.export_db_btn.pack(side='left', padx=5)
        
        # Layout
        control_frame.pack(fill='x', padx=10, pady=5)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        alert_frame.pack(fill='x', padx=10, pady=5)
        export_frame.pack(pady=10)
        
    def create_visualization_tab(self):
        """Create visualization tab widgets"""
        # Plot controls
        control_frame = ttk.LabelFrame(self.viz_tab, text="Visualization Controls", padding=10)
        
        self.plot_swis_btn = ttk.Button(control_frame, text="📈 Plot SWIS Data", 
                                       command=self.plot_swis_data)
        self.plot_swis_btn.pack(side='left', padx=5)
        
        self.plot_predictions_btn = ttk.Button(control_frame, text="🎯 Plot Predictions", 
                                              command=self.plot_predictions)
        self.plot_predictions_btn.pack(side='left', padx=5)
        
        self.plot_cme_zones_btn = ttk.Button(control_frame, text="⚡ Plot CME Zones", 
                                            command=self.plot_cme_zones)
        self.plot_cme_zones_btn.pack(side='left', padx=5)
        
        # Plot options
        options_frame = ttk.LabelFrame(self.viz_tab, text="Plot Options", padding=10)
        
        self.plot_param_var = tk.StringVar(value="all")
        ttk.Label(options_frame, text="Parameters:").pack(side='left')
        param_combo = ttk.Combobox(options_frame, textvariable=self.plot_param_var,
                                  values=["all", "flux", "density", "temperature", "speed"])
        param_combo.pack(side='left', padx=5)
        
        self.time_range_var = tk.StringVar(value="24h")
        ttk.Label(options_frame, text="Time Range:").pack(side='left', padx=(20,5))
        time_combo = ttk.Combobox(options_frame, textvariable=self.time_range_var,
                                 values=["1h", "6h", "12h", "24h", "7d", "all"])
        time_combo.pack(side='left', padx=5)
        
        # Plot area placeholder
        plot_frame = ttk.LabelFrame(self.viz_tab, text="Plots will appear in separate windows", padding=10)
        
        info_text = """
        📊 Visualization Features:
        
        • SWIS Data Plots: Time-series visualization of solar wind parameters
        • Prediction Confidence: Neural network output confidence over time
        • CME Event Zones: Highlighted regions where CME events occurred
        • Feature Analysis: Moving averages, gradients, and derived metrics
        • Real-time Updates: Live plotting during prediction monitoring
        
        Click the buttons above to generate plots in separate matplotlib windows.
        """
        
        ttk.Label(plot_frame, text=info_text, justify='left').pack(anchor='w')
        
        # Layout
        control_frame.pack(fill='x', padx=10, pady=5)
        options_frame.pack(fill='x', padx=10, pady=5)
        plot_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
    def create_database_tab(self):
        """Create database management tab widgets"""
        # Database controls
        control_frame = ttk.LabelFrame(self.db_tab, text="Database Operations", padding=10)
        
        self.init_db_btn = ttk.Button(control_frame, text="🗄️ Initialize Database", 
                                     command=self.initialize_database)
        self.init_db_btn.pack(side='left', padx=5)
        
        self.backup_db_btn = ttk.Button(control_frame, text="💾 Backup Database", 
                                       command=self.backup_database)
        self.backup_db_btn.pack(side='left', padx=5)
        
        self.clear_db_btn = ttk.Button(control_frame, text="🗑️ Clear Tables", 
                                      command=self.clear_database)
        self.clear_db_btn.pack(side='left', padx=5)
        
        # Database status
        status_frame = ttk.LabelFrame(self.db_tab, text="Database Status", padding=10)
        
        self.db_status_text = scrolledtext.ScrolledText(status_frame, height=8, width=80,
                                                       bg='#2d2d2d', fg='#ffffff')
        self.db_status_text.pack(fill='both', expand=True)
        
        # Query interface
        query_frame = ttk.LabelFrame(self.db_tab, text="SQL Query Interface", padding=10)
        
        ttk.Label(query_frame, text="Query:").pack(anchor='w')
        self.query_text = scrolledtext.ScrolledText(query_frame, height=4, width=80,
                                                   bg='#2d2d2d', fg='#ffffff')
        self.query_text.pack(fill='x', pady=5)
        
        self.execute_query_btn = ttk.Button(query_frame, text="Execute Query", 
                                           command=self.execute_query)
        self.execute_query_btn.pack(pady=5)
        
        self.query_results = scrolledtext.ScrolledText(query_frame, height=8, width=80,
                                                      bg='#2d2d2d', fg='#ffffff')
        self.query_results.pack(fill='both', expand=True, pady=5)
        
        # Layout
        control_frame.pack(fill='x', padx=10, pady=5)
        status_frame.pack(fill='x', padx=10, pady=5)
        query_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
    def setup_layout(self):
        """Setup main window layout"""
        self.header_frame.pack(fill='x', padx=10, pady=5)
        self.header_label.pack()
        
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.status_frame.pack(fill='x', padx=10, pady=5)
        self.status_label.pack(anchor='w')
        
    def update_status(self, message):
        """Update status bar message"""
        self.status_label.config(text=f"Status: {message}")
        self.root.update_idletasks()
        
    def log_message(self, message, widget=None):
        """Log message to specified widget or training log"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        if widget is None:
            widget = self.training_log
            
        widget.insert(tk.END, formatted_message)
        widget.see(tk.END)
        self.root.update_idletasks()
        
    # Data Loading Methods
    def browse_swis_file(self):
        """Browse for SWIS data file"""
        filename = filedialog.askopenfilename(
            title="Select SWIS Data File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.swis_path_var.set(filename)
            
    def browse_cme_file(self):
        """Browse for CME events file"""
        filename = filedialog.askopenfilename(
            title="Select CME Events File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.cme_path_var.set(filename)
            
    def load_swis_data(self):
        """Load SWIS solar wind data"""
        try:
            self.update_status("Loading SWIS data...")
            file_path = self.swis_path_var.get()
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            self.swis_data = self.data_loader.load_swis_data(file_path)
            
            # Update preview
            preview_text = f"SWIS Data Loaded Successfully!\n"
            preview_text += f"Records: {len(self.swis_data)}\n"
            preview_text += f"Time Range: {self.swis_data[0]['timestamp']} to {self.swis_data[-1]['timestamp']}\n"
            preview_text += f"Parameters: {list(self.swis_data[0].keys())}\n\n"
            preview_text += "Sample Data:\n"
            for i, record in enumerate(self.swis_data[:5]):
                preview_text += f"{i+1}: {record}\n"
                
            self.data_preview.delete(1.0, tk.END)
            self.data_preview.insert(1.0, preview_text)
            
            self.update_status("SWIS data loaded successfully")
            messagebox.showinfo("Success", f"Loaded {len(self.swis_data)} SWIS records")
            
        except Exception as e:
            error_msg = f"Error loading SWIS data: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def load_cme_data(self):
        """Load CME events data"""
        try:
            self.update_status("Loading CME events...")
            file_path = self.cme_path_var.get()
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            self.cme_events = self.data_loader.load_cme_events(file_path)
            
            # Update preview
            preview_text = f"\nCME Events Loaded Successfully!\n"
            preview_text += f"Events: {len(self.cme_events)}\n"
            preview_text += f"Columns: {list(self.cme_events.columns)}\n\n"
            preview_text += "Sample Events:\n"
            preview_text += self.cme_events.head().to_string()
            
            self.data_preview.insert(tk.END, preview_text)
            
            self.update_status("CME events loaded successfully")
            messagebox.showinfo("Success", f"Loaded {len(self.cme_events)} CME events")
            
        except Exception as e:
            error_msg = f"Error loading CME events: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def process_data(self):
        """Process and align SWIS data with CME events"""
        try:
            if self.swis_data is None:
                raise ValueError("Please load SWIS data first")
            if self.cme_events is None:
                raise ValueError("Please load CME events first")
                
            self.update_status("Processing and aligning data...")
            
            # Process features
            self.processed_features = self.feature_engineer.process_features(self.swis_data)
            
            # Align with CME events
            self.training_data = self.feature_engineer.align_with_cme_events(
                self.processed_features, self.cme_events
            )
            
            # Update preview
            preview_text = f"\nData Processing Complete!\n"
            preview_text += f"Processed Features: {self.processed_features.shape}\n"
            preview_text += f"Training Samples: {len(self.training_data['X'])}\n"
            preview_text += f"Positive CME Cases: {sum(self.training_data['y'])}\n"
            preview_text += f"Feature Names: {self.training_data['feature_names']}\n"
            
            self.data_preview.insert(tk.END, preview_text)
            
            self.update_status("Data processing completed")
            messagebox.showinfo("Success", "Data processed and aligned successfully")
            
        except Exception as e:
            error_msg = f"Error processing data: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    # Training Methods
    def start_training(self):
        """Start neural network training"""
        try:
            if self.training_data is None:
                raise ValueError("Please process data first")
                
            if self.is_training:
                messagebox.showwarning("Warning", "Training is already in progress")
                return
                
            # Get training parameters
            hidden_layers = [int(x.strip()) for x in self.hidden_layers_var.get().split(',')]
            learning_rate = float(self.learning_rate_var.get())
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            
            # Initialize neural network
            input_size = self.training_data['X'].shape[1]
            self.neural_net = NeuralNetwork(
                input_size=input_size,
                hidden_layers=hidden_layers,
                output_size=1,
                learning_rate=learning_rate
            )
            
            # Update UI
            self.is_training = True
            self.train_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            
            # Clear training log
            self.training_log.delete(1.0, tk.END)
            
            # Start training in separate thread
            self.training_thread = threading.Thread(
                target=self.train_neural_network,
                args=(epochs, batch_size),
                daemon=True
            )
            self.training_thread.start()
            
        except Exception as e:
            error_msg = f"Error starting training: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def train_neural_network(self, epochs, batch_size):
        """Train the neural network (runs in separate thread)"""
        try:
            X_train = self.training_data['X']
            y_train = self.training_data['y'].reshape(-1, 1)
            
            self.log_message(f"Starting training: {epochs} epochs, batch size {batch_size}")
            self.log_message(f"Training data shape: {X_train.shape}")
            
            for epoch in range(epochs):
                if not self.is_training:  # Check if training was stopped
                    break
                    
                # Train for one epoch
                epoch_loss = self.neural_net.train_epoch(X_train, y_train, batch_size)
                
                # Calculate accuracy
                predictions = self.neural_net.predict(X_train)
                accuracy = self.calculate_accuracy(predictions, y_train)
                
                # Update progress
                progress = ((epoch + 1) / epochs) * 100
                self.progress_var.set(progress)
                
                # Update labels
                self.root.after(0, lambda e=epoch+1, ep=epochs: 
                               self.epoch_label.config(text=f"Epoch: {e}/{ep}"))
                self.root.after(0, lambda l=epoch_loss: 
                               self.loss_label.config(text=f"Loss: {l:.6f}"))
                self.root.after(0, lambda a=accuracy: 
                               self.accuracy_label.config(text=f"Accuracy: {a:.2%}"))
                
                # Log progress every 10 epochs
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    self.root.after(0, lambda e=epoch+1, l=epoch_loss, a=accuracy:
                                   self.log_message(f"Epoch {e}: Loss={l:.6f}, Accuracy={a:.2%}"))
                
            # Training completed
            if self.is_training:
                self.root.after(0, lambda: self.log_message("Training completed successfully!"))
                self.root.after(0, lambda: self.update_status("Training completed"))
            else:
                self.root.after(0, lambda: self.log_message("Training stopped by user"))
                self.root.after(0, lambda: self.update_status("Training stopped"))
                
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.root.after(0, lambda: self.log_message(f"ERROR: {error_msg}"))
            self.root.after(0, lambda: self.update_status(error_msg))
            
        finally:
            # Reset UI
            self.is_training = False
            self.root.after(0, lambda: self.train_btn.config(state='normal'))
            self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
            
    def stop_training(self):
        """Stop neural network training"""
        self.is_training = False
        self.log_message("Stopping training...")
        
    def calculate_accuracy(self, predictions, targets):
        """Calculate prediction accuracy"""
        import numpy as np
        pred_binary = (predictions > 0.5).astype(int)
        return np.mean(pred_binary == targets)
        
    # Prediction Methods
    def make_prediction(self):
        """Make CME prediction using trained model"""
        try:
            if self.neural_net is None:
                raise ValueError("Please train the neural network first")
                
            self.update_status("Making predictions...")
            
            # Use latest processed features for prediction
            if self.processed_features is None:
                raise ValueError("No processed features available")
                
            # Get recent data (last 24 hours worth)
            recent_features = self.processed_features[-24:]  # Assuming hourly data
            
            # Make predictions
            predictions = []
            for features in recent_features:
                pred = self.neural_net.predict(features.reshape(1, -1))
                predictions.append(float(pred[0][0]))
                
            # Store results
            self.prediction_results = {
                'predictions': predictions,
                'timestamps': [f"Hour {i+1}" for i in range(len(predictions))],
                'features': recent_features
            }
            
            # Display results
            self.display_prediction_results()
            
            # Check for alerts
            self.check_cme_alert()
            
            self.update_status("Predictions completed")
            
        except Exception as e:
            error_msg = f"Error making predictions: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def display_prediction_results(self):
        """Display prediction results in the text widget"""
        if self.prediction_results is None:
            return
            
        self.prediction_display.delete(1.0, tk.END)
        
        results_text = "🔮 CME Prediction Results\n"
        results_text += "=" * 50 + "\n\n"
        
        predictions = self.prediction_results['predictions']
        timestamps = self.prediction_results['timestamps']
        
        for i, (time, pred) in enumerate(zip(timestamps, predictions)):
            risk_level = self.get_risk_level(pred)
            results_text += f"{time}: {pred:.4f} ({risk_level})\n"
            
        results_text += "\n" + "=" * 50 + "\n"
        results_text += f"Average Probability: {sum(predictions)/len(predictions):.4f}\n"
        results_text += f"Maximum Probability: {max(predictions):.4f}\n"
        results_text += f"High Risk Periods: {sum(1 for p in predictions if p > 0.7)}\n"
        
        self.prediction_display.insert(1.0, results_text)
        
    def get_risk_level(self, probability):
        """Get risk level string based on probability"""
        if probability < 0.3:
            return "🟢 Low Risk"
        elif probability < 0.7:
            return "🟡 Medium Risk"
        else:
            return "🔴 High Risk"
            
    def check_cme_alert(self):
        """Check if CME alert should be triggered"""
        if self.prediction_results is None:
            return
            
        threshold = float(self.alert_threshold_var.get())
        max_prob = max(self.prediction_results['predictions'])
        
        if max_prob >= threshold:
            self.alert_status.config(text="🔴 CME ALERT!", foreground='red')
            messagebox.showwarning("CME Alert", 
                                 f"High CME probability detected: {max_prob:.2%}")
        else:
            self.alert_status.config(text="🟢 No Alert", foreground='green')
            
    def toggle_live_prediction(self):
        """Toggle live prediction monitoring"""
        if self.live_prediction_var.get():
            self.start_live_monitoring()
        else:
            self.stop_live_monitoring()
            
    def start_live_monitoring(self):
        """Start live prediction monitoring"""
        self.log_message("Starting live monitoring...", self.prediction_display)
        # Implementation for live monitoring would go here
        
    def stop_live_monitoring(self):
        """Stop live prediction monitoring"""
        self.log_message("Stopping live monitoring...", self.prediction_display)
        
    # Visualization Methods
    def plot_swis_data(self):
        """Plot SWIS solar wind data"""
        try:
            if self.swis_data is None:
                raise ValueError("Please load SWIS data first")
                
            self.update_status("Generating SWIS data plot...")
            
            parameter = self.plot_param_var.get()
            time_range = self.time_range_var.get()
            
            self.visualizer.plot_swis_data(self.swis_data, parameter, time_range)
            
            self.update_status("SWIS plot generated")
            
        except Exception as e:
            error_msg = f"Error plotting SWIS data: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def plot_predictions(self):
        """Plot prediction results"""
        try:
            if self.prediction_results is None:
                raise ValueError("Please make predictions first")
                
            self.update_status("Generating prediction plot...")
            
            self.visualizer.plot_predictions(self.prediction_results)
            
            self.update_status("Prediction plot generated")
            
        except Exception as e:
            error_msg = f"Error plotting predictions: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def plot_cme_zones(self):
        """Plot CME event zones"""
        try:
            if self.cme_events is None:
                raise ValueError("Please load CME events first")
            if self.swis_data is None:
                raise ValueError("Please load SWIS data first")
                
            self.update_status("Generating CME zones plot...")
            
            self.visualizer.plot_cme_zones(self.swis_data, self.cme_events)
            
            self.update_status("CME zones plot generated")
            
        except Exception as e:
            error_msg = f"Error plotting CME zones: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    # Database Methods
    def initialize_database(self):
        """Initialize database tables"""
        try:
            self.update_status("Initializing database...")
            
            self.db_manager.initialize_tables()
            
            status_text = "Database initialized successfully!\n"
            status_text += "Created tables:\n"
            status_text += "- swis_data\n"
            status_text += "- cme_events\n"
            status_text += "- cme_predictions\n"
            
            self.db_status_text.delete(1.0, tk.END)
            self.db_status_text.insert(1.0, status_text)
            
            self.update_status("Database initialized")
            messagebox.showinfo("Success", "Database initialized successfully")
            
        except Exception as e:
            error_msg = f"Error initializing database: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def backup_database(self):
        """Backup database to file"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Database Backup",
                defaultextension=".db",
                filetypes=[("Database files", "*.db"), ("All files", "*.*")]
            )
            
            if filename:
                self.update_status("Creating database backup...")
                self.db_manager.backup_database(filename)
                self.update_status("Database backup completed")
                messagebox.showinfo("Success", f"Database backed up to {filename}")
                
        except Exception as e:
            error_msg = f"Error backing up database: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def clear_database(self):
        """Clear all database tables"""
        try:
            result = messagebox.askyesno("Confirm", 
                                       "Are you sure you want to clear all database tables?")
            if result:
                self.update_status("Clearing database...")
                self.db_manager.clear_tables()
                
                status_text = "Database tables cleared!\n"
                self.db_status_text.delete(1.0, tk.END)
                self.db_status_text.insert(1.0, status_text)
                
                self.update_status("Database cleared")
                messagebox.showinfo("Success", "Database tables cleared")
                
        except Exception as e:
            error_msg = f"Error clearing database: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def execute_query(self):
        """Execute SQL query"""
        try:
            query = self.query_text.get(1.0, tk.END).strip()
            if not query:
                messagebox.showwarning("Warning", "Please enter a query")
                return
                
            self.update_status("Executing query...")
            
            results = self.db_manager.execute_query(query)
            
            # Display results
            if results:
                results_text = "Query Results:\n"
                results_text += "=" * 50 + "\n"
                for row in results:
                    results_text += str(row) + "\n"
            else:
                results_text = "Query executed successfully (no results returned)"
                
            self.query_results.delete(1.0, tk.END)
            self.query_results.insert(1.0, results_text)
            
            self.update_status("Query completed")
            
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            self.update_status(error_msg)
            self.query_results.delete(1.0, tk.END)
            self.query_results.insert(1.0, f"ERROR: {error_msg}")
            
    # Export Methods
    def export_predictions_csv(self):
        """Export predictions to CSV file"""
        try:
            if self.prediction_results is None:
                raise ValueError("No predictions to export")
                
            filename = filedialog.asksaveasfilename(
                title="Export Predictions to CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                self.update_status("Exporting predictions to CSV...")
                
                import csv
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Timestamp', 'CME_Probability', 'Risk_Level'])
                    
                    for time, pred in zip(self.prediction_results['timestamps'], 
                                        self.prediction_results['predictions']):
                        risk_level = self.get_risk_level(pred)
                        writer.writerow([time, pred, risk_level])
                        
                self.update_status("Predictions exported to CSV")
                messagebox.showinfo("Success", f"Predictions exported to {filename}")
                
        except Exception as e:
            error_msg = f"Error exporting to CSV: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def save_predictions_db(self):
        """Save predictions to database"""
        try:
            if self.prediction_results is None:
                raise ValueError("No predictions to save")
                
            self.update_status("Saving predictions to database...")
            
            self.db_manager.save_predictions(self.prediction_results)
            
            self.update_status("Predictions saved to database")
            messagebox.showinfo("Success", "Predictions saved to database")
            
        except Exception as e:
            error_msg = f"Error saving to database: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", error_msg)


def main():
    """Main application entry point"""
    try:
        # Create main window
        root = tk.Tk()
        
        # Initialize application
        app = SynapseHorizonGUI(root)
        
        # Start main loop
        root.mainloop()
        
    except Exception as e:
        print(f"Application error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
