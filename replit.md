# Synapse Horizon - CME Prediction System

## Overview

Synapse Horizon is a sophisticated desktop application that predicts Halo Coronal Mass Ejections (CMEs) using solar wind particle data from the SWIS-ASPEX payload aboard India's Aditya-L1 spacecraft. The system employs a custom-built neural network implemented from scratch using only NumPy, without relying on external ML frameworks like TensorFlow or PyTorch.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

### Frontend Architecture
- **GUI Framework**: Tkinter-based desktop application with a modern dark theme
- **Visualization**: matplotlib integration for real-time plots and dashboards
- **User Interface**: Tabbed interface with organized workflow sections
- **Styling**: Custom space-themed styling with Bootstrap-like design principles

### Backend Architecture
- **Neural Network**: Custom implementation built from scratch using NumPy
- **Data Processing**: Advanced feature engineering with rolling statistics and spectral analysis
- **Database Layer**: SQLite for local data storage with cross-platform compatibility
- **Modular Design**: Separate modules for different functionalities (neural_net, data_loader, feature_engineering, etc.)

## Key Components

### 1. Custom Neural Network (`neural_net.py`)
- **Problem**: Need for CME prediction without external ML libraries
- **Solution**: From-scratch neural network implementation using only NumPy
- **Features**: Custom activation functions, backpropagation, Xavier initialization, gradient descent with momentum
- **Pros**: Full control over architecture, no external dependencies
- **Cons**: More development time, potential performance limitations

### 2. Data Processing Pipeline
- **Data Loader** (`data_loader.py`): Handles JSON and CSV data loading with timestamp alignment
- **Feature Engineering** (`feature_engineering.py`): Implements 50+ engineered features including rolling statistics and gradients
- **Database Manager** (`database.py`): SQLite-based data storage with backup and recovery

### 3. Visualization System (`visualization.py`)
- **Problem**: Need for comprehensive data visualization and monitoring
- **Solution**: matplotlib-based plotting system with multiple visualization types
- **Features**: Multi-parameter plots, CME event zones, prediction confidence visualization

### 4. Configuration Management (`config.py`)
- **Problem**: Centralized configuration for various system parameters
- **Solution**: Single configuration class with all constants and settings
- **Benefits**: Easy parameter tuning, consistent configuration across modules

## Data Flow

1. **Data Ingestion**: SWIS solar wind data (JSON) and CME events (CSV) are loaded
2. **Data Alignment**: Timestamps are aligned between different data sources
3. **Feature Engineering**: Raw data is processed into 50+ engineered features
4. **Database Storage**: Processed data is stored in SQLite database
5. **Model Training**: Custom neural network trains on historical data
6. **Prediction**: Trained model generates CME probability predictions
7. **Visualization**: Results are displayed through interactive GUI components

## External Dependencies

### Core Dependencies
- **NumPy**: Mathematical operations and neural network computations
- **Pandas**: Data manipulation and time-series processing
- **matplotlib**: Visualization and plotting
- **Tkinter**: GUI framework (built into Python)
- **SQLite**: Local database storage (built into Python)
- **SciPy**: Scientific computing for feature engineering

### Optional Dependencies
- **seaborn**: Enhanced visualization styling
- **JSON**: Data serialization (built into Python)
- **pickle**: Model serialization (built into Python)

## Deployment Strategy

### Local Desktop Application
- **Target Platform**: Cross-platform desktop application
- **Database**: SQLite for portability and no external database dependencies
- **Distribution**: Standalone Python application with bundled dependencies
- **Data Storage**: Local file system with automatic backup capabilities

### Key Architectural Decisions

1. **SQLite over PostgreSQL**: 
   - **Rationale**: Simplified deployment, no external database server required
   - **Trade-off**: Reduced concurrent access capabilities but improved portability

2. **Custom Neural Network**:
   - **Rationale**: Full control over architecture, educational value, no external ML dependencies
   - **Trade-off**: More development effort but complete customization

3. **Modular Design**:
   - **Rationale**: Maintainable codebase with clear separation of concerns
   - **Benefits**: Easy testing, debugging, and feature additions

4. **Tkinter GUI**:
   - **Rationale**: Built-in Python GUI framework, no additional dependencies
   - **Trade-off**: Limited styling options but high compatibility

The system is designed for scientific research and space weather prediction applications, emphasizing reliability, portability, and educational value through its from-scratch neural network implementation.