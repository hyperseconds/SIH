# 🛰️ Synapse Horizon - CME Prediction System

**Advanced Coronal Mass Ejection Prediction using Custom Neural Networks**

Synapse Horizon is a sophisticated desktop application designed to predict Halo Coronal Mass Ejections (CMEs) using solar wind particle data from the SWIS-ASPEX payload onboard India's Aditya-L1 spacecraft. The system employs a completely custom neural network implementation built from scratch using only NumPy, providing accurate CME predictions through advanced feature engineering and time-series analysis.

---

## 🌟 Key Features

### 🧠 **Custom Neural Network**
- **Pure NumPy Implementation**: No external ML libraries (TensorFlow, PyTorch, scikit-learn)
- **From-Scratch Architecture**: Custom dense layers, activation functions, and backpropagation
- **Advanced Optimizations**: Xavier initialization, gradient descent with momentum
- **Ensemble Support**: Multiple model ensemble for improved accuracy

### 📊 **Advanced Data Processing**
- **SWIS Data Integration**: Solar Wind Ion Spectrometer Level-2 data processing
- **CME Event Correlation**: CACTUS CME database integration and alignment
- **Feature Engineering**: 50+ engineered features including rolling statistics, gradients, spectral analysis
- **Data Validation**: Comprehensive quality checks and anomaly detection

### 🎯 **Intelligent Prediction**
- **Time-Series Analysis**: Multi-horizon CME probability prediction
- **Risk Assessment**: Three-tier risk classification (Low/Medium/High)
- **Real-Time Monitoring**: Live prediction updates and alerts
- **Confidence Intervals**: Uncertainty quantification for predictions

### 🖥️ **Professional GUI**
- **Modern Dark Theme**: Space-themed interface with custom styling
- **Tabbed Interface**: Organized workflow with dedicated sections
- **Interactive Visualizations**: Real-time plots and dashboards
- **Progress Monitoring**: Training progress and system status indicators

### 📈 **Advanced Visualization**
- **Multi-Parameter Plots**: Solar wind parameter time-series
- **CME Event Zones**: Highlighted CME periods on data plots
- **Prediction Confidence**: Uncertainty visualization and risk timelines
- **Feature Importance**: Analysis of most predictive parameters
- **Training Metrics**: Loss curves and performance monitoring

### 💾 **Data Management**
- **SQLite Database**: Cross-platform local data storage
- **Export Capabilities**: CSV, JSON, and database export options
- **Data Backup**: Automated backup and recovery systems
- **Query Interface**: SQL query execution for custom analysis

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space

### Installation

1. **Clone or download the application files**
   ```bash
   # Ensure all files are in the same directory
   ls
   # Should show: main.py, neural_net.py, data_loader.py, etc.
   ```

2. **Install required packages**
   ```bash
   pip install numpy scipy matplotlib pandas
   ```

3. **Run the application**
   ```bash
   python main.py
   