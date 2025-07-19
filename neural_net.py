#!/usr/bin/env python3
"""
Custom Neural Network Implementation for CME Prediction
Built from scratch using only NumPy - no TensorFlow/PyTorch/scikit-learn
"""

import numpy as np
import json
import pickle
from typing import List, Tuple, Optional, Dict, Any

class ActivationFunction:
    """Collection of activation functions and their derivatives"""
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU function"""
        return np.where(x > 0, 1.0, alpha)


class LossFunction:
    """Collection of loss functions and their derivatives"""
    
    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary cross-entropy loss"""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Derivative of binary cross-entropy loss"""
        # Clip predictions to prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / len(y_true)
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean squared error loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mean_squared_error_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Derivative of mean squared error loss"""
        return 2 * (y_pred - y_true) / len(y_true)


class Layer:
    """Dense layer implementation"""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        """
        Initialize dense layer
        
        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'leaky_relu')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights using Xavier initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        
        # Store last inputs and outputs for backpropagation
        self.last_input = None
        self.last_z = None
        self.last_output = None
        
        # Gradient accumulators
        self.weight_gradients = np.zeros_like(self.weights)
        self.bias_gradients = np.zeros_like(self.biases)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer"""
        self.last_input = x.copy()
        
        # Linear transformation
        self.last_z = np.dot(x, self.weights) + self.biases
        
        # Apply activation function
        if self.activation == 'relu':
            self.last_output = ActivationFunction.relu(self.last_z)
        elif self.activation == 'sigmoid':
            self.last_output = ActivationFunction.sigmoid(self.last_z)
        elif self.activation == 'tanh':
            self.last_output = ActivationFunction.tanh(self.last_z)
        elif self.activation == 'leaky_relu':
            self.last_output = ActivationFunction.leaky_relu(self.last_z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
            
        return self.last_output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through the layer"""
        # Compute activation derivative
        if self.activation == 'relu':
            activation_grad = ActivationFunction.relu_derivative(self.last_z)
        elif self.activation == 'sigmoid':
            activation_grad = ActivationFunction.sigmoid_derivative(self.last_z)
        elif self.activation == 'tanh':
            activation_grad = ActivationFunction.tanh_derivative(self.last_z)
        elif self.activation == 'leaky_relu':
            activation_grad = ActivationFunction.leaky_relu_derivative(self.last_z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
        
        # Gradient with respect to pre-activation
        grad_z = grad_output * activation_grad
        
        # Gradients with respect to weights and biases
        self.weight_gradients = np.dot(self.last_input.T, grad_z)
        self.bias_gradients = np.sum(grad_z, axis=0, keepdims=True)
        
        # Gradient with respect to input
        grad_input = np.dot(grad_z, self.weights.T)
        
        return grad_input
    
    def update_weights(self, learning_rate: float):
        """Update weights and biases using gradients"""
        self.weights -= learning_rate * self.weight_gradients
        self.biases -= learning_rate * self.bias_gradients
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get layer parameters"""
        return {
            'weights': self.weights.copy(),
            'biases': self.biases.copy()
        }
    
    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        """Set layer parameters"""
        self.weights = parameters['weights'].copy()
        self.biases = parameters['biases'].copy()


class NeuralNetwork:
    """Custom Neural Network implementation"""
    
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int, 
                 learning_rate: float = 0.001, activation: str = 'relu',
                 output_activation: str = 'sigmoid'):
        """
        Initialize neural network
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            output_size: Number of output neurons
            learning_rate: Learning rate for optimization
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation
        
        # Build network layers
        self.layers = []
        
        # First hidden layer
        if hidden_layers:
            self.layers.append(Layer(input_size, hidden_layers[0], activation))
            
            # Additional hidden layers
            for i in range(1, len(hidden_layers)):
                self.layers.append(Layer(hidden_layers[i-1], hidden_layers[i], activation))
            
            # Output layer
            self.layers.append(Layer(hidden_layers[-1], output_size, output_activation))
        else:
            # No hidden layers - direct input to output
            self.layers.append(Layer(input_size, output_size, output_activation))
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Backward pass through the network"""
        # Compute loss
        loss = LossFunction.binary_cross_entropy(y_true, y_pred)
        
        # Compute initial gradient
        grad = LossFunction.binary_cross_entropy_derivative(y_true, y_pred)
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        return loss
    
    def update_weights(self):
        """Update all layer weights"""
        for layer in self.layers:
            layer.update_weights(self.learning_rate)
    
    def train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Train on a single batch"""
        # Forward pass
        y_pred = self.forward(x_batch)
        
        # Backward pass
        loss = self.backward(y_batch, y_pred)
        
        # Update weights
        self.update_weights()
        
        return loss
    
    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> float:
        """Train for one epoch"""
        n_samples = X.shape[0]
        total_loss = 0.0
        n_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Train on batches
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            x_batch = X_shuffled[i:end_idx]
            y_batch = y_shuffled[i:end_idx]
            
            batch_loss = self.train_batch(x_batch, y_batch)
            total_loss += batch_loss
            n_batches += 1
        
        # Average loss
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        self.loss_history.append(avg_loss)
        
        return avg_loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.forward(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        return self.predict(X)
    
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary classes"""
        probabilities = self.predict(X)
        return (probabilities > threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X)
        y_pred_classes = self.predict_classes(X)
        
        # Calculate metrics
        loss = LossFunction.binary_cross_entropy(y, y_pred)
        accuracy = np.mean(y_pred_classes.flatten() == y.flatten())
        
        # Precision and recall
        true_positives = np.sum((y_pred_classes == 1) & (y == 1))
        false_positives = np.sum((y_pred_classes == 1) & (y == 0))
        false_negatives = np.sum((y_pred_classes == 0) & (y == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        model_data = {
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history,
            'layers': []
        }
        
        # Save layer parameters
        for layer in self.layers:
            layer_data = {
                'input_size': layer.input_size,
                'output_size': layer.output_size,
                'activation': layer.activation,
                'parameters': layer.get_parameters()
            }
            model_data['layers'].append(layer_data)
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore network architecture
        self.input_size = model_data['input_size']
        self.hidden_layers = model_data['hidden_layers']
        self.output_size = model_data['output_size']
        self.learning_rate = model_data['learning_rate']
        self.activation = model_data['activation']
        self.output_activation = model_data['output_activation']
        self.loss_history = model_data['loss_history']
        self.accuracy_history = model_data['accuracy_history']
        
        # Restore layers
        self.layers = []
        for layer_data in model_data['layers']:
            layer = Layer(
                layer_data['input_size'],
                layer_data['output_size'],
                layer_data['activation']
            )
            layer.set_parameters(layer_data['parameters'])
            self.layers.append(layer)
    
    def get_layer_outputs(self, X: np.ndarray) -> List[np.ndarray]:
        """Get outputs from all layers (for debugging/visualization)"""
        outputs = []
        current_output = X
        
        for layer in self.layers:
            current_output = layer.forward(current_output)
            outputs.append(current_output.copy())
        
        return outputs
    
    def get_weights_summary(self) -> Dict[str, Any]:
        """Get summary of network weights"""
        summary = {
            'total_parameters': 0,
            'layers': []
        }
        
        for i, layer in enumerate(self.layers):
            layer_params = layer.weights.size + layer.biases.size
            summary['total_parameters'] += layer_params
            
            layer_info = {
                'layer': i,
                'shape': f"{layer.input_size} -> {layer.output_size}",
                'activation': layer.activation,
                'parameters': layer_params,
                'weight_mean': float(np.mean(layer.weights)),
                'weight_std': float(np.std(layer.weights)),
                'bias_mean': float(np.mean(layer.biases)),
                'bias_std': float(np.std(layer.biases))
            }
            summary['layers'].append(layer_info)
        
        return summary


class EnsembleNetwork:
    """Ensemble of neural networks for improved predictions"""
    
    def __init__(self, n_models: int = 5, **kwargs):
        """
        Initialize ensemble of neural networks
        
        Args:
            n_models: Number of models in ensemble
            **kwargs: Arguments passed to each NeuralNetwork
        """
        self.n_models = n_models
        self.models = []
        self.model_kwargs = kwargs
        
        # Initialize models with different random seeds
        for i in range(n_models):
            np.random.seed(i * 42)  # Different seed for each model
            model = NeuralNetwork(**kwargs)
            self.models.append(model)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
              batch_size: int = 32, validation_split: float = 0.2) -> Dict[str, List[float]]:
        """Train all models in the ensemble"""
        # Split data
        n_samples = X.shape[0]
        n_val = int(n_samples * validation_split)
        
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        ensemble_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        # Train each model
        for epoch in range(epochs):
            epoch_train_losses = []
            epoch_val_losses = []
            epoch_train_accuracies = []
            epoch_val_accuracies = []
            
            for i, model in enumerate(self.models):
                # Train one epoch
                train_loss = model.train_epoch(X_train, y_train, batch_size)
                
                # Evaluate
                train_metrics = model.evaluate(X_train, y_train)
                val_metrics = model.evaluate(X_val, y_val)
                
                epoch_train_losses.append(train_loss)
                epoch_val_losses.append(val_metrics['loss'])
                epoch_train_accuracies.append(train_metrics['accuracy'])
                epoch_val_accuracies.append(val_metrics['accuracy'])
            
            # Average across models
            ensemble_history['train_loss'].append(np.mean(epoch_train_losses))
            ensemble_history['val_loss'].append(np.mean(epoch_val_losses))
            ensemble_history['train_accuracy'].append(np.mean(epoch_train_accuracies))
            ensemble_history['val_accuracy'].append(np.mean(epoch_val_accuracies))
        
        return ensemble_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions by averaging"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Mean and standard deviation
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def save_ensemble(self, filepath_prefix: str):
        """Save all models in ensemble"""
        for i, model in enumerate(self.models):
            filepath = f"{filepath_prefix}_model_{i}.pkl"
            model.save_model(filepath)
        
        # Save ensemble metadata
        metadata = {
            'n_models': self.n_models,
            'model_kwargs': self.model_kwargs
        }
        
        with open(f"{filepath_prefix}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_ensemble(self, filepath_prefix: str):
        """Load all models in ensemble"""
        # Load metadata
        with open(f"{filepath_prefix}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.n_models = metadata['n_models']
        self.model_kwargs = metadata['model_kwargs']
        
        # Load models
        self.models = []
        for i in range(self.n_models):
            filepath = f"{filepath_prefix}_model_{i}.pkl"
            model = NeuralNetwork(**self.model_kwargs)
            model.load_model(filepath)
            self.models.append(model)


# Utility functions for model analysis
def analyze_feature_importance(model: NeuralNetwork, X: np.ndarray, 
                             feature_names: List[str]) -> Dict[str, float]:
    """
    Analyze feature importance using perturbation method
    
    Args:
        model: Trained neural network
        X: Input data
        feature_names: Names of features
    
    Returns:
        Dictionary mapping feature names to importance scores
    """
    baseline_pred = model.predict(X)
    baseline_loss = np.mean(baseline_pred)
    
    importance_scores = {}
    
    for i, feature_name in enumerate(feature_names):
        # Perturb feature by adding noise
        X_perturbed = X.copy()
        noise = np.random.normal(0, np.std(X[:, i]), X.shape[0])
        X_perturbed[:, i] += noise
        
        # Calculate change in prediction
        perturbed_pred = model.predict(X_perturbed)
        perturbed_loss = np.mean(perturbed_pred)
        
        # Importance is change in loss
        importance = abs(perturbed_loss - baseline_loss)
        importance_scores[feature_name] = importance
    
    return importance_scores


def cross_validate(X: np.ndarray, y: np.ndarray, model_params: Dict[str, Any], 
                  k_folds: int = 5) -> Dict[str, List[float]]:
    """
    Perform k-fold cross validation
    
    Args:
        X: Input features
        y: Target labels
        model_params: Parameters for NeuralNetwork
        k_folds: Number of folds
    
    Returns:
        Dictionary with cross-validation metrics
    """
    n_samples = X.shape[0]
    fold_size = n_samples // k_folds
    
    cv_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'loss': []
    }
    
    for fold in range(k_folds):
        # Split data
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < k_folds - 1 else n_samples
        
        val_indices = list(range(start_idx, end_idx))
        train_indices = list(range(0, start_idx)) + list(range(end_idx, n_samples))
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Train model
        model = NeuralNetwork(**model_params)
        
        # Simple training loop
        for epoch in range(50):  # Reduced epochs for cross-validation
            model.train_epoch(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_val, y_val)
        
        for key in cv_results:
            cv_results[key].append(metrics[key])
    
    return cv_results
