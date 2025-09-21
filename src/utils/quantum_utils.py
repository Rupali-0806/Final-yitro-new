"""
Quantum utility functions for quantum circuit operations.
"""

import numpy as np
import torch
import pennylane as qml
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class QuantumCircuitBuilder:
    """
    Builder class for creating quantum circuits for different modalities using PennyLane.
    """
    
    @staticmethod
    def create_variational_circuit(n_qubits: int, n_layers: int, 
                                 feature_map: str = 'angle_encoding'):
        """
        Create a variational quantum circuit using PennyLane.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            feature_map: Type of feature encoding ('angle_encoding', 'amplitude_encoding')
            
        Returns:
            PennyLane QNode function
        """
        dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(dev)
        def circuit(inputs, weights):
            # Feature encoding
            if feature_map == 'angle_encoding':
                for i in range(n_qubits):
                    qml.RY(inputs[i], wires=i)
            elif feature_map == 'amplitude_encoding':
                for i in range(n_qubits):
                    qml.RY(inputs[i] * np.pi, wires=i)
            
            # Variational layers
            for layer in range(n_layers):
                # Rotation gates
                for i in range(n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                
                # Entangling gates
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])  # Circular entanglement
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return circuit
    
    @staticmethod
    def create_quantum_convolutional_layer(n_qubits: int):
        """
        Create a quantum convolutional layer using PennyLane.
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            PennyLane QNode function
        """
        dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(dev)
        def conv_circuit(patch, weights):
            # Apply local rotations
            for i in range(min(len(patch), n_qubits)):
                qml.RY(patch[i] * weights[i, 0], wires=i)
            
            # Apply controlled rotations (convolution-like operation)
            for i in range(n_qubits - 1):
                qml.CRY(weights[i, 1], wires=[i, i + 1])
            
            return qml.expval(qml.PauliZ(0))
        
        return conv_circuit

class PennyLaneCircuits:
    """
    PennyLane quantum circuits for gradient-based optimization.
    """
    
    @staticmethod
    def create_pennylane_device(n_qubits: int, shots: Optional[int] = None):
        """Create PennyLane device."""
        if shots is None:
            return qml.device('default.qubit', wires=n_qubits)
        else:
            return qml.device('default.qubit', wires=n_qubits, shots=shots)
    
    @staticmethod
    def variational_classifier(n_qubits: int, n_layers: int):
        """
        Create a variational quantum classifier using PennyLane.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of layers
            
        Returns:
            PennyLane QNode function
        """
        dev = PennyLaneCircuits.create_pennylane_device(n_qubits)
        
        @qml.qnode(dev)
        def circuit(inputs, weights):
            # Feature encoding
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Variational layers
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                
                # Entangling gates
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])
            
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return circuit

class QuantumFeatureExtractor:
    """
    Quantum feature extraction utilities.
    """
    
    @staticmethod
    def quantum_kernel_matrix(X: np.ndarray, n_qubits: int, gamma: float = 1.0) -> np.ndarray:
        """
        Compute quantum kernel matrix for data.
        
        Args:
            X: Input data (n_samples, n_features)
            n_qubits: Number of qubits
            gamma: Kernel parameter
            
        Returns:
            Quantum kernel matrix
        """
        n_samples = X.shape[0]
        kernel_matrix = np.zeros((n_samples, n_samples))
        
        dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(dev)
        def kernel_circuit(x1, x2):
            # Encode first sample
            for i in range(min(len(x1), n_qubits)):
                qml.RY(x1[i] * gamma, wires=i)
            
            # Encode second sample (adjoint)
            for i in range(min(len(x2), n_qubits)):
                qml.RY(-x2[i] * gamma, wires=i)
            
            return qml.probs(wires=0)
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                probs = kernel_circuit(X[i], X[j])
                kernel_matrix[i, j] = probs[0]  # Probability of measuring |0⟩
                kernel_matrix[j, i] = kernel_matrix[i, j]
        
        return kernel_matrix
    
    @staticmethod
    def quantum_feature_map(X: np.ndarray, n_qubits: int, n_layers: int = 2) -> np.ndarray:
        """
        Extract quantum features from classical data.
        
        Args:
            X: Input data (n_samples, n_features)
            n_qubits: Number of qubits
            n_layers: Number of feature map layers
            
        Returns:
            Quantum features (n_samples, n_qubits)
        """
        n_samples = X.shape[0]
        quantum_features = np.zeros((n_samples, n_qubits))
        
        dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(dev)
        def feature_map_circuit(x):
            # Multi-layer feature encoding
            for layer in range(n_layers):
                for i in range(min(len(x), n_qubits)):
                    qml.RY(x[i] * (layer + 1), wires=i)
                
                # Add entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        for i, sample in enumerate(X):
            quantum_features[i] = feature_map_circuit(sample)
        
        return quantum_features

def initialize_quantum_weights(n_layers: int, n_qubits: int, 
                              weight_shape: Tuple = (3,)) -> np.ndarray:
    """
    Initialize quantum circuit weights.
    
    Args:
        n_layers: Number of variational layers
        n_qubits: Number of qubits
        weight_shape: Shape of weights per qubit per layer
        
    Returns:
        Initialized weight array
    """
    shape = (n_layers, n_qubits) + weight_shape
    return np.random.uniform(0, 2 * np.pi, shape)

def measure_circuit_gradients(circuit_func, weights: np.ndarray, 
                            inputs: np.ndarray) -> np.ndarray:
    """
    Compute gradients using parameter shift rule.
    
    Args:
        circuit_func: Quantum circuit function
        weights: Circuit weights
        inputs: Input data
        
    Returns:
        Computed gradients
    """
    gradients = np.zeros_like(weights)
    shift = np.pi / 2
    
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            for k in range(weights.shape[2]):
                # Shift parameter up
                weights_plus = weights.copy()
                weights_plus[i, j, k] += shift
                
                # Shift parameter down  
                weights_minus = weights.copy()
                weights_minus[i, j, k] -= shift
                
                # Compute gradient using parameter shift rule
                output_plus = circuit_func(inputs, weights_plus)
                output_minus = circuit_func(inputs, weights_minus)
                
                gradients[i, j, k] = 0.5 * (output_plus - output_minus)
    
    return gradients