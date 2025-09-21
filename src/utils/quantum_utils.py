"""
Quantum utility functions for quantum circuit operations.
"""

import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.primitives import Estimator
import pennylane as qml
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class QuantumCircuitBuilder:
    """
    Builder class for creating quantum circuits for different modalities.
    """
    
    @staticmethod
    def create_variational_circuit(n_qubits: int, n_layers: int, 
                                 feature_map: str = 'angle_encoding') -> QuantumCircuit:
        """
        Create a variational quantum circuit.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            feature_map: Type of feature encoding ('angle_encoding', 'amplitude_encoding')
            
        Returns:
            Parameterized quantum circuit
        """
        # Create parameter vectors
        input_params = ParameterVector('input', n_qubits)
        weight_params = ParameterVector('weights', n_layers * n_qubits * 3)
        
        qc = QuantumCircuit(n_qubits)
        
        # Feature encoding
        if feature_map == 'angle_encoding':
            for i in range(n_qubits):
                qc.ry(input_params[i], i)
        elif feature_map == 'amplitude_encoding':
            # Simplified amplitude encoding
            for i in range(n_qubits):
                qc.ry(input_params[i] * np.pi, i)
        
        # Variational layers
        param_idx = 0
        for layer in range(n_layers):
            # Rotation gates
            for i in range(n_qubits):
                qc.rx(weight_params[param_idx], i)
                param_idx += 1
                qc.ry(weight_params[param_idx], i)
                param_idx += 1
                qc.rz(weight_params[param_idx], i)
                param_idx += 1
            
            # Entangling gates
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            if n_qubits > 1:
                qc.cx(n_qubits - 1, 0)  # Circular entanglement
        
        return qc
    
    @staticmethod
    def create_quantum_convolutional_layer(n_qubits: int) -> QuantumCircuit:
        """
        Create a quantum convolutional layer.
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            Quantum convolutional circuit
        """
        params = ParameterVector('conv_params', n_qubits * 2)
        qc = QuantumCircuit(n_qubits)
        
        param_idx = 0
        # Apply local rotations
        for i in range(n_qubits):
            qc.ry(params[param_idx], i)
            param_idx += 1
        
        # Apply controlled rotations (convolution-like operation)
        for i in range(n_qubits - 1):
            qc.cry(params[param_idx], i, i + 1)
            param_idx += 1
        
        return qc

class PennyLaneCircuits:
    """
    PennyLane quantum circuits for gradient-based optimization.
    """
    
    @staticmethod
    def create_pennylane_device(n_qubits: int, shots: Optional[int] = None) -> qml.Device:
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