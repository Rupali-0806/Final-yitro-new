"""
Quantum tabular model for structured medical data processing.
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import Tuple, Optional
import logging
from ..utils.quantum_utils import PennyLaneCircuits, initialize_quantum_weights

logger = logging.getLogger(__name__)

class QuantumTabularModel(nn.Module):
    """
    Quantum machine learning model for tabular medical data.
    Uses quantum feature maps and variational circuits.
    """
    
    def __init__(self,
                 input_dim: int,
                 n_qubits: int = 8,
                 n_quantum_layers: int = 4,
                 quantum_dim: int = 32,
                 output_dim: int = 9,
                 feature_map_layers: int = 2):
        """
        Initialize Quantum Tabular model.
        
        Args:
            input_dim: Number of input features
            n_qubits: Number of qubits in quantum circuit
            n_quantum_layers: Number of quantum variational layers
            quantum_dim: Intermediate quantum feature dimension
            output_dim: Number of disease classes
            feature_map_layers: Layers in quantum feature map
        """
        super(QuantumTabularModel, self).__init__()
        
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_quantum_layers = n_quantum_layers
        self.quantum_dim = quantum_dim
        self.feature_map_layers = feature_map_layers
        
        # Classical preprocessing
        self.feature_projector = nn.Sequential(
            nn.Linear(input_dim, quantum_dim),
            nn.BatchNorm1d(quantum_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(quantum_dim, n_qubits),
            nn.Tanh()  # Bound features for quantum encoding
        )
        
        # Quantum circuit setup
        self.quantum_device = qml.device('default.qubit', wires=n_qubits)
        
        # Quantum weights - separate for feature map and variational layers
        self.feature_map_weights = nn.Parameter(
            torch.tensor(initialize_quantum_weights(feature_map_layers, n_qubits, (2,)), 
                        dtype=torch.float32)
        )
        
        self.variational_weights = nn.Parameter(
            torch.tensor(initialize_quantum_weights(n_quantum_layers, n_qubits), 
                        dtype=torch.float32)
        )
        
        # Create quantum circuits
        self.quantum_circuit = self._create_quantum_circuit()
        
        # Post-quantum classical layers
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, quantum_dim),
            nn.BatchNorm1d(quantum_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(quantum_dim, quantum_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(quantum_dim // 2, output_dim)
        )
        
        logger.info(f"QuantumTabularModel initialized: {input_dim} -> {n_qubits} qubits -> {output_dim}")
    
    def _create_quantum_circuit(self):
        """Create the quantum circuit for tabular data processing."""
        
        @qml.qnode(self.quantum_device, interface='torch')
        def circuit(inputs, fm_weights, var_weights):
            # Quantum feature map - encode classical data into quantum states
            for layer in range(self.feature_map_layers):
                # Amplitude encoding with rotation gates
                for i in range(self.n_qubits):
                    qml.RY(inputs[i] * fm_weights[layer, i, 0], wires=i)
                    qml.RZ(inputs[i] * fm_weights[layer, i, 1], wires=i)
                
                # Create entanglement patterns
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Add non-linear quantum features
                for i in range(self.n_qubits):
                    qml.RY(inputs[i] * inputs[(i + 1) % self.n_qubits] * 0.5, wires=i)
            
            # Variational quantum layers for learning
            for layer in range(self.n_quantum_layers):
                # Parameterized single-qubit rotations
                for i in range(self.n_qubits):
                    qml.RX(var_weights[layer, i, 0], wires=i)
                    qml.RY(var_weights[layer, i, 1], wires=i)
                    qml.RZ(var_weights[layer, i, 2], wires=i)
                
                # Entangling gates with different patterns per layer
                if layer % 2 == 0:
                    # Linear entanglement
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                else:
                    # Circular entanglement
                    for i in range(self.n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                
                # Add controlled rotations for expressivity
                for i in range(0, self.n_qubits - 1, 2):
                    qml.CRY(var_weights[layer, i, 0], wires=[i, i + 1])
            
            # Measurements - both Pauli-Z and Pauli-X for richer features
            measurements = []
            for i in range(self.n_qubits):
                measurements.append(qml.expval(qml.PauliZ(i)))
            
            return measurements
        
        return circuit
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum tabular model.
        
        Args:
            x: Input tabular features (batch_size, input_dim)
            
        Returns:
            Disease classification logits
        """
        batch_size = x.size(0)
        
        # Classical feature preprocessing
        classical_features = self.feature_projector(x)  # (batch_size, n_qubits)
        
        # Apply quantum processing
        quantum_outputs = []
        for i in range(batch_size):
            # Get features for this sample
            features = classical_features[i]
            
            # Apply quantum circuit
            quantum_result = self.quantum_circuit(
                features, 
                self.feature_map_weights, 
                self.variational_weights
            )
            quantum_outputs.append(torch.stack(quantum_result))
        
        quantum_features = torch.stack(quantum_outputs)  # (batch_size, n_qubits)
        
        # Final classification
        logits = self.classifier(quantum_features)
        
        return logits
    
    def extract_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract quantum features without classification.
        
        Args:
            x: Input tabular features
            
        Returns:
            Quantum features tensor
        """
        batch_size = x.size(0)
        
        # Classical feature preprocessing
        classical_features = self.feature_projector(x)
        
        # Apply quantum processing
        quantum_outputs = []
        for i in range(batch_size):
            features = classical_features[i]
            
            quantum_result = self.quantum_circuit(
                features, 
                self.feature_map_weights, 
                self.variational_weights
            )
            quantum_outputs.append(torch.stack(quantum_result))
        
        return torch.stack(quantum_outputs)
    
    def compute_quantum_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute quantum kernel between two sets of samples.
        
        Args:
            x1: First set of samples
            x2: Second set of samples
            
        Returns:
            Quantum kernel matrix
        """
        features1 = self.feature_projector(x1)
        features2 = self.feature_projector(x2)
        
        kernel_matrix = torch.zeros(x1.size(0), x2.size(0))
        
        for i, f1 in enumerate(features1):
            for j, f2 in enumerate(features2):
                # Create quantum state overlap circuit
                @qml.qnode(self.quantum_device, interface='torch')
                def kernel_circuit():
                    # Encode first sample
                    for k in range(self.n_qubits):
                        qml.RY(f1[k], wires=k)
                    
                    # Encode second sample (inverse)
                    for k in range(self.n_qubits):
                        qml.RY(-f2[k], wires=k)
                    
                    return qml.probs(wires=0)
                
                probs = kernel_circuit()
                kernel_matrix[i, j] = probs[0]  # Probability of |0⟩
        
        return kernel_matrix
    
    def get_circuit_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get quantum circuit weights for analysis."""
        return self.feature_map_weights.detach(), self.variational_weights.detach()

def build_quantum_tabular_model(input_dim: int,
                               n_qubits: int = 8,
                               n_layers: int = 4,
                               output_dim: int = 9) -> QuantumTabularModel:
    """
    Builder function for Quantum Tabular model.
    
    Args:
        input_dim: Number of input features
        n_qubits: Number of qubits
        n_layers: Number of quantum layers
        output_dim: Number of output classes
        
    Returns:
        Configured QuantumTabularModel
    """
    model = QuantumTabularModel(
        input_dim=input_dim,
        n_qubits=n_qubits,
        n_quantum_layers=n_layers,
        output_dim=output_dim
    )
    
    logger.info(f"Built Quantum Tabular model: {input_dim} -> {n_qubits} qubits -> {output_dim}")
    return model

if __name__ == "__main__":
    # Test the model
    logging.basicConfig(level=logging.INFO)
    
    input_dim = 16  # Example: 16 tabular features
    model = build_quantum_tabular_model(input_dim=input_dim)
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        outputs = model(test_input)
        print(f"Model output shape: {outputs.shape}")
        
        features = model.extract_quantum_features(test_input)
        print(f"Quantum features shape: {features.shape}")
        
        kernel = model.compute_quantum_kernel(test_input[:2], test_input[2:])
        print(f"Quantum kernel shape: {kernel.shape}")