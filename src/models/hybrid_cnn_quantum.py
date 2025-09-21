"""
Hybrid CNN-Quantum model for medical image processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Tuple, Optional
import logging
from ..utils.quantum_utils import PennyLaneCircuits, initialize_quantum_weights

logger = logging.getLogger(__name__)

class QuantumConvolutionalLayer(nn.Module):
    """
    Quantum convolutional layer for image processing.
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super(QuantumConvolutionalLayer, self).__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Quantum device for convolution
        self.quantum_device = qml.device('default.qubit', wires=n_qubits)
        
        # Quantum weights
        self.quantum_weights = nn.Parameter(
            torch.tensor(initialize_quantum_weights(n_layers, n_qubits), 
                        dtype=torch.float32)
        )
        
        # Create quantum convolution circuit
        self.quantum_conv = self._create_quantum_conv_circuit()
    
    def _create_quantum_conv_circuit(self):
        """Create quantum convolutional circuit."""
        
        @qml.qnode(self.quantum_device, interface='torch')
        def conv_circuit(patch, weights):
            # Encode 2x2 patch into quantum state
            for i in range(min(len(patch), self.n_qubits)):
                qml.RY(patch[i], wires=i)
            
            # Quantum convolution layers
            for layer in range(self.n_layers):
                # Local rotations
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                
                # Nearest neighbor interactions (convolution-like)
                for i in range(0, self.n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
                
                # Cross interactions
                if self.n_qubits >= 4:
                    qml.CNOT(wires=[0, 2])
                    qml.CNOT(wires=[1, 3])
            
            # Measure first qubit as convolution output
            return qml.expval(qml.PauliZ(0))
        
        return conv_circuit
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantum convolution to input.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Quantum convolution output
        """
        batch_size, channels, height, width = x.shape
        
        # Apply quantum convolution to 2x2 patches
        output_height = height - 1
        output_width = width - 1
        
        output = torch.zeros(batch_size, channels, output_height, output_width)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Extract 2x2 patch
                        patch = x[b, c, i:i+2, j:j+2].flatten()
                        
                        # Apply quantum circuit
                        result = self.quantum_conv(patch, self.quantum_weights)
                        output[b, c, i, j] = result
        
        return output

class HybridCNNQuantumModel(nn.Module):
    """
    Hybrid CNN-Quantum model for chest X-ray analysis.
    Combines classical convolutional layers with quantum processing.
    """
    
    def __init__(self,
                 n_qubits: int = 8,
                 n_quantum_layers: int = 3,
                 quantum_conv_qubits: int = 4,
                 output_dim: int = 9,
                 use_quantum_conv: bool = True):
        """
        Initialize Hybrid CNN-Quantum model.
        
        Args:
            n_qubits: Number of qubits in main quantum circuit
            n_quantum_layers: Number of quantum variational layers
            quantum_conv_qubits: Number of qubits for quantum convolution
            output_dim: Number of disease classes
            use_quantum_conv: Whether to use quantum convolution layers
        """
        super(HybridCNNQuantumModel, self).__init__()
        
        self.n_qubits = n_qubits
        self.n_quantum_layers = n_quantum_layers
        self.use_quantum_conv = use_quantum_conv
        
        # Classical CNN backbone
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Quantum convolution layer (optional)
        if use_quantum_conv:
            self.quantum_conv = QuantumConvolutionalLayer(quantum_conv_qubits, 2)
            self.quantum_conv_projection = nn.Conv2d(128, 64, kernel_size=1)
        
        # Pooling and normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        
        # Classical feature reduction
        self.feature_reducer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),  # After adaptive pooling
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_qubits),
            nn.Tanh()  # Bound for quantum encoding
        )
        
        # Quantum processing setup
        self.quantum_device = qml.device('default.qubit', wires=n_qubits)
        self.quantum_weights = nn.Parameter(
            torch.tensor(initialize_quantum_weights(n_quantum_layers, n_qubits), 
                        dtype=torch.float32)
        )
        
        # Create main quantum circuit
        self.quantum_circuit = self._create_quantum_circuit()
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, n_qubits * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_qubits * 2, n_qubits),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_qubits, output_dim)
        )
        
        logger.info(f"HybridCNNQuantumModel initialized with {n_qubits} qubits, quantum_conv={use_quantum_conv}")
    
    def _create_quantum_circuit(self):
        """Create the main quantum circuit for feature processing."""
        
        @qml.qnode(self.quantum_device, interface='torch')
        def circuit(features, weights):
            # Quantum feature encoding
            for i in range(self.n_qubits):
                qml.RY(features[i] * np.pi, wires=i)
            
            # Variational quantum layers
            for layer in range(self.n_quantum_layers):
                # Parameterized rotations
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                
                # Entangling patterns for image-like connectivity
                # Create 2D-like connectivity for spatial relationships
                if self.n_qubits >= 4:
                    # Horizontal connections
                    for i in range(0, self.n_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])
                    
                    # Vertical connections
                    for i in range(self.n_qubits // 2):
                        if i + self.n_qubits // 2 < self.n_qubits:
                            qml.CNOT(wires=[i, i + self.n_qubits // 2])
                else:
                    # Linear entanglement for smaller systems
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                
                # Add controlled rotations for expressivity
                for i in range(0, self.n_qubits - 1, 2):
                    qml.CRY(weights[layer, i, 1], wires=[i, (i + 1) % self.n_qubits])
            
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid CNN-Quantum model.
        
        Args:
            x: Input images (batch_size, 1, height, width)
            
        Returns:
            Disease classification logits
        """
        batch_size = x.size(0)
        
        # Classical CNN feature extraction
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(x)
        
        # Optional quantum convolution
        if self.use_quantum_conv:
            # Apply quantum convolution to a subset of feature maps
            x_subset = x[:, :64, :, :]  # Use first 64 channels
            x_quantum = self.quantum_conv(x_subset)
            x_quantum = self.quantum_conv_projection(x_quantum)
            
            # Combine quantum and classical features
            x_classical = x[:, 64:, :, :]
            x = torch.cat([x_quantum, x_classical], dim=1)
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        
        # Reduce to quantum-compatible features
        classical_features = self.feature_reducer(x)  # (batch_size, n_qubits)
        
        # Apply quantum processing
        quantum_outputs = []
        for i in range(batch_size):
            features = classical_features[i]
            
            # Apply quantum circuit
            quantum_result = self.quantum_circuit(features, self.quantum_weights)
            quantum_outputs.append(torch.stack(quantum_result))
        
        quantum_features = torch.stack(quantum_outputs)  # (batch_size, n_qubits)
        
        # Final classification
        logits = self.classifier(quantum_features)
        
        return logits
    
    def extract_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN features before quantum processing."""
        with torch.no_grad():
            x = F.relu(self.batch_norm1(self.conv1(x)))
            x = self.pool(x)
            
            x = F.relu(self.batch_norm2(self.conv2(x)))
            x = self.pool(x)
            
            x = F.relu(self.batch_norm3(self.conv3(x)))
            x = self.pool(x)
            
            if self.use_quantum_conv:
                x_subset = x[:, :64, :, :]
                x_quantum = self.quantum_conv(x_subset)
                x_quantum = self.quantum_conv_projection(x_quantum)
                
                x_classical = x[:, 64:, :, :]
                x = torch.cat([x_quantum, x_classical], dim=1)
            
            x = self.adaptive_pool(x)
            
        return x
    
    def extract_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract quantum features from images."""
        batch_size = x.size(0)
        
        # Get CNN features
        cnn_features = self.extract_cnn_features(x)
        classical_features = self.feature_reducer(cnn_features)
        
        # Apply quantum processing
        quantum_outputs = []
        for i in range(batch_size):
            features = classical_features[i]
            quantum_result = self.quantum_circuit(features, self.quantum_weights)
            quantum_outputs.append(torch.stack(quantum_result))
        
        return torch.stack(quantum_outputs)
    
    def get_quantum_weights(self) -> torch.Tensor:
        """Get quantum circuit weights."""
        return self.quantum_weights.detach()

def build_hybrid_cnn_quantum_model(n_qubits: int = 8,
                                  n_layers: int = 3,
                                  output_dim: int = 9,
                                  use_quantum_conv: bool = True) -> HybridCNNQuantumModel:
    """
    Builder function for Hybrid CNN-Quantum model.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of quantum layers
        output_dim: Number of output classes
        use_quantum_conv: Whether to use quantum convolution
        
    Returns:
        Configured HybridCNNQuantumModel
    """
    model = HybridCNNQuantumModel(
        n_qubits=n_qubits,
        n_quantum_layers=n_layers,
        output_dim=output_dim,
        use_quantum_conv=use_quantum_conv
    )
    
    logger.info(f"Built Hybrid CNN-Quantum model with {n_qubits} qubits, quantum_conv={use_quantum_conv}")
    return model

if __name__ == "__main__":
    # Test the model
    logging.basicConfig(level=logging.INFO)
    
    model = build_hybrid_cnn_quantum_model()
    
    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 224, 224)  # Chest X-ray size
    
    with torch.no_grad():
        outputs = model(test_input)
        print(f"Model output shape: {outputs.shape}")
        
        cnn_features = model.extract_cnn_features(test_input)
        print(f"CNN features shape: {cnn_features.shape}")
        
        quantum_features = model.extract_quantum_features(test_input)
        print(f"Quantum features shape: {quantum_features.shape}")