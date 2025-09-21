"""
Quantum Natural Language Processing model for clinical text analysis.
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple, Optional
import logging
from ..utils.quantum_utils import PennyLaneCircuits, initialize_quantum_weights

logger = logging.getLogger(__name__)

class QuantumNLPModel(nn.Module):
    """
    Quantum-enhanced NLP model for clinical text processing.
    Combines classical BERT embeddings with quantum processing.
    """
    
    def __init__(self, 
                 bert_model_name: str = 'bert-base-uncased',
                 n_qubits: int = 8,
                 n_quantum_layers: int = 3,
                 classical_dim: int = 768,
                 quantum_dim: int = 64,
                 output_dim: int = 9):
        """
        Initialize Quantum NLP model.
        
        Args:
            bert_model_name: Pre-trained BERT model name
            n_qubits: Number of qubits in quantum circuit
            n_quantum_layers: Number of quantum variational layers
            classical_dim: BERT embedding dimension
            quantum_dim: Quantum feature dimension
            output_dim: Number of disease classes
        """
        super(QuantumNLPModel, self).__init__()
        
        self.n_qubits = n_qubits
        self.n_quantum_layers = n_quantum_layers
        self.quantum_dim = quantum_dim
        
        # Classical BERT encoder
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # Freeze BERT parameters for faster training
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Classical dimension reduction
        self.classical_projector = nn.Sequential(
            nn.Linear(classical_dim, quantum_dim),
            nn.LayerNorm(quantum_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Quantum circuit setup
        self.quantum_device = qml.device('default.qubit', wires=n_qubits)
        self.quantum_weights = nn.Parameter(
            torch.tensor(initialize_quantum_weights(n_quantum_layers, n_qubits), 
                        dtype=torch.float32)
        )
        
        # Create quantum circuit
        self.quantum_circuit = self._create_quantum_circuit()
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim // 2, output_dim)
        )
        
        logger.info(f"QuantumNLPModel initialized with {n_qubits} qubits, {n_quantum_layers} layers")
    
    def _create_quantum_circuit(self):
        """Create the quantum circuit for text processing."""
        
        @qml.qnode(self.quantum_device, interface='torch')
        def circuit(inputs, weights):
            # Feature encoding - map classical features to quantum states
            for i in range(min(len(inputs), self.n_qubits)):
                qml.RY(inputs[i], wires=i)
            
            # Variational quantum layers
            for layer in range(self.n_quantum_layers):
                # Parameterized rotations
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                
                # Entangling gates
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Add global entanglement
                if self.n_qubits > 2:
                    qml.CNOT(wires=[0, self.n_qubits - 1])
            
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum NLP model.
        
        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask for inputs
            
        Returns:
            Disease classification logits
        """
        batch_size = input_ids.size(0)
        
        # Classical BERT encoding
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token embedding
            cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
        
        # Project to quantum dimension
        classical_features = self.classical_projector(cls_embeddings)  # (batch_size, quantum_dim)
        
        # Apply quantum processing
        quantum_outputs = []
        for i in range(batch_size):
            # Normalize features for quantum encoding
            features = classical_features[i]
            features = torch.tanh(features) * np.pi  # Scale to [-π, π]
            
            # Apply quantum circuit
            quantum_result = self.quantum_circuit(features[:self.n_qubits], self.quantum_weights)
            quantum_outputs.append(torch.stack(quantum_result))
        
        quantum_features = torch.stack(quantum_outputs)  # (batch_size, n_qubits)
        
        # Final classification
        logits = self.classifier(quantum_features)
        
        return logits
    
    def extract_quantum_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract quantum features without classification.
        
        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask for inputs
            
        Returns:
            Quantum features tensor
        """
        batch_size = input_ids.size(0)
        
        # Classical BERT encoding
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]
        
        # Project to quantum dimension
        classical_features = self.classical_projector(cls_embeddings)
        
        # Apply quantum processing
        quantum_outputs = []
        for i in range(batch_size):
            features = classical_features[i]
            features = torch.tanh(features) * np.pi
            
            quantum_result = self.quantum_circuit(features[:self.n_qubits], self.quantum_weights)
            quantum_outputs.append(torch.stack(quantum_result))
        
        return torch.stack(quantum_outputs)
    
    def get_attention_weights(self) -> torch.Tensor:
        """Get quantum circuit weights for interpretability."""
        return self.quantum_weights.detach()

def build_quantum_nlp_model(n_qubits: int = 8, 
                           n_layers: int = 3,
                           output_dim: int = 9) -> QuantumNLPModel:
    """
    Builder function for Quantum NLP model.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of quantum layers
        output_dim: Number of output classes
        
    Returns:
        Configured QuantumNLPModel
    """
    model = QuantumNLPModel(
        n_qubits=n_qubits,
        n_quantum_layers=n_layers,
        output_dim=output_dim
    )
    
    logger.info(f"Built Quantum NLP model with {n_qubits} qubits")
    return model

if __name__ == "__main__":
    # Test the model
    logging.basicConfig(level=logging.INFO)
    
    model = build_quantum_nlp_model()
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        print(f"Model output shape: {outputs.shape}")
        
        features = model.extract_quantum_features(input_ids, attention_mask)
        print(f"Quantum features shape: {features.shape}")