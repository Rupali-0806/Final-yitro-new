"""
Multimodal fusion layer for combining quantum features from different modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from ..utils.quantum_utils import PennyLaneCircuits, initialize_quantum_weights

logger = logging.getLogger(__name__)

class AttentionFusion(nn.Module):
    """
    Attention-based fusion mechanism for multimodal features.
    """
    
    def __init__(self, feature_dims: List[int], hidden_dim: int = 128):
        super(AttentionFusion, self).__init__()
        
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        
        # Attention networks for each modality
        self.attention_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for dim in feature_dims
        ])
        
        # Feature projection networks
        self.projection_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ) for dim in feature_dims
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply attention-based fusion.
        
        Args:
            features: List of feature tensors from different modalities
            
        Returns:
            Fused feature tensor
        """
        # Project features to common dimension
        projected_features = []
        attention_weights = []
        
        for i, (feat, proj_net, att_net) in enumerate(zip(features, self.projection_networks, self.attention_networks)):
            projected = proj_net(feat)
            attention = att_net(feat)
            
            projected_features.append(projected)
            attention_weights.append(attention)
        
        # Normalize attention weights
        attention_weights = torch.stack(attention_weights, dim=1)  # (batch_size, n_modalities, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        weighted_features = []
        for i, feat in enumerate(projected_features):
            weighted = feat * attention_weights[:, i, :]
            weighted_features.append(weighted)
        
        # Combine features
        fused_features = torch.stack(weighted_features, dim=1).sum(dim=1)
        
        return fused_features

class QuantumFusion(nn.Module):
    """
    Quantum-based fusion mechanism using quantum interference.
    """
    
    def __init__(self, feature_dims: List[int], n_qubits: int = 6, n_layers: int = 2):
        super(QuantumFusion, self).__init__()
        
        self.feature_dims = feature_dims
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_modalities = len(feature_dims)
        
        # Project each modality to quantum dimension
        self.quantum_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, n_qubits),
                nn.Tanh()  # Bound for quantum encoding
            ) for dim in feature_dims
        ])
        
        # Quantum device and weights
        self.quantum_device = qml.device('default.qubit', wires=n_qubits)
        self.quantum_weights = nn.Parameter(
            torch.tensor(initialize_quantum_weights(n_layers, n_qubits), 
                        dtype=torch.float32)
        )
        
        # Fusion weights for combining modalities
        self.fusion_weights = nn.Parameter(torch.ones(self.n_modalities) / self.n_modalities)
        
        # Create quantum fusion circuit
        self.quantum_fusion_circuit = self._create_fusion_circuit()
    
    def _create_fusion_circuit(self):
        """Create quantum circuit for multimodal fusion."""
        
        @qml.qnode(self.quantum_device, interface='torch')
        def fusion_circuit(modal_features, weights, fusion_weights):
            # Encode each modality with weighted superposition
            for i in range(self.n_qubits):
                # Initialize qubit state
                qml.RY(0.0, wires=i)
                
                # Superpose features from all modalities
                for mod_idx in range(self.n_modalities):
                    if i < len(modal_features[mod_idx]):
                        angle = modal_features[mod_idx][i] * fusion_weights[mod_idx] * np.pi
                        qml.RY(angle, wires=i)
            
            # Quantum interference layers for fusion
            for layer in range(self.n_layers):
                # Parameterized rotations
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                
                # Create entanglement between modalities
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Cross-modal entanglement
                if self.n_qubits > 2:
                    qml.CNOT(wires=[0, self.n_qubits - 1])
                
                # Controlled rotations for feature interaction
                for i in range(0, self.n_qubits - 1, 2):
                    qml.CRY(weights[layer, i, 1], wires=[i, i + 1])
            
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return fusion_circuit
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply quantum fusion to multimodal features.
        
        Args:
            features: List of feature tensors from different modalities
            
        Returns:
            Quantum-fused feature tensor
        """
        batch_size = features[0].size(0)
        
        # Project to quantum dimensions
        quantum_features = []
        for i, (feat, projector) in enumerate(zip(features, self.quantum_projectors)):
            quantum_feat = projector(feat)
            quantum_features.append(quantum_feat)
        
        # Apply quantum fusion
        fused_outputs = []
        for b in range(batch_size):
            # Get features for this batch
            batch_features = [qf[b] for qf in quantum_features]
            
            # Apply quantum fusion circuit
            quantum_result = self.quantum_fusion_circuit(
                batch_features, 
                self.quantum_weights,
                F.softmax(self.fusion_weights, dim=0)
            )
            fused_outputs.append(torch.stack(quantum_result))
        
        return torch.stack(fused_outputs)

class MultimodalFusionLayer(nn.Module):
    """
    Complete multimodal fusion layer with multiple fusion strategies.
    """
    
    def __init__(self,
                 text_dim: int = 8,
                 tabular_dim: int = 8,
                 image_dim: int = 8,
                 fusion_type: str = 'hybrid',
                 hidden_dim: int = 128,
                 output_dim: int = 9,
                 quantum_fusion_qubits: int = 6):
        """
        Initialize Multimodal Fusion Layer.
        
        Args:
            text_dim: Dimension of text features
            tabular_dim: Dimension of tabular features
            image_dim: Dimension of image features
            fusion_type: Type of fusion ('attention', 'quantum', 'hybrid', 'concat')
            hidden_dim: Hidden dimension for fusion
            output_dim: Final output dimension
            quantum_fusion_qubits: Number of qubits for quantum fusion
        """
        super(MultimodalFusionLayer, self).__init__()
        
        self.fusion_type = fusion_type
        self.feature_dims = [text_dim, tabular_dim, image_dim]
        
        # Different fusion strategies
        if fusion_type == 'attention':
            self.fusion_layer = AttentionFusion(self.feature_dims, hidden_dim)
            fusion_output_dim = hidden_dim
            
        elif fusion_type == 'quantum':
            self.fusion_layer = QuantumFusion(self.feature_dims, quantum_fusion_qubits)
            fusion_output_dim = quantum_fusion_qubits
            
        elif fusion_type == 'hybrid':
            # Combine attention and quantum fusion
            self.attention_fusion = AttentionFusion(self.feature_dims, hidden_dim)
            self.quantum_fusion = QuantumFusion(self.feature_dims, quantum_fusion_qubits)
            fusion_output_dim = hidden_dim + quantum_fusion_qubits
            
        elif fusion_type == 'concat':
            # Simple concatenation
            self.fusion_layer = None
            fusion_output_dim = sum(self.feature_dims)
            
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Post-fusion processing
        self.post_fusion = nn.Sequential(
            nn.Linear(fusion_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Modality importance weights (learnable)
        self.modality_weights = nn.Parameter(torch.ones(3) / 3)
        
        logger.info(f"MultimodalFusionLayer initialized with {fusion_type} fusion")
    
    def forward(self, text_features: torch.Tensor, 
                tabular_features: torch.Tensor, 
                image_features: torch.Tensor) -> torch.Tensor:
        """
        Perform multimodal fusion.
        
        Args:
            text_features: Text quantum features
            tabular_features: Tabular quantum features
            image_features: Image quantum features
            
        Returns:
            Fused prediction logits
        """
        # Apply modality weights
        weighted_features = [
            text_features * self.modality_weights[0],
            tabular_features * self.modality_weights[1],
            image_features * self.modality_weights[2]
        ]
        
        # Apply fusion strategy
        if self.fusion_type == 'concat':
            fused_features = torch.cat(weighted_features, dim=1)
            
        elif self.fusion_type == 'attention':
            fused_features = self.fusion_layer(weighted_features)
            
        elif self.fusion_type == 'quantum':
            fused_features = self.fusion_layer(weighted_features)
            
        elif self.fusion_type == 'hybrid':
            # Combine attention and quantum fusion
            attention_fused = self.attention_fusion(weighted_features)
            quantum_fused = self.quantum_fusion(weighted_features)
            fused_features = torch.cat([attention_fused, quantum_fused], dim=1)
        
        # Final prediction
        output = self.post_fusion(fused_features)
        
        return output
    
    def get_modality_importance(self) -> torch.Tensor:
        """Get normalized modality importance weights."""
        return F.softmax(self.modality_weights, dim=0)
    
    def get_fusion_features(self, text_features: torch.Tensor, 
                           tabular_features: torch.Tensor, 
                           image_features: torch.Tensor) -> torch.Tensor:
        """Get fused features without final classification."""
        weighted_features = [
            text_features * self.modality_weights[0],
            tabular_features * self.modality_weights[1],
            image_features * self.modality_weights[2]
        ]
        
        if self.fusion_type == 'concat':
            return torch.cat(weighted_features, dim=1)
        elif self.fusion_type == 'attention':
            return self.fusion_layer(weighted_features)
        elif self.fusion_type == 'quantum':
            return self.fusion_layer(weighted_features)
        elif self.fusion_type == 'hybrid':
            attention_fused = self.attention_fusion(weighted_features)
            quantum_fused = self.quantum_fusion(weighted_features)
            return torch.cat([attention_fused, quantum_fused], dim=1)

def create_fusion_layer(text_dim: int = 8,
                       tabular_dim: int = 8,
                       image_dim: int = 8,
                       fusion_type: str = 'hybrid',
                       output_dim: int = 9) -> MultimodalFusionLayer:
    """
    Create a multimodal fusion layer.
    
    Args:
        text_dim: Text feature dimension
        tabular_dim: Tabular feature dimension
        image_dim: Image feature dimension
        fusion_type: Fusion strategy
        output_dim: Output classes
        
    Returns:
        Configured fusion layer
    """
    fusion_layer = MultimodalFusionLayer(
        text_dim=text_dim,
        tabular_dim=tabular_dim,
        image_dim=image_dim,
        fusion_type=fusion_type,
        output_dim=output_dim
    )
    
    logger.info(f"Created {fusion_type} fusion layer")
    return fusion_layer

if __name__ == "__main__":
    # Test the fusion layer
    logging.basicConfig(level=logging.INFO)
    
    # Test different fusion types
    fusion_types = ['attention', 'quantum', 'hybrid', 'concat']
    
    for fusion_type in fusion_types:
        print(f"\nTesting {fusion_type} fusion:")
        
        fusion_layer = create_fusion_layer(fusion_type=fusion_type)
        
        # Create test features
        batch_size = 4
        text_feat = torch.randn(batch_size, 8)
        tabular_feat = torch.randn(batch_size, 8)
        image_feat = torch.randn(batch_size, 8)
        
        with torch.no_grad():
            output = fusion_layer(text_feat, tabular_feat, image_feat)
            print(f"  Output shape: {output.shape}")
            
            if fusion_type != 'concat':
                fused_feat = fusion_layer.get_fusion_features(text_feat, tabular_feat, image_feat)
                print(f"  Fused features shape: {fused_feat.shape}")
            
            importance = fusion_layer.get_modality_importance()
            print(f"  Modality importance: {importance.detach().numpy()}")