"""
Demonstration script for Quantum Healthcare System.
This version works offline without requiring BERT downloads.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("🏥 Quantum Healthcare AI - Demonstration")
print("=" * 50)

def create_simple_tokenizer():
    """Create a simple tokenizer that doesn't require internet."""
    class SimpleTokenizer:
        def __init__(self, vocab_size=1000, max_length=128):
            self.vocab_size = vocab_size
            self.max_length = max_length
        
        def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors='pt'):
            if isinstance(texts, str):
                texts = [texts]
            
            max_len = max_length or self.max_length
            
            # Simple word-based tokenization (demo purposes)
            tokenized = []
            for text in texts:
                words = text.lower().split()[:max_len-2]  # Leave space for special tokens
                # Convert to token IDs (simple hash-based approach)
                token_ids = [101]  # [CLS] token
                for word in words:
                    token_id = abs(hash(word)) % (self.vocab_size - 2) + 2
                    token_ids.append(token_id)
                token_ids.append(102)  # [SEP] token
                
                # Pad to max_length
                while len(token_ids) < max_len:
                    token_ids.append(0)  # [PAD] token
                
                tokenized.append(token_ids[:max_len])
            
            # Create attention masks
            attention_masks = []
            for tokens in tokenized:
                mask = [1 if token != 0 else 0 for token in tokens]
                attention_masks.append(mask)
            
            result = {
                'input_ids': torch.tensor(tokenized),
                'attention_mask': torch.tensor(attention_masks)
            }
            
            return result
    
    return SimpleTokenizer()

def demo_quantum_circuit():
    """Demonstrate quantum circuit functionality."""
    print("\n🔬 Quantum Circuit Demonstration")
    print("-" * 30)
    
    # Create a simple quantum circuit for medical data processing
    n_qubits = 4
    n_layers = 2
    
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def medical_quantum_circuit(patient_data, weights):
        """Quantum circuit for processing patient vital signs."""
        
        # Encode patient data (normalized vital signs)
        for i in range(n_qubits):
            qml.RY(patient_data[i] * np.pi, wires=i)
        
        # Variational layers for learning medical patterns
        for layer in range(n_layers):
            # Parameterized rotations
            for i in range(n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)
            
            # Entanglement between vital signs
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])  # Ring connectivity
        
        # Measure quantum features
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Simulate patient data: [heart_rate, blood_pressure, temperature, oxygen_saturation]
    # Normalized to [-1, 1] range
    patient_examples = {
        "Healthy Patient": [0.0, 0.0, 0.0, 0.0],      # Normal vitals
        "Heart Disease": [0.6, 0.8, 0.1, -0.2],       # High HR, High BP
        "Pneumonia": [0.4, 0.2, 0.9, -0.6],           # Fever, Low O2
        "Sepsis": [0.8, -0.3, 0.7, -0.5]              # High HR, Low BP, Fever
    }
    
    # Random quantum weights for demonstration
    weights = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 3))
    
    print("Patient Data → Quantum Features:")
    for patient_type, vitals in patient_examples.items():
        quantum_features = medical_quantum_circuit(vitals, weights)
        features_rounded = [round(float(f), 3) for f in quantum_features]
        print(f"  {patient_type:15s}: {features_rounded}")
    
    return medical_quantum_circuit, weights

def demo_multimodal_fusion():
    """Demonstrate multimodal data fusion."""
    print("\n🔗 Multimodal Fusion Demonstration")
    print("-" * 30)
    
    # Simulate different modality features
    batch_size = 3
    
    # Text features (from clinical notes)
    text_features = torch.randn(batch_size, 8)  # 8-dimensional quantum text features
    
    # Tabular features (from vital signs)  
    tabular_features = torch.randn(batch_size, 8)  # 8-dimensional quantum tabular features
    
    # Image features (from chest X-rays)
    image_features = torch.randn(batch_size, 8)  # 8-dimensional quantum image features
    
    # Simple attention-based fusion
    class SimpleAttentionFusion(nn.Module):
        def __init__(self, feature_dim=8):
            super().__init__()
            self.attention_weights = nn.Parameter(torch.ones(3) / 3)  # Equal weights initially
            self.fusion_layer = nn.Linear(feature_dim, 9)  # 9 disease classes
        
        def forward(self, text_feat, tabular_feat, image_feat):
            # Normalize attention weights
            weights = torch.softmax(self.attention_weights, dim=0)
            
            # Weighted combination
            fused = (weights[0] * text_feat + 
                    weights[1] * tabular_feat + 
                    weights[2] * image_feat)
            
            # Final prediction
            return self.fusion_layer(fused)
    
    # Create and test fusion model
    fusion_model = SimpleAttentionFusion()
    
    with torch.no_grad():
        predictions = fusion_model(text_features, tabular_features, image_features)
        probabilities = torch.softmax(predictions, dim=1)
    
    disease_classes = [
        'Normal', 'Pneumonia', 'Heart Failure', 'Sepsis', 'COPD', 
        'Diabetes', 'Stroke', 'Kidney Injury', 'MI'
    ]
    
    print("Multimodal Predictions:")
    for i in range(batch_size):
        predicted_class = torch.argmax(probabilities[i]).item()
        confidence = probabilities[i, predicted_class].item()
        print(f"  Patient {i+1}: {disease_classes[predicted_class]} ({confidence:.1%} confidence)")
    
    # Show attention weights
    attention = torch.softmax(fusion_model.attention_weights, dim=0)
    print(f"\nModality Importance:")
    print(f"  Text: {attention[0]:.1%}")
    print(f"  Tabular: {attention[1]:.1%}")  
    print(f"  Image: {attention[2]:.1%}")

def demo_risk_assessment():
    """Demonstrate clinical risk assessment."""
    print("\n📊 Clinical Risk Assessment")
    print("-" * 30)
    
    # Sample patient data
    patient_cases = [
        {
            "name": "Patient A (Low Risk)",
            "age": 35,
            "vital_signs": {"heart_rate": 72, "bp_systolic": 118, "temperature": 98.6, "o2_sat": 98},
            "clinical_notes": "Routine check-up, no acute complaints",
            "risk_factors": []
        },
        {
            "name": "Patient B (Moderate Risk)", 
            "age": 67,
            "vital_signs": {"heart_rate": 95, "bp_systolic": 145, "temperature": 99.2, "o2_sat": 94},
            "clinical_notes": "Mild shortness of breath, fatigue",
            "risk_factors": ["age > 65", "elevated BP", "respiratory symptoms"]
        },
        {
            "name": "Patient C (High Risk)",
            "age": 78,
            "vital_signs": {"heart_rate": 115, "bp_systolic": 90, "temperature": 102.1, "o2_sat": 88},
            "clinical_notes": "Severe chest pain, difficulty breathing, altered mental status",
            "risk_factors": ["age > 75", "hypotension", "fever", "hypoxia", "chest pain"]
        }
    ]
    
    def calculate_risk_score(patient):
        score = 0
        
        # Age-based risk
        if patient["age"] > 75:
            score += 3
        elif patient["age"] > 65:
            score += 2
        elif patient["age"] > 50:
            score += 1
        
        # Vital sign based risk
        vitals = patient["vital_signs"]
        if vitals["heart_rate"] > 100:
            score += 2
        if vitals["bp_systolic"] < 100 or vitals["bp_systolic"] > 160:
            score += 2
        if vitals["temperature"] > 100.4:
            score += 2
        if vitals["o2_sat"] < 92:
            score += 3
        
        # Clinical factor risk
        score += len(patient["risk_factors"])
        
        return min(score, 10)  # Cap at 10
    
    print("Risk Assessment Results:")
    for patient in patient_cases:
        risk_score = calculate_risk_score(patient)
        
        if risk_score <= 3:
            risk_level = "LOW"
            recommendation = "Routine care"
        elif risk_score <= 6:
            risk_level = "MODERATE" 
            recommendation = "Close monitoring"
        else:
            risk_level = "HIGH"
            recommendation = "Immediate intervention"
        
        print(f"\n{patient['name']}:")
        print(f"  Risk Score: {risk_score}/10")
        print(f"  Risk Level: {risk_level}")
        print(f"  Recommendation: {recommendation}")
        print(f"  Key Factors: {', '.join(patient['risk_factors']) if patient['risk_factors'] else 'None'}")

def demo_ui_simulation():
    """Simulate the UI functionality."""
    print("\n🖥️  UI Interface Simulation")
    print("-" * 30)
    
    print("Quantum Healthcare AI Web Interface would include:")
    print("✅ Patient data input forms:")
    print("   • Clinical notes text area")
    print("   • Vital signs numerical inputs")
    print("   • Chest X-ray image upload")
    
    print("\n✅ Real-time quantum processing:")
    print("   • Text → Quantum NLP features")
    print("   • Vitals → Quantum tabular features") 
    print("   • Images → Hybrid CNN-Quantum features")
    print("   • Fusion → Disease prediction")
    
    print("\n✅ Interactive results display:")
    print("   • Disease prediction with confidence")
    print("   • Risk assessment visualization")
    print("   • Clinical summary generation")
    print("   • Downloadable reports")
    
    print("\n✅ Sample cases and examples:")
    print("   • Pre-loaded patient scenarios")
    print("   • Educational disease information")
    print("   • Model performance metrics")
    
    print("\n🚀 Launch commands:")
    print("   Streamlit: streamlit run app.py")
    print("   Gradio:    python app_gradio.py")

def main():
    """Run the complete demonstration."""
    
    # Core quantum functionality
    quantum_circuit, weights = demo_quantum_circuit()
    
    # Multimodal processing
    demo_multimodal_fusion()
    
    # Clinical assessment
    demo_risk_assessment()
    
    # UI overview
    demo_ui_simulation()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 DEMONSTRATION SUMMARY")
    print("=" * 50)
    print("✅ Quantum circuits for medical data processing")
    print("✅ Multimodal fusion (text + tabular + image)")
    print("✅ Clinical risk assessment algorithms")
    print("✅ Complete web interface ready")
    print("✅ End-to-end disease prediction pipeline")
    
    print("\n📋 Next Steps:")
    print("1. Install full dependencies: pip install -r requirements.txt")
    print("2. Run training pipeline: python train.py")
    print("3. Launch web interface: streamlit run app.py")
    print("4. Test with real medical data")
    
    print("\n⚕️ Note: This is a research prototype for educational purposes.")
    print("Always consult healthcare professionals for medical decisions.")

if __name__ == "__main__":
    main()