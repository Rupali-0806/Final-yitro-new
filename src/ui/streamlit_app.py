"""
Streamlit web interface for quantum healthcare disease prediction.
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import PIL.Image
from io import BytesIO
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.quantum_nlp import build_quantum_nlp_model
from src.models.quantum_tabular import build_quantum_tabular_model
from src.models.hybrid_cnn_quantum import build_hybrid_cnn_quantum_model
from src.models.fusion_layer import create_fusion_layer
from src.data.preprocessor import MIMICPreprocessor
from src.utils.medical_utils import MedicalDataUtils, DiseaseClassificationUtils

logger = logging.getLogger(__name__)

@st.cache_resource
def load_trained_models(model_dir: str = "outputs/models"):
    """Load trained quantum models."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model configuration (should match training config)
        config = {
            'text_qubits': 8,
            'text_layers': 3,
            'tabular_qubits': 8,
            'tabular_layers': 4,
            'tabular_input_dim': 16,
            'image_qubits': 8,
            'image_layers': 3,
            'n_classes': 9
        }
        
        # Build models
        text_model = build_quantum_nlp_model(
            n_qubits=config['text_qubits'],
            n_layers=config['text_layers'],
            output_dim=config['n_classes']
        ).to(device)
        
        tabular_model = build_quantum_tabular_model(
            input_dim=config['tabular_input_dim'],
            n_qubits=config['tabular_qubits'],
            n_layers=config['tabular_layers'],
            output_dim=config['n_classes']
        ).to(device)
        
        image_model = build_hybrid_cnn_quantum_model(
            n_qubits=config['image_qubits'],
            n_layers=config['image_layers'],
            output_dim=config['n_classes']
        ).to(device)
        
        fusion_model = create_fusion_layer(
            text_dim=config['text_qubits'],
            tabular_dim=config['tabular_qubits'],
            image_dim=config['image_qubits'],
            fusion_type='hybrid',
            output_dim=config['n_classes']
        ).to(device)
        
        # Load weights if available
        model_path = Path(model_dir)
        if model_path.exists():
            try:
                text_model.load_state_dict(torch.load(model_path / 'best_text_model.pth', map_location=device))
                tabular_model.load_state_dict(torch.load(model_path / 'best_tabular_model.pth', map_location=device))
                image_model.load_state_dict(torch.load(model_path / 'best_image_model.pth', map_location=device))
                fusion_model.load_state_dict(torch.load(model_path / 'best_fusion_model.pth', map_location=device))
                st.success("Trained models loaded successfully!")
            except FileNotFoundError:
                st.warning("Pre-trained models not found. Using randomly initialized models for demonstration.")
        else:
            st.warning("Model directory not found. Using randomly initialized models for demonstration.")
        
        # Set to evaluation mode
        text_model.eval()
        tabular_model.eval()
        image_model.eval()
        fusion_model.eval()
        
        return {
            'text': text_model,
            'tabular': tabular_model,
            'image': image_model,
            'fusion': fusion_model,
            'device': device
        }
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

@st.cache_resource
def load_preprocessor():
    """Load data preprocessor."""
    return MIMICPreprocessor()

def process_clinical_text(text: str, preprocessor: MIMICPreprocessor) -> Dict[str, torch.Tensor]:
    """Process clinical text input."""
    # Create a temporary DataFrame for processing
    text_df = pd.DataFrame({'clinical_notes': [text]})
    
    # Preprocess text
    processed_text = preprocessor.preprocess_text_data(text_df)
    
    return processed_text

def process_tabular_data(vital_signs: Dict[str, float]) -> torch.Tensor:
    """Process tabular vital signs data."""
    # Create DataFrame from input
    tabular_df = pd.DataFrame([vital_signs])
    
    # Add gender encoding (default to male for demo)
    tabular_df['gender_encoded'] = 1
    
    # Ensure all required columns exist
    required_cols = ['age', 'heart_rate', 'blood_pressure_systolic', 
                    'blood_pressure_diastolic', 'temperature', 'respiratory_rate',
                    'oxygen_saturation', 'white_blood_cell_count', 'hemoglobin',
                    'platelets', 'creatinine', 'bun', 'glucose', 'los_hours', 'gender_encoded']
    
    for col in required_cols:
        if col not in tabular_df.columns:
            if col == 'los_hours':
                tabular_df[col] = 24  # Default to 24 hours
            elif col == 'gender_encoded':
                tabular_df[col] = 1  # Default to male
            else:
                tabular_df[col] = 0  # Default value
    
    # Normalize using simple z-score (in production, use trained scaler)
    numeric_cols = [col for col in required_cols if col != 'gender_encoded']
    tabular_df[numeric_cols] = (tabular_df[numeric_cols] - tabular_df[numeric_cols].mean()) / (tabular_df[numeric_cols].std() + 1e-8)
    
    return torch.FloatTensor(tabular_df[required_cols].values)

def process_image_data(image: PIL.Image.Image) -> torch.Tensor:
    """Process chest X-ray image."""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to model input size
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch and channel dimensions
    image_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
    
    return image_tensor

def predict_disease(models: Dict, text_input: str, tabular_input: Dict, 
                   image_input: Optional[PIL.Image.Image], preprocessor: MIMICPreprocessor) -> Tuple[str, float, Dict]:
    """Make disease prediction using multimodal inputs."""
    device = models['device']
    
    try:
        with torch.no_grad():
            # Process inputs
            text_data = process_clinical_text(text_input, preprocessor)
            tabular_data = process_tabular_data(tabular_input)
            
            # Extract quantum features
            text_features = models['text'].extract_quantum_features(
                text_data['input_ids'].to(device),
                text_data['attention_mask'].to(device)
            )
            
            tabular_features = models['tabular'].extract_quantum_features(
                tabular_data.to(device)
            )
            
            # Handle image input
            if image_input is not None:
                image_data = process_image_data(image_input)
                image_features = models['image'].extract_quantum_features(
                    image_data.to(device)
                )
            else:
                # Use zero features if no image provided
                image_features = torch.zeros(1, 8).to(device)
            
            # Fusion prediction
            fusion_output = models['fusion'](text_features, tabular_features, image_features)
            probabilities = torch.softmax(fusion_output, dim=1)
            
            # Get prediction
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            # Disease classes
            disease_classes = [
                'Pneumonia', 'Heart Failure', 'Sepsis', 'COPD', 'Diabetes', 
                'Acute Kidney Injury', 'Stroke', 'Myocardial Infarction', 'Normal'
            ]
            
            predicted_disease = disease_classes[predicted_class]
            
            # Create risk profile
            risk_profile = MedicalDataUtils.create_disease_risk_profile(
                pd.DataFrame([tabular_input]), 
                MedicalDataUtils.extract_clinical_keywords(text_input)
            )
            
            return predicted_disease, confidence, risk_profile
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0, {}

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Quantum Healthcare AI",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 Quantum Healthcare AI")
    st.subheader("Multimodal Disease Prediction System")
    
    st.markdown("""
    This system uses quantum machine learning to predict diseases from multiple data modalities:
    - **Clinical Notes** (Text): Quantum NLP processing
    - **Vital Signs** (Tabular): Quantum feature encoding  
    - **Chest X-rays** (Image): Hybrid CNN-Quantum processing
    """)
    
    # Load models and preprocessor
    models = load_trained_models()
    preprocessor = load_preprocessor()
    
    if models is None:
        st.error("Failed to load models. Please check the model directory.")
        return
    
    # Sidebar for input controls
    st.sidebar.header("📊 Patient Information")
    
    # Clinical text input
    st.sidebar.subheader("Clinical Notes")
    clinical_text = st.sidebar.text_area(
        "Enter clinical observations:",
        placeholder="Patient presents with chest pain and shortness of breath...",
        height=150
    )
    
    # Vital signs input
    st.sidebar.subheader("Vital Signs & Lab Values")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=65)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=80)
        sys_bp = st.number_input("Systolic BP", min_value=70, max_value=250, value=120)
        temp = st.number_input("Temperature (°F)", min_value=95.0, max_value=105.0, value=98.6)
        resp_rate = st.number_input("Respiratory Rate", min_value=8, max_value=40, value=16)
    
    with col2:
        o2_sat = st.number_input("O2 Saturation (%)", min_value=80, max_value=100, value=98)
        dias_bp = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80)
        wbc = st.number_input("WBC Count", min_value=1.0, max_value=50.0, value=7.5)
        hemoglobin = st.number_input("Hemoglobin", min_value=6.0, max_value=18.0, value=12.0)
        platelets = st.number_input("Platelets", min_value=50, max_value=600, value=250)
    
    # Additional lab values
    creatinine = st.sidebar.number_input("Creatinine", min_value=0.5, max_value=10.0, value=1.0)
    bun = st.sidebar.number_input("BUN", min_value=5, max_value=100, value=20)
    glucose = st.sidebar.number_input("Glucose", min_value=50, max_value=400, value=110)
    
    # Image upload
    st.sidebar.subheader("Chest X-ray")
    uploaded_image = st.sidebar.file_uploader(
        "Upload chest X-ray image:",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a chest X-ray image for analysis"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔬 Quantum Analysis")
        
        if st.button("🚀 Analyze Patient", type="primary", use_container_width=True):
            if not clinical_text.strip():
                st.warning("Please enter clinical notes for analysis.")
                return
            
            # Prepare inputs
            vital_signs = {
                'age': age,
                'heart_rate': heart_rate,
                'blood_pressure_systolic': sys_bp,
                'blood_pressure_diastolic': dias_bp,
                'temperature': temp,
                'respiratory_rate': resp_rate,
                'oxygen_saturation': o2_sat,
                'white_blood_cell_count': wbc,
                'hemoglobin': hemoglobin,
                'platelets': platelets,
                'creatinine': creatinine,
                'bun': bun,
                'glucose': glucose
            }
            
            # Process image if uploaded
            image = None
            if uploaded_image is not None:
                image = PIL.Image.open(uploaded_image)
            
            # Make prediction
            with st.spinner("🔄 Processing with quantum algorithms..."):
                prediction, confidence, risk_profile = predict_disease(
                    models, clinical_text, vital_signs, image, preprocessor
                )
            
            # Display results
            st.success("Analysis Complete!")
            
            # Prediction result
            st.markdown("### 🎯 Prediction Result")
            
            confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            st.markdown(f"""
            **Predicted Condition:** {prediction}  
            **Confidence:** {confidence_color} {confidence:.1%}
            """)
            
            # Confidence interpretation
            if confidence > 0.8:
                st.info("✅ High confidence prediction. Clinical correlation recommended.")
            elif confidence > 0.6:
                st.warning("📋 Moderate confidence. Additional evaluation may be needed.")
            else:
                st.error("⚠️ Low confidence prediction. Recommend comprehensive clinical assessment.")
            
            # Risk assessment
            st.markdown("### 📊 Risk Assessment")
            
            risk_df = pd.DataFrame(list(risk_profile.items()), columns=['Category', 'Risk Score'])
            risk_df['Risk Level'] = risk_df['Risk Score'].apply(
                lambda x: 'Low' if x < 0.3 else 'Moderate' if x < 0.7 else 'High'
            )
            
            st.dataframe(risk_df, use_container_width=True)
            
            # Clinical summary
            clinical_summary = DiseaseClassificationUtils.generate_clinical_summary(
                prediction, confidence, risk_profile
            )
            
            st.markdown("### 📋 Clinical Summary")
            st.markdown(clinical_summary)
    
    with col2:
        st.header("📈 Model Information")
        
        # Model status
        st.markdown("### 🔧 Quantum Models")
        st.markdown("""
        ✅ **Text Model:** Quantum NLP (8 qubits)  
        ✅ **Tabular Model:** Quantum features (8 qubits)  
        ✅ **Image Model:** Hybrid CNN-Quantum (8 qubits)  
        ✅ **Fusion Layer:** Quantum interference
        """)
        
        # Display uploaded image
        if uploaded_image is not None:
            st.markdown("### 🖼️ Uploaded X-ray")
            image = PIL.Image.open(uploaded_image)
            st.image(image, caption="Chest X-ray", use_column_width=True)
        
        # Sample cases
        st.markdown("### 💡 Sample Cases")
        
        sample_cases = {
            "Pneumonia": {
                "text": "Patient presents with productive cough, fever, and chest pain. Chest X-ray shows consolidation in right lower lobe.",
                "vitals": {"heart_rate": 95, "temperature": 101.2, "respiratory_rate": 22, "oxygen_saturation": 94}
            },
            "Heart Failure": {
                "text": "Elderly patient with shortness of breath, bilateral lower extremity edema, and fatigue. History of hypertension.",
                "vitals": {"heart_rate": 110, "blood_pressure_systolic": 160, "respiratory_rate": 24, "oxygen_saturation": 92}
            },
            "Normal": {
                "text": "Routine check-up. Patient feels well with no acute complaints. Vital signs stable.",
                "vitals": {"heart_rate": 75, "blood_pressure_systolic": 118, "temperature": 98.6, "oxygen_saturation": 98}
            }
        }
        
        selected_case = st.selectbox("Load sample case:", ["None"] + list(sample_cases.keys()))
        
        if selected_case != "None" and st.button("Load Case", use_container_width=True):
            case_data = sample_cases[selected_case]
            st.session_state.sample_text = case_data["text"]
            for key, value in case_data["vitals"].items():
                st.session_state[key] = value
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>Quantum Healthcare AI</strong> - Powered by Quantum Machine Learning</p>
        <p><em>⚕️ This is a research prototype. Always consult healthcare professionals for medical decisions.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()