"""
Gradio interface for quantum healthcare disease prediction.
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
import PIL.Image
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

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

class QuantumHealthcareInterface:
    """Gradio interface for quantum healthcare predictions."""
    
    def __init__(self, model_dir: str = "outputs/models"):
        self.models = self.load_models(model_dir)
        self.preprocessor = MIMICPreprocessor()
        self.disease_classes = [
            'Pneumonia', 'Heart Failure', 'Sepsis', 'COPD', 'Diabetes', 
            'Acute Kidney Injury', 'Stroke', 'Myocardial Infarction', 'Normal'
        ]
    
    def load_models(self, model_dir: str) -> Dict[str, Any]:
        """Load trained quantum models."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model configuration
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
                print("✅ Trained models loaded successfully!")
            except FileNotFoundError:
                print("⚠️ Pre-trained models not found. Using randomly initialized models for demonstration.")
        
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
    
    def process_inputs(self, clinical_text: str, age: int, heart_rate: int, 
                      sys_bp: int, dias_bp: int, temperature: float, resp_rate: int,
                      o2_sat: int, wbc: float, hemoglobin: float, platelets: int,
                      creatinine: float, bun: int, glucose: int, 
                      chest_xray: Optional[PIL.Image.Image]) -> Tuple[str, float, str, str]:
        """Process all inputs and make prediction."""
        
        try:
            device = self.models['device']
            
            # Process clinical text
            text_df = pd.DataFrame({'clinical_notes': [clinical_text]})
            text_data = self.preprocessor.preprocess_text_data(text_df)
            
            # Process tabular data
            vital_signs = {
                'age': age, 'heart_rate': heart_rate,
                'blood_pressure_systolic': sys_bp, 'blood_pressure_diastolic': dias_bp,
                'temperature': temperature, 'respiratory_rate': resp_rate,
                'oxygen_saturation': o2_sat, 'white_blood_cell_count': wbc,
                'hemoglobin': hemoglobin, 'platelets': platelets,
                'creatinine': creatinine, 'bun': bun, 'glucose': glucose,
                'los_hours': 24, 'gender_encoded': 1  # defaults
            }
            
            tabular_df = pd.DataFrame([vital_signs])
            # Simple normalization for demo
            numeric_cols = list(vital_signs.keys())[:-1]  # exclude gender_encoded
            tabular_df[numeric_cols] = (tabular_df[numeric_cols] - tabular_df[numeric_cols].mean()) / (tabular_df[numeric_cols].std() + 1e-8)
            tabular_data = torch.FloatTensor(tabular_df.values)
            
            # Process image
            if chest_xray is not None:
                if chest_xray.mode != 'L':
                    chest_xray = chest_xray.convert('L')
                chest_xray = chest_xray.resize((224, 224))
                image_array = np.array(chest_xray) / 255.0
                image_data = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
            else:
                image_data = torch.zeros(1, 1, 224, 224)
            
            with torch.no_grad():
                # Extract quantum features
                text_features = self.models['text'].extract_quantum_features(
                    text_data['input_ids'].to(device),
                    text_data['attention_mask'].to(device)
                )
                
                tabular_features = self.models['tabular'].extract_quantum_features(
                    tabular_data.to(device)
                )
                
                image_features = self.models['image'].extract_quantum_features(
                    image_data.to(device)
                )
                
                # Fusion prediction
                fusion_output = self.models['fusion'](text_features, tabular_features, image_features)
                probabilities = torch.softmax(fusion_output, dim=1)
                
                # Get prediction
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                predicted_disease = self.disease_classes[predicted_class]
                
                # Create risk profile
                risk_profile = MedicalDataUtils.create_disease_risk_profile(
                    pd.DataFrame([vital_signs]), 
                    MedicalDataUtils.extract_clinical_keywords(clinical_text)
                )
                
                # Format results
                confidence_text = f"{confidence:.1%}"
                
                # Risk assessment
                risk_text = "**Risk Assessment:**\n"
                for category, risk in risk_profile.items():
                    risk_level = "Low" if risk < 0.3 else "Moderate" if risk < 0.7 else "High"
                    risk_text += f"- {category.title()}: {risk_level} ({risk:.1%})\n"
                
                # Clinical summary
                clinical_summary = DiseaseClassificationUtils.generate_clinical_summary(
                    predicted_disease, confidence, risk_profile
                )
                
                return predicted_disease, confidence_text, risk_text, clinical_summary
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return "Error in prediction", "0%", f"Error: {str(e)}", "Please check inputs and try again."
    
    def create_interface(self) -> gr.Interface:
        """Create Gradio interface."""
        
        # Define sample cases
        sample_cases = {
            "Pneumonia": [
                "Patient presents with productive cough, fever, and chest pain. Chest X-ray shows consolidation in right lower lobe.",
                67, 95, 135, 85, 101.2, 22, 94, 12.5, 11.0, 180, 1.1, 22, 140
            ],
            "Heart Failure": [
                "Elderly patient with shortness of breath, bilateral lower extremity edema, and fatigue. History of hypertension.",
                74, 110, 160, 95, 98.8, 24, 92, 8.2, 10.5, 220, 1.3, 35, 125
            ],
            "Normal": [
                "Routine check-up. Patient feels well with no acute complaints. Vital signs stable.",
                45, 75, 118, 78, 98.6, 16, 98, 7.0, 13.5, 280, 1.0, 18, 95
            ]
        }
        
        def load_sample_case(case_name):
            """Load a sample case."""
            if case_name in sample_cases:
                return sample_cases[case_name]
            return [""] + [0] * 13  # Empty case
        
        with gr.Blocks(title="Quantum Healthcare AI", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🏥 Quantum Healthcare AI
            ## Multimodal Disease Prediction System
            
            This system uses quantum machine learning to predict diseases from clinical notes, vital signs, and chest X-rays.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📝 Clinical Information")
                    
                    clinical_text = gr.Textbox(
                        label="Clinical Notes",
                        placeholder="Enter clinical observations, symptoms, and findings...",
                        lines=5
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            age = gr.Slider(18, 100, value=65, label="Age")
                            heart_rate = gr.Slider(30, 200, value=80, label="Heart Rate (bpm)")
                            sys_bp = gr.Slider(70, 250, value=120, label="Systolic BP")
                            temperature = gr.Slider(95.0, 105.0, value=98.6, step=0.1, label="Temperature (°F)")
                            resp_rate = gr.Slider(8, 40, value=16, label="Respiratory Rate")
                            wbc = gr.Slider(1.0, 50.0, value=7.5, step=0.1, label="WBC Count")
                            creatinine = gr.Slider(0.5, 10.0, value=1.0, step=0.1, label="Creatinine")
                        
                        with gr.Column():
                            o2_sat = gr.Slider(80, 100, value=98, label="O2 Saturation (%)")
                            dias_bp = gr.Slider(40, 150, value=80, label="Diastolic BP")
                            hemoglobin = gr.Slider(6.0, 18.0, value=12.0, step=0.1, label="Hemoglobin")
                            platelets = gr.Slider(50, 600, value=250, label="Platelets")
                            bun = gr.Slider(5, 100, value=20, label="BUN")
                            glucose = gr.Slider(50, 400, value=110, label="Glucose")
                    
                    chest_xray = gr.Image(
                        label="Chest X-ray (Optional)",
                        type="pil",
                        height=200
                    )
                    
                    # Sample cases
                    gr.Markdown("### 💡 Sample Cases")
                    case_dropdown = gr.Dropdown(
                        choices=[""] + list(sample_cases.keys()),
                        label="Load Sample Case",
                        value=""
                    )
                    
                    load_case_btn = gr.Button("Load Selected Case", variant="secondary")
                    
                    # Main prediction button
                    predict_btn = gr.Button("🚀 Analyze Patient", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Prediction Results")
                    
                    predicted_disease = gr.Textbox(
                        label="Predicted Condition",
                        interactive=False
                    )
                    
                    confidence = gr.Textbox(
                        label="Confidence",
                        interactive=False
                    )
                    
                    risk_assessment = gr.Markdown(
                        label="Risk Assessment"
                    )
                    
                    clinical_summary = gr.Markdown(
                        label="Clinical Summary"
                    )
                    
                    gr.Markdown("""
                    ### 🔧 Quantum Models
                    - **Text:** Quantum NLP (8 qubits)
                    - **Tabular:** Quantum features (8 qubits)  
                    - **Image:** Hybrid CNN-Quantum (8 qubits)
                    - **Fusion:** Quantum interference
                    """)
            
            # Event handlers
            predict_btn.click(
                fn=self.process_inputs,
                inputs=[
                    clinical_text, age, heart_rate, sys_bp, dias_bp, 
                    temperature, resp_rate, o2_sat, wbc, hemoglobin, 
                    platelets, creatinine, bun, glucose, chest_xray
                ],
                outputs=[predicted_disease, confidence, risk_assessment, clinical_summary]
            )
            
            load_case_btn.click(
                fn=load_sample_case,
                inputs=[case_dropdown],
                outputs=[
                    clinical_text, age, heart_rate, sys_bp, dias_bp,
                    temperature, resp_rate, o2_sat, wbc, hemoglobin,
                    platelets, creatinine, bun, glucose
                ]
            )
            
            gr.Markdown("""
            ---
            **⚕️ Disclaimer:** This is a research prototype for demonstration purposes. 
            Always consult healthcare professionals for medical decisions.
            """)
        
        return interface

def create_gradio_interface(model_dir: str = "outputs/models") -> gr.Interface:
    """Create and return Gradio interface."""
    healthcare_interface = QuantumHealthcareInterface(model_dir)
    return healthcare_interface.create_interface()

if __name__ == "__main__":
    # Launch Gradio interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )