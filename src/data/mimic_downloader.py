"""
MIMIC-IV data downloader and access module.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MIMICDownloader:
    """
    Downloads and manages MIMIC-IV dataset components.
    """
    
    def __init__(self, data_dir: str = "data/mimic"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_sample_data(self) -> Dict[str, Any]:
        """
        Downloads sample MIMIC-IV data for demonstration.
        In practice, this would connect to PhysioNet.
        """
        logger.info("Generating sample MIMIC-IV data for demonstration...")
        
        # Generate synthetic tabular data (ICU stays)
        n_patients = 1000
        tabular_data = self._generate_sample_tabular(n_patients)
        
        # Generate synthetic text data (clinical notes)
        text_data = self._generate_sample_text(n_patients)
        
        # Generate synthetic image data (chest X-rays)
        image_data = self._generate_sample_images(n_patients)
        
        # Generate labels (disease outcomes)
        labels = self._generate_sample_labels(n_patients)
        
        # Save to files
        self._save_data(tabular_data, text_data, image_data, labels)
        
        return {
            'tabular': tabular_data,
            'text': text_data, 
            'image': image_data,
            'labels': labels
        }
    
    def _generate_sample_tabular(self, n_patients: int) -> pd.DataFrame:
        """Generate synthetic tabular ICU data."""
        np.random.seed(42)
        
        data = {
            'patient_id': range(n_patients),
            'age': np.random.normal(65, 15, n_patients).clip(18, 100),
            'gender': np.random.choice(['M', 'F'], n_patients),
            'heart_rate': np.random.normal(80, 15, n_patients).clip(40, 200),
            'blood_pressure_systolic': np.random.normal(120, 20, n_patients).clip(70, 250),
            'blood_pressure_diastolic': np.random.normal(80, 15, n_patients).clip(40, 150),
            'temperature': np.random.normal(98.6, 2, n_patients).clip(95, 105),
            'respiratory_rate': np.random.normal(16, 4, n_patients).clip(8, 40),
            'oxygen_saturation': np.random.normal(98, 3, n_patients).clip(85, 100),
            'white_blood_cell_count': np.random.lognormal(2, 0.5, n_patients),
            'hemoglobin': np.random.normal(12, 2, n_patients).clip(6, 18),
            'platelets': np.random.normal(250, 80, n_patients).clip(50, 600),
            'creatinine': np.random.lognormal(0, 0.3, n_patients),
            'bun': np.random.normal(20, 10, n_patients).clip(5, 100),
            'glucose': np.random.normal(110, 30, n_patients).clip(50, 400),
            'los_hours': np.random.exponential(72, n_patients).clip(1, 720)  # Length of stay
        }
        
        return pd.DataFrame(data)
    
    def _generate_sample_text(self, n_patients: int) -> pd.DataFrame:
        """Generate synthetic clinical notes."""
        np.random.seed(42)
        
        # Sample clinical note templates
        templates = [
            "Patient presents with chest pain and shortness of breath. Vital signs stable. EKG shows normal sinus rhythm.",
            "Elderly patient admitted with pneumonia. Productive cough with fever. Chest X-ray shows consolidation.",
            "Patient with diabetes mellitus presenting with diabetic ketoacidosis. Blood glucose elevated.",
            "Post-operative patient recovering well. No signs of infection. Pain controlled with medication.",
            "Patient with heart failure presenting with fluid overload. Diuretics initiated.",
            "Trauma patient with multiple injuries. CT scan shows no internal bleeding.",
            "Patient with COPD exacerbation. Bronchodilators and steroids administered.",
            "Septic patient with altered mental status. Blood cultures pending.",
            "Patient with acute kidney injury. Creatinine elevated. Nephrology consulted.",
            "Stroke patient with left-sided weakness. MRI shows acute infarct."
        ]
        
        notes = []
        for i in range(n_patients):
            # Combine 1-3 templates with variations
            num_notes = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            selected_templates = np.random.choice(templates, num_notes, replace=False)
            combined_note = " ".join(selected_templates)
            notes.append(combined_note)
        
        data = {
            'patient_id': range(n_patients),
            'clinical_notes': notes
        }
        
        return pd.DataFrame(data)
    
    def _generate_sample_images(self, n_patients: int) -> np.ndarray:
        """Generate synthetic chest X-ray data."""
        np.random.seed(42)
        
        # Simulate 224x224 grayscale chest X-rays
        images = np.random.rand(n_patients, 224, 224) * 0.5 + 0.3  # Base lung tissue
        
        # Add some realistic patterns
        for i in range(n_patients):
            # Add ribcage structure
            for rib in range(8):
                y_pos = 50 + rib * 20
                images[i, y_pos:y_pos+2, :] *= 0.7
            
            # Add heart shadow
            center_x, center_y = 112, 140
            for x in range(224):
                for y in range(224):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < 40:
                        images[i, y, x] *= 0.8
        
        return images
    
    def _generate_sample_labels(self, n_patients: int) -> pd.DataFrame:
        """Generate disease labels."""
        np.random.seed(42)
        
        diseases = [
            'Pneumonia', 'Heart Failure', 'Sepsis', 'COPD', 'Diabetes', 
            'Acute Kidney Injury', 'Stroke', 'Myocardial Infarction', 'Normal'
        ]
        
        # Create realistic disease distribution
        disease_probs = [0.15, 0.12, 0.10, 0.08, 0.15, 0.08, 0.07, 0.05, 0.20]
        
        labels = np.random.choice(diseases, n_patients, p=disease_probs)
        
        data = {
            'patient_id': range(n_patients),
            'disease_label': labels
        }
        
        return pd.DataFrame(data)
    
    def _save_data(self, tabular_data: pd.DataFrame, text_data: pd.DataFrame,
                   image_data: np.ndarray, labels: pd.DataFrame):
        """Save generated data to files."""
        tabular_data.to_csv(self.data_dir / 'tabular_data.csv', index=False)
        text_data.to_csv(self.data_dir / 'text_data.csv', index=False)
        np.save(self.data_dir / 'image_data.npy', image_data)
        labels.to_csv(self.data_dir / 'labels.csv', index=False)
        
        logger.info(f"Sample data saved to {self.data_dir}")
    
    def load_existing_data(self) -> Dict[str, Any]:
        """Load previously downloaded data."""
        try:
            tabular_data = pd.read_csv(self.data_dir / 'tabular_data.csv')
            text_data = pd.read_csv(self.data_dir / 'text_data.csv')
            image_data = np.load(self.data_dir / 'image_data.npy')
            labels = pd.read_csv(self.data_dir / 'labels.csv')
            
            return {
                'tabular': tabular_data,
                'text': text_data,
                'image': image_data,
                'labels': labels
            }
        except FileNotFoundError:
            logger.warning("No existing data found. Run download_sample_data() first.")
            return None