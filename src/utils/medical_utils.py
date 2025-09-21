"""
Utility functions for healthcare and medical data processing.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class MedicalDataUtils:
    """
    Utility functions for medical data processing and analysis.
    """
    
    @staticmethod
    def normalize_vital_signs(data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize vital signs to clinical ranges.
        
        Args:
            data: DataFrame with vital signs
            
        Returns:
            Normalized vital signs DataFrame
        """
        normalized_data = data.copy()
        
        # Define normal ranges for vital signs
        normal_ranges = {
            'heart_rate': (60, 100),
            'blood_pressure_systolic': (90, 140),
            'blood_pressure_diastolic': (60, 90),
            'temperature': (97.0, 99.5),
            'respiratory_rate': (12, 20),
            'oxygen_saturation': (95, 100)
        }
        
        for column, (min_val, max_val) in normal_ranges.items():
            if column in normalized_data.columns:
                # Normalize to [0, 1] range where 0.5 represents normal
                normalized_data[column] = (normalized_data[column] - min_val) / (max_val - min_val)
                # Clip to reasonable bounds
                normalized_data[column] = np.clip(normalized_data[column], -1, 2)
        
        return normalized_data
    
    @staticmethod
    def calculate_severity_scores(data: pd.DataFrame) -> pd.Series:
        """
        Calculate simplified severity scores from vital signs.
        
        Args:
            data: DataFrame with vital signs
            
        Returns:
            Series with severity scores
        """
        severity_scores = pd.Series(index=data.index, dtype=float)
        
        for idx, row in data.iterrows():
            score = 0
            
            # Heart rate scoring
            hr = row.get('heart_rate', 80)
            if hr < 50 or hr > 120:
                score += 2
            elif hr < 60 or hr > 100:
                score += 1
            
            # Blood pressure scoring
            sys_bp = row.get('blood_pressure_systolic', 120)
            if sys_bp < 90 or sys_bp > 180:
                score += 2
            elif sys_bp < 100 or sys_bp > 140:
                score += 1
            
            # Temperature scoring
            temp = row.get('temperature', 98.6)
            if temp < 96 or temp > 102:
                score += 2
            elif temp < 97 or temp > 100:
                score += 1
            
            # Oxygen saturation scoring
            spo2 = row.get('oxygen_saturation', 98)
            if spo2 < 90:
                score += 3
            elif spo2 < 95:
                score += 1
            
            severity_scores[idx] = score
        
        return severity_scores
    
    @staticmethod
    def extract_clinical_keywords(text: str) -> List[str]:
        """
        Extract clinical keywords from text.
        
        Args:
            text: Clinical text
            
        Returns:
            List of clinical keywords
        """
        # Common clinical keywords
        clinical_keywords = [
            'chest pain', 'shortness of breath', 'fever', 'cough', 'nausea',
            'vomiting', 'diarrhea', 'fatigue', 'headache', 'dizziness',
            'palpitations', 'syncope', 'edema', 'rash', 'seizure',
            'pneumonia', 'sepsis', 'heart failure', 'stroke', 'diabetes',
            'hypertension', 'copd', 'asthma', 'cancer', 'infection'
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in clinical_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    @staticmethod
    def create_disease_risk_profile(tabular_data: pd.DataFrame, 
                                  text_keywords: List[str]) -> Dict[str, float]:
        """
        Create a disease risk profile based on data.
        
        Args:
            tabular_data: Patient vital signs and lab values
            text_keywords: Clinical keywords from notes
            
        Returns:
            Dictionary with disease risk scores
        """
        risk_profile = {
            'cardiovascular': 0.0,
            'respiratory': 0.0,
            'infectious': 0.0,
            'neurological': 0.0,
            'metabolic': 0.0
        }
        
        # Risk factors from tabular data
        if 'age' in tabular_data.columns:
            age = tabular_data['age'].iloc[0] if len(tabular_data) > 0 else 65
            if age > 65:
                risk_profile['cardiovascular'] += 0.2
                risk_profile['respiratory'] += 0.1
        
        if 'blood_pressure_systolic' in tabular_data.columns:
            sys_bp = tabular_data['blood_pressure_systolic'].iloc[0] if len(tabular_data) > 0 else 120
            if sys_bp > 140:
                risk_profile['cardiovascular'] += 0.3
        
        if 'glucose' in tabular_data.columns:
            glucose = tabular_data['glucose'].iloc[0] if len(tabular_data) > 0 else 100
            if glucose > 126:
                risk_profile['metabolic'] += 0.4
        
        # Risk factors from text keywords
        keyword_risks = {
            'chest pain': {'cardiovascular': 0.4},
            'shortness of breath': {'cardiovascular': 0.3, 'respiratory': 0.4},
            'fever': {'infectious': 0.3},
            'cough': {'respiratory': 0.3, 'infectious': 0.2},
            'headache': {'neurological': 0.2},
            'seizure': {'neurological': 0.5},
            'pneumonia': {'respiratory': 0.6, 'infectious': 0.4},
            'sepsis': {'infectious': 0.8},
            'heart failure': {'cardiovascular': 0.7},
            'stroke': {'neurological': 0.8, 'cardiovascular': 0.3},
            'diabetes': {'metabolic': 0.6}
        }
        
        for keyword in text_keywords:
            if keyword in keyword_risks:
                for disease_category, risk_increase in keyword_risks[keyword].items():
                    risk_profile[disease_category] += risk_increase
        
        # Normalize risk scores to [0, 1]
        for category in risk_profile:
            risk_profile[category] = min(risk_profile[category], 1.0)
        
        return risk_profile

class DiseaseClassificationUtils:
    """
    Utilities for disease classification and prediction.
    """
    
    @staticmethod
    def get_disease_categories() -> Dict[str, List[str]]:
        """
        Get disease categories and their associated conditions.
        
        Returns:
            Dictionary mapping categories to disease lists
        """
        return {
            'cardiovascular': [
                'Heart Failure', 'Myocardial Infarction', 'Hypertension',
                'Arrhythmia', 'Coronary Artery Disease'
            ],
            'respiratory': [
                'Pneumonia', 'COPD', 'Asthma', 'Pulmonary Embolism',
                'Acute Respiratory Distress Syndrome'
            ],
            'infectious': [
                'Sepsis', 'Pneumonia', 'Urinary Tract Infection',
                'Cellulitis', 'Bacteremia'
            ],
            'neurological': [
                'Stroke', 'Seizure Disorder', 'Dementia',
                'Parkinson Disease', 'Multiple Sclerosis'
            ],
            'metabolic': [
                'Diabetes', 'Diabetic Ketoacidosis', 'Hypoglycemia',
                'Thyroid Disorder', 'Electrolyte Imbalance'
            ]
        }
    
    @staticmethod
    def map_disease_to_category(disease: str) -> str:
        """
        Map a specific disease to its category.
        
        Args:
            disease: Disease name
            
        Returns:
            Disease category
        """
        categories = DiseaseClassificationUtils.get_disease_categories()
        
        for category, diseases in categories.items():
            if disease in diseases:
                return category
        
        return 'other'
    
    @staticmethod
    def calculate_prediction_confidence(probabilities: np.ndarray, 
                                      threshold: float = 0.8) -> Tuple[float, bool]:
        """
        Calculate prediction confidence and reliability.
        
        Args:
            probabilities: Model output probabilities
            threshold: Confidence threshold
            
        Returns:
            Tuple of (confidence_score, is_reliable)
        """
        max_prob = np.max(probabilities)
        second_max_prob = np.partition(probabilities, -2)[-2]
        
        # Confidence based on gap between top predictions
        confidence_gap = max_prob - second_max_prob
        confidence_score = max_prob * (1 + confidence_gap)
        
        is_reliable = confidence_score >= threshold
        
        return confidence_score, is_reliable
    
    @staticmethod
    def generate_clinical_summary(prediction: str, confidence: float,
                                risk_profile: Dict[str, float]) -> str:
        """
        Generate a clinical summary of the prediction.
        
        Args:
            prediction: Predicted disease
            confidence: Prediction confidence
            risk_profile: Disease risk profile
            
        Returns:
            Clinical summary text
        """
        summary = f"**Predicted Condition**: {prediction}\n"
        summary += f"**Confidence**: {confidence:.2%}\n\n"
        
        summary += "**Risk Assessment**:\n"
        for category, risk in risk_profile.items():
            risk_level = "Low" if risk < 0.3 else "Moderate" if risk < 0.7 else "High"
            summary += f"- {category.title()}: {risk_level} ({risk:.2%})\n"
        
        if confidence < 0.6:
            summary += "\n⚠️ **Note**: Low confidence prediction. Consider additional clinical evaluation."
        elif confidence < 0.8:
            summary += "\n📋 **Note**: Moderate confidence. Clinical correlation recommended."
        else:
            summary += "\n✅ **Note**: High confidence prediction."
        
        return summary