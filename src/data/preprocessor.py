"""
Data preprocessing module for MIMIC-IV multimodal data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MIMICPreprocessor:
    """
    Preprocesses MIMIC-IV data for quantum machine learning models.
    """
    
    def __init__(self):
        self.tabular_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    def preprocess_tabular_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess tabular ICU data.
        
        Args:
            data: Raw tabular data DataFrame
            
        Returns:
            Preprocessed and normalized tabular features
        """
        logger.info("Preprocessing tabular data...")
        
        # Select numerical features
        numerical_cols = ['age', 'heart_rate', 'blood_pressure_systolic', 
                         'blood_pressure_diastolic', 'temperature', 'respiratory_rate',
                         'oxygen_saturation', 'white_blood_cell_count', 'hemoglobin',
                         'platelets', 'creatinine', 'bun', 'glucose', 'los_hours']
        
        # Encode categorical features
        categorical_cols = ['gender']
        processed_data = data[numerical_cols].copy()
        
        # Handle gender encoding
        processed_data['gender_encoded'] = (data['gender'] == 'M').astype(int)
        
        # Handle missing values
        processed_data = processed_data.fillna(processed_data.mean())
        
        # Normalize features
        processed_data = self.tabular_scaler.fit_transform(processed_data)
        
        logger.info(f"Tabular data preprocessed: {processed_data.shape}")
        return processed_data
    
    def preprocess_text_data(self, data: pd.DataFrame, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Preprocess clinical notes text data.
        
        Args:
            data: DataFrame with clinical notes
            max_length: Maximum sequence length for tokenization
            
        Returns:
            Dictionary with tokenized text tensors
        """
        logger.info("Preprocessing text data...")
        
        # Extract clinical notes
        texts = data['clinical_notes'].tolist()
        
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        logger.info(f"Text data preprocessed: {len(texts)} samples, max_length={max_length}")
        return encoded
    
    def preprocess_image_data(self, images: np.ndarray) -> torch.Tensor:
        """
        Preprocess chest X-ray image data.
        
        Args:
            images: Raw image array (N, H, W)
            
        Returns:
            Preprocessed image tensor
        """
        logger.info("Preprocessing image data...")
        
        # Normalize pixel values to [0, 1]
        processed_images = images.astype(np.float32)
        processed_images = (processed_images - processed_images.min()) / (processed_images.max() - processed_images.min())
        
        # Add channel dimension for grayscale (N, 1, H, W)
        processed_images = np.expand_dims(processed_images, axis=1)
        
        # Convert to PyTorch tensor
        processed_images = torch.from_numpy(processed_images)
        
        logger.info(f"Image data preprocessed: {processed_images.shape}")
        return processed_images
    
    def preprocess_labels(self, labels: pd.DataFrame) -> torch.Tensor:
        """
        Preprocess disease labels.
        
        Args:
            labels: DataFrame with disease labels
            
        Returns:
            Encoded label tensor
        """
        logger.info("Preprocessing labels...")
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels['disease_label'])
        label_tensor = torch.from_numpy(encoded_labels).long()
        
        logger.info(f"Labels preprocessed: {len(encoded_labels)} samples, {len(self.label_encoder.classes_)} classes")
        logger.info(f"Classes: {list(self.label_encoder.classes_)}")
        
        return label_tensor
    
    def create_train_test_split(self, tabular_data: np.ndarray, text_data: Dict[str, torch.Tensor],
                               image_data: torch.Tensor, labels: torch.Tensor, 
                               test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Create train/test splits for all modalities.
        
        Args:
            tabular_data: Preprocessed tabular features
            text_data: Preprocessed text features
            image_data: Preprocessed image features
            labels: Preprocessed labels
            test_size: Fraction of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        logger.info("Creating train/test splits...")
        
        n_samples = len(labels)
        indices = np.arange(n_samples)
        
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, 
            stratify=labels.numpy()
        )
        
        # Split tabular data
        train_tabular = tabular_data[train_indices]
        test_tabular = tabular_data[test_indices]
        
        # Split text data
        train_text = {
            'input_ids': text_data['input_ids'][train_indices],
            'attention_mask': text_data['attention_mask'][train_indices]
        }
        test_text = {
            'input_ids': text_data['input_ids'][test_indices],
            'attention_mask': text_data['attention_mask'][test_indices]
        }
        
        # Split image data
        train_images = image_data[train_indices]
        test_images = image_data[test_indices]
        
        # Split labels
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        
        train_data = {
            'tabular': train_tabular,
            'text': train_text,
            'image': train_images,
            'labels': train_labels
        }
        
        test_data = {
            'tabular': test_tabular,
            'text': test_text,
            'image': test_images,
            'labels': test_labels
        }
        
        logger.info(f"Train split: {len(train_indices)} samples")
        logger.info(f"Test split: {len(test_indices)} samples")
        
        return train_data, test_data
    
    def get_label_classes(self) -> list:
        """Get the list of disease classes."""
        return list(self.label_encoder.classes_)
    
    def inverse_transform_labels(self, encoded_labels: np.ndarray) -> list:
        """Convert encoded labels back to disease names."""
        return self.label_encoder.inverse_transform(encoded_labels)