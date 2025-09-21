"""
Comprehensive training module for quantum healthcare models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import json
from pathlib import Path
import wandb
from tqdm import tqdm

from ..models.quantum_nlp import build_quantum_nlp_model
from ..models.quantum_tabular import build_quantum_tabular_model
from ..models.hybrid_cnn_quantum import build_hybrid_cnn_quantum_model
from ..models.fusion_layer import create_fusion_layer
from ..training.evaluator import ModelEvaluator
from ..data.data_loader import load_and_preprocess_mimic_data

logger = logging.getLogger(__name__)

class MultimodalDataset(torch.utils.data.Dataset):
    """
    Dataset class for multimodal healthcare data.
    """
    
    def __init__(self, tabular_data: np.ndarray, text_data: Dict[str, torch.Tensor],
                 image_data: torch.Tensor, labels: torch.Tensor):
        self.tabular_data = torch.FloatTensor(tabular_data)
        self.text_data = text_data
        self.image_data = image_data
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'tabular': self.tabular_data[idx],
            'text': {
                'input_ids': self.text_data['input_ids'][idx],
                'attention_mask': self.text_data['attention_mask'][idx]
            },
            'image': self.image_data[idx],
            'labels': self.labels[idx]
        }

class QuantumHealthcareTrainer:
    """
    Comprehensive trainer for quantum healthcare models.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 device: str = None,
                 use_wandb: bool = False):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
            device: Device for training
            use_wandb: Whether to use Weights & Biases logging
        """
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb
        
        # Initialize logging
        if use_wandb:
            wandb.init(project="quantum-healthcare", config=config)
        
        # Create output directories
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_dir = self.output_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)
        
        self.results_dir = self.output_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        
        logger.info(f"QuantumHealthcareTrainer initialized on {self.device}")
    
    def load_data(self, data_dir: str = "data/mimic") -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare data loaders.
        
        Args:
            data_dir: Directory containing data
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        logger.info("Loading and preprocessing data...")
        
        train_data, test_data, class_names, preprocessor = load_and_preprocess_mimic_data(data_dir)
        
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.preprocessor = preprocessor
        
        # Create datasets
        train_dataset = MultimodalDataset(
            train_data['tabular'], train_data['text'], 
            train_data['image'], train_data['labels']
        )
        
        test_dataset = MultimodalDataset(
            test_data['tabular'], test_data['text'],
            test_data['image'], test_data['labels']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 16),
            shuffle=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False
        )
        
        logger.info(f"Data loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
        return train_loader, test_loader
    
    def build_models(self) -> Dict[str, nn.Module]:
        """
        Build all quantum models.
        
        Returns:
            Dictionary of models
        """
        logger.info("Building quantum models...")
        
        model_config = self.config.get('models', {})
        
        # Build individual modality models
        text_model = build_quantum_nlp_model(
            n_qubits=model_config.get('text_qubits', 8),
            n_layers=model_config.get('text_layers', 3),
            output_dim=self.n_classes
        )
        
        tabular_model = build_quantum_tabular_model(
            input_dim=model_config.get('tabular_input_dim', 16),
            n_qubits=model_config.get('tabular_qubits', 8),
            n_layers=model_config.get('tabular_layers', 4),
            output_dim=self.n_classes
        )
        
        image_model = build_hybrid_cnn_quantum_model(
            n_qubits=model_config.get('image_qubits', 8),
            n_layers=model_config.get('image_layers', 3),
            output_dim=self.n_classes,
            use_quantum_conv=model_config.get('use_quantum_conv', True)
        )
        
        # Build fusion model
        fusion_model = create_fusion_layer(
            text_dim=model_config.get('text_qubits', 8),
            tabular_dim=model_config.get('tabular_qubits', 8),
            image_dim=model_config.get('image_qubits', 8),
            fusion_type=model_config.get('fusion_type', 'hybrid'),
            output_dim=self.n_classes
        )
        
        models = {
            'text': text_model.to(self.device),
            'tabular': tabular_model.to(self.device),
            'image': image_model.to(self.device),
            'fusion': fusion_model.to(self.device)
        }
        
        self.models = models
        logger.info("All models built successfully")
        return models
    
    def setup_optimizers(self):
        """Setup optimizers and schedulers for all models."""
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        for name, model in self.models.items():
            # Use different learning rates for different components
            if name == 'text':
                # Lower learning rate for BERT-based model
                optimizer = optim.AdamW(model.parameters(), lr=lr*0.1, weight_decay=weight_decay)
            else:
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            self.optimizers[name] = optimizer
            self.schedulers[name] = scheduler
        
        logger.info("Optimizers and schedulers configured")
    
    def train_individual_models(self, train_loader: DataLoader, 
                               test_loader: DataLoader, epochs: int = 50):
        """
        Train individual modality models separately.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            epochs: Number of training epochs
        """
        logger.info("Training individual modality models...")
        
        criterion = nn.CrossEntropyLoss()
        
        for modality in ['text', 'tabular', 'image']:
            logger.info(f"Training {modality} model...")
            
            model = self.models[modality]
            optimizer = self.optimizers[modality]
            scheduler = self.schedulers[modality]
            
            best_loss = float('inf')
            patience_counter = 0
            patience = self.config.get('patience', 10)
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                    optimizer.zero_grad()
                    
                    # Get modality-specific input
                    if modality == 'text':
                        outputs = model(batch['text']['input_ids'].to(self.device),
                                      batch['text']['attention_mask'].to(self.device))
                    elif modality == 'tabular':
                        outputs = model(batch['tabular'].to(self.device))
                    elif modality == 'image':
                        outputs = model(batch['image'].to(self.device))
                    
                    labels = batch['labels'].to(self.device)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in test_loader:
                        if modality == 'text':
                            outputs = model(batch['text']['input_ids'].to(self.device),
                                          batch['text']['attention_mask'].to(self.device))
                        elif modality == 'tabular':
                            outputs = model(batch['tabular'].to(self.device))
                        elif modality == 'image':
                            outputs = model(batch['image'].to(self.device))
                        
                        labels = batch['labels'].to(self.device)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                # Calculate metrics
                train_loss /= len(train_loader)
                val_loss /= len(test_loader)
                train_acc = 100 * train_correct / train_total
                val_acc = 100 * val_correct / val_total
                
                logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), self.model_dir / f'best_{modality}_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping for {modality} model at epoch {epoch+1}")
                        break
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        f'{modality}_train_loss': train_loss,
                        f'{modality}_train_acc': train_acc,
                        f'{modality}_val_loss': val_loss,
                        f'{modality}_val_acc': val_acc,
                        'epoch': epoch
                    })
        
        logger.info("Individual model training completed")
    
    def train_fusion_model(self, train_loader: DataLoader, 
                          test_loader: DataLoader, epochs: int = 30):
        """
        Train the multimodal fusion model.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            epochs: Number of training epochs
        """
        logger.info("Training fusion model...")
        
        # Load best individual models
        for modality in ['text', 'tabular', 'image']:
            model_path = self.model_dir / f'best_{modality}_model.pth'
            if model_path.exists():
                self.models[modality].load_state_dict(torch.load(model_path))
                # Freeze individual models
                for param in self.models[modality].parameters():
                    param.requires_grad = False
                self.models[modality].eval()
        
        criterion = nn.CrossEntropyLoss()
        fusion_model = self.models['fusion']
        optimizer = self.optimizers['fusion']
        scheduler = self.schedulers['fusion']
        
        best_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 10)
        
        for epoch in range(epochs):
            # Training phase
            fusion_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc=f"Fusion Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                # Extract features from individual models
                with torch.no_grad():
                    text_features = self.models['text'].extract_quantum_features(
                        batch['text']['input_ids'].to(self.device),
                        batch['text']['attention_mask'].to(self.device)
                    )
                    tabular_features = self.models['tabular'].extract_quantum_features(
                        batch['tabular'].to(self.device)
                    )
                    image_features = self.models['image'].extract_quantum_features(
                        batch['image'].to(self.device)
                    )
                
                # Fusion prediction
                outputs = fusion_model(text_features, tabular_features, image_features)
                labels = batch['labels'].to(self.device)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            fusion_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    text_features = self.models['text'].extract_quantum_features(
                        batch['text']['input_ids'].to(self.device),
                        batch['text']['attention_mask'].to(self.device)
                    )
                    tabular_features = self.models['tabular'].extract_quantum_features(
                        batch['tabular'].to(self.device)
                    )
                    image_features = self.models['image'].extract_quantum_features(
                        batch['image'].to(self.device)
                    )
                    
                    outputs = fusion_model(text_features, tabular_features, image_features)
                    labels = batch['labels'].to(self.device)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            logger.info(f"Fusion Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best fusion model
                torch.save(fusion_model.state_dict(), self.model_dir / 'best_fusion_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping for fusion model at epoch {epoch+1}")
                    break
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'fusion_train_loss': train_loss,
                    'fusion_train_acc': train_acc,
                    'fusion_val_loss': val_loss,
                    'fusion_val_acc': val_acc,
                    'epoch': epoch
                })
        
        logger.info("Fusion model training completed")
    
    def evaluate_models(self, test_loader: DataLoader):
        """
        Comprehensive evaluation of all models.
        
        Args:
            test_loader: Test data loader
        """
        logger.info("Evaluating models...")
        
        evaluator = ModelEvaluator(self.class_names)
        
        # Load best models
        for modality in ['text', 'tabular', 'image', 'fusion']:
            model_path = self.model_dir / f'best_{modality}_model.pth'
            if model_path.exists():
                self.models[modality].load_state_dict(torch.load(model_path))
        
        # Evaluate individual models
        for modality in ['text', 'tabular', 'image']:
            logger.info(f"Evaluating {modality} model...")
            
            model = self.models[modality]
            model.eval()
            
            all_predictions = []
            all_labels = []
            all_probabilities = []
            
            with torch.no_grad():
                for batch in test_loader:
                    if modality == 'text':
                        outputs = model(batch['text']['input_ids'].to(self.device),
                                      batch['text']['attention_mask'].to(self.device))
                    elif modality == 'tabular':
                        outputs = model(batch['tabular'].to(self.device))
                    elif modality == 'image':
                        outputs = model(batch['image'].to(self.device))
                    
                    labels = batch['labels']
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
            
            # Calculate metrics
            y_true = np.array(all_labels)
            y_pred = np.array(all_predictions)
            y_prob = np.array(all_probabilities)
            
            metrics = evaluator._calculate_metrics(y_true, y_pred, y_prob)
            
            # Save results
            results_file = self.results_dir / f'{modality}_evaluation.txt'
            evaluator.save_detailed_results(metrics, str(results_file))
            
            # Plot results
            evaluator.plot_confusion_matrix(
                metrics['confusion_matrix'], 
                title=f'{modality.title()} Model Confusion Matrix',
                save_path=str(self.results_dir / f'{modality}_confusion_matrix.png')
            )
            
            evaluator.plot_class_performance(
                metrics, 
                save_path=str(self.results_dir / f'{modality}_class_performance.png')
            )
        
        # Evaluate fusion model
        logger.info("Evaluating fusion model...")
        
        fusion_model = self.models['fusion']
        fusion_model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Extract features
                text_features = self.models['text'].extract_quantum_features(
                    batch['text']['input_ids'].to(self.device),
                    batch['text']['attention_mask'].to(self.device)
                )
                tabular_features = self.models['tabular'].extract_quantum_features(
                    batch['tabular'].to(self.device)
                )
                image_features = self.models['image'].extract_quantum_features(
                    batch['image'].to(self.device)
                )
                
                outputs = fusion_model(text_features, tabular_features, image_features)
                labels = batch['labels']
                
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate fusion metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        fusion_metrics = evaluator._calculate_metrics(y_true, y_pred, y_prob)
        
        # Save fusion results
        results_file = self.results_dir / 'fusion_evaluation.txt'
        evaluator.save_detailed_results(fusion_metrics, str(results_file))
        
        # Plot fusion results
        evaluator.plot_confusion_matrix(
            fusion_metrics['confusion_matrix'], 
            title='Fusion Model Confusion Matrix',
            save_path=str(self.results_dir / 'fusion_confusion_matrix.png')
        )
        
        evaluator.plot_class_performance(
            fusion_metrics, 
            save_path=str(self.results_dir / 'fusion_class_performance.png')
        )
        
        evaluator.plot_roc_curves(
            y_true, y_prob,
            save_path=str(self.results_dir / 'fusion_roc_curves.png')
        )
        
        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Fusion Model Accuracy: {fusion_metrics['accuracy']:.4f}")
        print(f"Fusion Model F1-Score: {fusion_metrics['f1_score']:.4f}")
        
        if self.use_wandb:
            wandb.log({
                'final_accuracy': fusion_metrics['accuracy'],
                'final_f1_score': fusion_metrics['f1_score'],
                'final_precision': fusion_metrics['precision'],
                'final_recall': fusion_metrics['recall']
            })
        
        logger.info("Model evaluation completed")
    
    def train_complete_pipeline(self, data_dir: str = "data/mimic"):
        """
        Train the complete quantum healthcare pipeline.
        
        Args:
            data_dir: Directory containing data
        """
        logger.info("Starting complete training pipeline...")
        
        # Load data
        train_loader, test_loader = self.load_data(data_dir)
        
        # Build models
        self.build_models()
        
        # Setup optimizers
        self.setup_optimizers()
        
        # Train individual models
        individual_epochs = self.config.get('individual_epochs', 20)
        self.train_individual_models(train_loader, test_loader, individual_epochs)
        
        # Train fusion model
        fusion_epochs = self.config.get('fusion_epochs', 15)
        self.train_fusion_model(train_loader, test_loader, fusion_epochs)
        
        # Evaluate all models
        self.evaluate_models(test_loader)
        
        # Save configuration
        config_file = self.output_dir / 'training_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info("Complete training pipeline finished")
        
        if self.use_wandb:
            wandb.finish()

def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        'batch_size': 8,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'individual_epochs': 20,
        'fusion_epochs': 15,
        'patience': 8,
        'output_dir': 'outputs',
        'models': {
            'text_qubits': 8,
            'text_layers': 3,
            'tabular_qubits': 8,
            'tabular_layers': 4,
            'tabular_input_dim': 16,
            'image_qubits': 8,
            'image_layers': 3,
            'use_quantum_conv': True,
            'fusion_type': 'hybrid'
        }
    }

if __name__ == "__main__":
    # Test training pipeline
    logging.basicConfig(level=logging.INFO)
    
    config = create_default_config()
    trainer = QuantumHealthcareTrainer(config, use_wandb=False)
    
    # Train complete pipeline
    trainer.train_complete_pipeline()