"""
Model evaluation utilities for quantum healthcare models.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive evaluation utility for quantum healthcare models.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of disease class names
        """
        self.class_names = class_names
        self.n_classes = len(class_names)
        
    def evaluate_model(self, model, data_loader, device: str = 'cpu') -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            data_loader: Data loader for evaluation
            device: Device for computation
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Handle different data formats
                if isinstance(batch, dict):
                    # Multimodal data
                    tabular = batch['tabular'].to(device)
                    text = {k: v.to(device) for k, v in batch['text'].items()}
                    image = batch['image'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(tabular, text, image)
                else:
                    # Single modality data
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(inputs)
                
                # Get predictions
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_true_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # ROC AUC (multiclass)
        try:
            if self.n_classes > 2:
                y_true_binarized = label_binarize(y_true, classes=range(self.n_classes))
                auc_score = roc_auc_score(y_true_binarized, y_prob, multi_class='ovr', average='weighted')
            else:
                auc_score = roc_auc_score(y_true, y_prob[:, 1])
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            auc_score = None
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'per_class_metrics': {
                'precision': dict(zip(self.class_names, precision_per_class)),
                'recall': dict(zip(self.class_names, recall_per_class)),
                'f1_score': dict(zip(self.class_names, f1_per_class))
            },
            'predictions': y_pred,
            'true_labels': y_true,
            'probabilities': y_prob
        }
        
        return metrics
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                             title: str = 'Confusion Matrix', 
                             save_path: str = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_class_performance(self, metrics: Dict[str, Any], 
                              save_path: str = None) -> plt.Figure:
        """
        Plot per-class performance metrics.
        
        Args:
            metrics: Evaluation metrics dictionary
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        per_class = metrics['per_class_metrics']
        
        # Prepare data
        classes = self.class_names
        precision = [per_class['precision'][cls] for cls in classes]
        recall = [per_class['recall'][cls] for cls in classes]
        f1 = [per_class['f1_score'][cls] for cls in classes]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Disease Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray,
                       save_path: str = None) -> plt.Figure:
        """
        Plot ROC curves for multiclass classification.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import roc_curve, auc
        
        # Binarize labels
        y_true_binarized = label_binarize(y_true, classes=range(self.n_classes))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, linewidth=2, 
                   label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_evaluation_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            metrics: Evaluation metrics dictionary
            
        Returns:
            Formatted evaluation report string
        """
        report = "=== QUANTUM HEALTHCARE MODEL EVALUATION REPORT ===\n\n"
        
        # Overall metrics
        report += "OVERALL PERFORMANCE:\n"
        report += f"Accuracy: {metrics['accuracy']:.4f}\n"
        report += f"Precision: {metrics['precision']:.4f}\n"
        report += f"Recall: {metrics['recall']:.4f}\n"
        report += f"F1-Score: {metrics['f1_score']:.4f}\n"
        
        if metrics['auc_score'] is not None:
            report += f"AUC Score: {metrics['auc_score']:.4f}\n"
        
        report += "\n"
        
        # Per-class performance
        report += "PER-CLASS PERFORMANCE:\n"
        per_class = metrics['per_class_metrics']
        
        for class_name in self.class_names:
            report += f"\n{class_name}:\n"
            report += f"  Precision: {per_class['precision'][class_name]:.4f}\n"
            report += f"  Recall: {per_class['recall'][class_name]:.4f}\n"
            report += f"  F1-Score: {per_class['f1_score'][class_name]:.4f}\n"
        
        # Confusion matrix summary
        report += "\nCONFUSION MATRIX DIAGONAL (Correct Predictions):\n"
        conf_matrix = metrics['confusion_matrix']
        for i, class_name in enumerate(self.class_names):
            total_class = conf_matrix[i].sum()
            correct = conf_matrix[i, i]
            report += f"{class_name}: {correct}/{total_class} ({correct/total_class:.2%})\n"
        
        return report
    
    def save_detailed_results(self, metrics: Dict[str, Any], 
                             file_path: str):
        """
        Save detailed evaluation results to file.
        
        Args:
            metrics: Evaluation metrics dictionary
            file_path: Path to save results
        """
        # Create detailed results DataFrame
        results_data = []
        
        for i, class_name in enumerate(self.class_names):
            per_class = metrics['per_class_metrics']
            
            results_data.append({
                'Disease': class_name,
                'Precision': per_class['precision'][class_name],
                'Recall': per_class['recall'][class_name],
                'F1_Score': per_class['f1_score'][class_name],
                'Support': metrics['classification_report'][class_name]['support']
            })
        
        # Add overall metrics
        results_data.append({
            'Disease': 'OVERALL',
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score'],
            'Support': len(metrics['true_labels'])
        })
        
        df = pd.DataFrame(results_data)
        
        # Save to CSV
        csv_path = file_path.replace('.txt', '.csv')
        df.to_csv(csv_path, index=False)
        
        # Save text report
        with open(file_path, 'w') as f:
            f.write(self.generate_evaluation_report(metrics))
        
        logger.info(f"Detailed results saved to {file_path} and {csv_path}")

if __name__ == "__main__":
    # Test the evaluator
    logging.basicConfig(level=logging.INFO)
    
    class_names = ['Pneumonia', 'Heart Failure', 'Sepsis', 'COPD', 'Normal']
    evaluator = ModelEvaluator(class_names)
    
    # Generate dummy evaluation data
    n_samples = 100
    n_classes = len(class_names)
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.rand(n_samples, n_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
    
    # Calculate metrics
    metrics = evaluator._calculate_metrics(y_true, y_pred, y_prob)
    
    # Generate report
    report = evaluator.generate_evaluation_report(metrics)
    print(report)