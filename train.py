"""
Main training script for quantum healthcare models.
"""

import argparse
import logging
import json
from pathlib import Path

from src.training.trainer import QuantumHealthcareTrainer, create_default_config

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Quantum Healthcare Models")
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data/mimic', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for individual models')
    parser.add_argument('--fusion-epochs', type=int, default=15, help='Number of epochs for fusion model')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Quantum Healthcare Training Pipeline")
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    config.update({
        'batch_size': args.batch_size,
        'individual_epochs': args.epochs,
        'fusion_epochs': args.fusion_epochs,
        'learning_rate': args.learning_rate,
        'output_dir': args.output_dir
    })
    
    logger.info(f"Training configuration: {json.dumps(config, indent=2)}")
    
    # Initialize trainer
    trainer = QuantumHealthcareTrainer(config, use_wandb=args.use_wandb)
    
    # Run complete training pipeline
    trainer.train_complete_pipeline(args.data_dir)
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()