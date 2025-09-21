"""
Main data loading interface for MIMIC-IV multimodal data.
"""

import logging
from typing import Tuple, Dict, Any
from .mimic_downloader import MIMICDownloader
from .preprocessor import MIMICPreprocessor

logger = logging.getLogger(__name__)

def load_and_preprocess_mimic_data(data_dir: str = "data/mimic", 
                                  download_fresh: bool = False,
                                  config_path: str = "config.yaml") -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Main function to load and preprocess MIMIC-IV multimodal data.
    
    Args:
        data_dir: Directory to store/load data
        download_fresh: Whether to download fresh data or use existing
        config_path: Path to configuration file
        
    Returns:
        Tuple of (train_data, test_data, label_classes, preprocessor)
    """
    logger.info("Starting MIMIC-IV data loading and preprocessing...")
    
    # Initialize components
    downloader = MIMICDownloader(data_dir, config_path)
    preprocessor = MIMICPreprocessor()
    
    # Log data source information
    logger.info(f"Data source: {downloader.get_data_source()}")
    
    # Load or download data
    if download_fresh:
        raw_data = downloader.download_sample_data()
    else:
        raw_data = downloader.load_existing_data()
        if raw_data is None:
            logger.info("No existing data found, attempting to download/generate data...")
            raw_data = downloader.download_sample_data()
    
    # Preprocess each modality
    logger.info("Preprocessing all modalities...")
    
    # Preprocess tabular data
    tabular_processed = preprocessor.preprocess_tabular_data(raw_data['tabular'])
    
    # Preprocess text data
    text_processed = preprocessor.preprocess_text_data(raw_data['text'])
    
    # Preprocess image data
    image_processed = preprocessor.preprocess_image_data(raw_data['image'])
    
    # Preprocess labels
    labels_processed = preprocessor.preprocess_labels(raw_data['labels'])
    
    # Create train/test splits
    train_data, test_data = preprocessor.create_train_test_split(
        tabular_processed, text_processed, image_processed, labels_processed
    )
    
    logger.info("Data loading and preprocessing complete!")
    
    return train_data, test_data, preprocessor.get_label_classes(), preprocessor

if __name__ == "__main__":
    # Test the data loading pipeline
    logging.basicConfig(level=logging.INFO)
    
    train_data, test_data, classes, preprocessor = load_and_preprocess_mimic_data()
    
    print(f"Train data shapes:")
    print(f"  Tabular: {train_data['tabular'].shape}")
    print(f"  Text input_ids: {train_data['text']['input_ids'].shape}")
    print(f"  Images: {train_data['image'].shape}")
    print(f"  Labels: {train_data['labels'].shape}")
    
    print(f"\nTest data shapes:")
    print(f"  Tabular: {test_data['tabular'].shape}")
    print(f"  Text input_ids: {test_data['text']['input_ids'].shape}")
    print(f"  Images: {test_data['image'].shape}")
    print(f"  Labels: {test_data['labels'].shape}")
    
    print(f"\nDisease classes: {classes}")
    
    # Test MIMIC-IV data verification
    downloader = MIMICDownloader()
    verification = downloader.verify_real_data()
    print(f"\nMIMIC-IV data verification:")
    for module, files in verification.items():
        print(f"  {module}:")
        for filename, available in files.items():
            status = "✓" if available else "✗"
            print(f"    {filename}: {status}")
    
    print(f"\nData source: {downloader.get_data_source()}")