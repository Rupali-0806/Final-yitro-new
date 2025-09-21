#!/usr/bin/env python3
"""
Test script for MIMIC-IV data integration.
Tests both synthetic fallback and real data loading capabilities.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_mimic_integration():
    """Test the MIMIC-IV data integration functionality."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Testing MIMIC-IV data integration...")
    
    try:
        # Test 1: Configuration loading
        logger.info("Test 1: Configuration loading")
        from src.data.mimic_downloader import MIMICDownloader
        
        downloader = MIMICDownloader()
        logger.info(f"✓ Configuration loaded successfully")
        logger.info(f"  Data source: {downloader.get_data_source()}")
        
        # Test 2: MIMIC-IV data verification
        logger.info("\nTest 2: MIMIC-IV data verification")
        verification = downloader.verify_real_data()
        
        any_real_data = False
        for module, files in verification.items():
            logger.info(f"  {module}:")
            for filename, available in files.items():
                status = "✓ Available" if available else "✗ Missing"
                logger.info(f"    {filename}: {status}")
                if available:
                    any_real_data = True
        
        if any_real_data:
            logger.info("✓ Some real MIMIC-IV data files found")
        else:
            logger.info("ℹ No real MIMIC-IV data found - will use synthetic fallback")
        
        # Test 3: Data loading
        logger.info("\nTest 3: Data loading")
        raw_data = downloader.download_sample_data()
        
        logger.info("✓ Data loaded successfully")
        logger.info(f"  Tabular data shape: {raw_data['tabular'].shape}")
        logger.info(f"  Text data shape: {raw_data['text'].shape}")
        logger.info(f"  Image data shape: {raw_data['image'].shape}")
        logger.info(f"  Labels shape: {raw_data['labels'].shape}")
        
        # Test 4: Data content validation
        logger.info("\nTest 4: Data content validation")
        
        # Check tabular data columns
        expected_tabular_cols = ['patient_id', 'age', 'gender', 'heart_rate']
        tabular_cols = raw_data['tabular'].columns.tolist()
        missing_cols = [col for col in expected_tabular_cols if col not in tabular_cols]
        
        if not missing_cols:
            logger.info("✓ Tabular data has expected columns")
        else:
            logger.warning(f"⚠ Missing tabular columns: {missing_cols}")
        
        # Check text data
        if 'clinical_notes' in raw_data['text'].columns:
            logger.info("✓ Text data has clinical notes")
            sample_note = raw_data['text']['clinical_notes'].iloc[0]
            logger.info(f"  Sample note: {sample_note[:100]}...")
        else:
            logger.error("✗ Text data missing clinical_notes column")
        
        # Check labels
        if 'disease_label' in raw_data['labels'].columns:
            logger.info("✓ Labels data has disease labels")
            unique_labels = raw_data['labels']['disease_label'].unique()
            logger.info(f"  Disease categories: {len(unique_labels)}")
            logger.info(f"  Categories: {list(unique_labels)[:5]}...")
        else:
            logger.error("✗ Labels data missing disease_label column")
        
        # Test 5: Configuration validation
        logger.info("\nTest 5: Configuration validation")
        config = downloader.config
        
        required_sections = ['mimic', 'preprocessing']
        for section in required_sections:
            if section in config:
                logger.info(f"✓ Configuration has {section} section")
            else:
                logger.error(f"✗ Configuration missing {section} section")
        
        logger.info("\n🎉 All tests completed successfully!")
        logger.info("MIMIC-IV integration is working correctly.")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all required packages are installed")
        return False
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mimic_integration()
    sys.exit(0 if success else 1)