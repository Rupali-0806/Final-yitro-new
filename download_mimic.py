#!/usr/bin/env python3
"""
MIMIC-IV Dataset Downloader
Downloads MIMIC-IV data from PhysioNet using provided credentials.
If PhysioNet is not accessible, creates demo data structure for testing.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import MIMICDownloader directly to avoid dependency issues
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'data'))
from mimic_downloader import MIMICDownloader

def main():
    """Main download function."""
    logger.info("🏥 MIMIC-IV Dataset Downloader")
    logger.info("=" * 50)
    
    # PhysioNet credentials
    username = "uragul500@gmail.com"
    password = "Ragul@4321"
    
    logger.info(f"Using PhysioNet username: {username}")
    
    try:
        # Create downloader with credentials
        downloader = MIMICDownloader(
            physionet_username=username,
            physionet_password=password
        )
        
        # Check current data status
        logger.info("Checking current data status...")
        data_source = downloader.get_data_source()
        logger.info(f"Current data source: {data_source}")
        
        # If real data not available, try to download it
        if not downloader._check_mimic_data_available():
            logger.info("Real MIMIC-IV data not found. Attempting download...")
            
            try:
                # Try to download the data
                downloader._download_from_physionet()
                logger.info("✅ Download completed successfully!")
                
            except Exception as download_error:
                logger.warning(f"PhysioNet download failed: {download_error}")
                logger.info("Creating demo data structure for testing...")
                
                # Create demo data structure as fallback
                downloader._create_demo_data_structure()
                logger.info("✅ Demo data structure created!")
                
        else:
            logger.info("✅ MIMIC-IV data already available locally.")
        
        # Verify available data
        logger.info("Verifying available data...")
        verification = downloader.verify_real_data()
        
        all_available = True
        for module, files in verification.items():
            logger.info(f"\n{module.upper()} module:")
            for name, available in files.items():
                status = "✓" if available else "✗"
                logger.info(f"  {name}: {status}")
                if not available:
                    all_available = False
        
        if all_available:
            logger.info("\n🎉 All required MIMIC-IV files are available!")
            
            # Test loading the data
            logger.info("Testing data loading...")
            data = downloader.download_sample_data()
            
            logger.info("Data loading test results:")
            logger.info(f"  Tabular data shape: {data['tabular'].shape}")
            logger.info(f"  Text data shape: {data['text'].shape}")
            logger.info(f"  Image data shape: {data['image'].shape}")
            logger.info(f"  Labels shape: {data['labels'].shape}")
            
            # Show data source
            source = downloader.get_data_source()
            logger.info(f"\nData source: {source}")
            
            logger.info("\n✅ MIMIC-IV dataset is ready for use!")
            
            # Show sample of the data
            logger.info("\nSample tabular data:")
            print(data['tabular'].head())
            
            logger.info("\nSample text data:")
            print(data['text'].head())
            
            logger.info("\nSample labels:")
            print(data['labels'].head())
            
        else:
            logger.warning("⚠️  Some required files are missing.")
            logger.info("The system will fall back to synthetic data when needed.")
            
    except Exception as e:
        logger.error(f"❌ Error during setup: {e}")
        logger.info("Falling back to synthetic data mode...")
        
        # Test synthetic data as ultimate fallback
        try:
            downloader = MIMICDownloader()
            data = downloader.download_sample_data()
            logger.info("✅ Synthetic data fallback is working!")
            logger.info(f"Synthetic data shapes - Tabular: {data['tabular'].shape}, Text: {data['text'].shape}")
        except Exception as fallback_error:
            logger.error(f"❌ Even synthetic data failed: {fallback_error}")
            sys.exit(1)

if __name__ == "__main__":
    main()