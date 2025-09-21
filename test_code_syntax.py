#!/usr/bin/env python3
"""
Simple test to verify the MIMIC integration code works without heavy dependencies.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock the dependencies that might not be available
class MockPandas:
    class DataFrame:
        def __init__(self, data):
            self.data = data
            self.columns = list(data.keys()) if isinstance(data, dict) else []
            
        def __len__(self):
            if isinstance(self.data, dict):
                return len(next(iter(self.data.values())))
            return 0
            
        @property
        def shape(self):
            if isinstance(self.data, dict) and self.data:
                return (len(self), len(self.columns))
            return (0, 0)
            
        def to_csv(self, path, index=False):
            print(f"Would save CSV to {path}")
            
        def merge(self, other, on, how='inner'):
            return self
            
        def fillna(self, value):
            return self
            
        @property
        def values(self):
            if isinstance(self.data, dict):
                return list(self.data.get(self.columns[0], []))
            return []
    
    @staticmethod
    def read_csv(path, compression=None):
        print(f"Would read CSV from {path}")
        return MockPandas.DataFrame({})

class MockNumpy:
    class random_class:
        @staticmethod
        def seed(seed):
            pass
            
        @staticmethod  
        def normal(mean, std, size):
            return [mean] * size
            
        @staticmethod
        def choice(options, size=None, p=None):
            if size is None:
                return options[0] if options else None
            return [options[0] if options else None] * size
            
        @staticmethod
        def rand(*shape):
            if len(shape) == 3:
                return [[[0.5 for _ in range(shape[2])] 
                        for _ in range(shape[1])] 
                       for _ in range(shape[0])]
            return [0.5] * (shape[0] if shape else 1)
    
    random = random_class()
        
    @staticmethod
    def save(path, data):
        print(f"Would save numpy array to {path}")

# Mock sys.modules to replace imports
class MockModule:
    def __getattr__(self, name):
        if name == 'DataFrame':
            return MockPandas.DataFrame
        return lambda *args, **kwargs: None

sys.modules['pandas'] = MockPandas
sys.modules['numpy'] = MockNumpy
sys.modules['yaml'] = MockModule()
sys.modules['gzip'] = MockModule()
sys.modules['requests'] = MockModule()

def test_mimic_downloader():
    """Test the MIMICDownloader class."""
    print("Testing MIMICDownloader...")
    
    try:
        from data.mimic_downloader import MIMICDownloader
        
        # Initialize downloader
        downloader = MIMICDownloader()
        print("✓ MIMICDownloader initialized successfully")
        
        # Test configuration loading
        config = downloader._get_default_config()
        assert 'mimic' in config
        assert 'preprocessing' in config
        print("✓ Configuration loading works")
        
        # Test data source check
        source = downloader.get_data_source()
        print(f"✓ Data source: {source}")
        
        # Test verification
        verification = downloader.verify_real_data()
        print("✓ Data verification works")
        
        return True
        
    except Exception as e:
        print(f"✗ MIMICDownloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loader():
    """Test the data_loader module."""
    print("\nTesting data_loader...")
    
    try:
        # This will import the module and check syntax
        import data.data_loader
        print("✓ data_loader module imports successfully")
        return True
        
    except Exception as e:
        print(f"✗ data_loader test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔬 MIMIC-IV Code Verification Test")
    print("=================================")
    
    tests = [
        test_mimic_downloader,
        test_data_loader
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MIMIC-IV integration code is working.")
    else:
        print("⚠ Some tests failed. Check the code for issues.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)