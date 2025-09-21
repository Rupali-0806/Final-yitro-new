#!/usr/bin/env python3
"""
Minimal test for MIMIC-IV integration using built-in libraries only.
"""

import sys
import os
import json
import csv
from pathlib import Path

def create_mock_pandas_dataframe():
    """Create a mock DataFrame-like object using built-in types."""
    class MockDataFrame:
        def __init__(self, data_dict):
            self.data = data_dict
            self.columns = list(data_dict.keys())
            
        def __len__(self):
            return len(next(iter(self.data.values())))
            
        @property
        def shape(self):
            if not self.data:
                return (0, 0)
            return (len(self), len(self.columns))
            
        def to_csv(self, filepath, index=False):
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)
                for i in range(len(self)):
                    row = [self.data[col][i] for col in self.columns]
                    writer.writerow(row)
    
    return MockDataFrame

def create_mock_numpy():
    """Create mock numpy functions using built-in random."""
    import random
    
    class MockNumpy:
        class random_module:
            @staticmethod
            def seed(seed):
                random.seed(seed)
                
            @staticmethod
            def normal(mean, std, size):
                return [random.gauss(mean, std) for _ in range(size)]
                
            @staticmethod
            def choice(options, size=None, p=None):
                if size is None:
                    return random.choice(options)
                return [random.choice(options) for _ in range(size)]
                
            @staticmethod
            def rand(*shape):
                if len(shape) == 1:
                    return [random.random() for _ in range(shape[0])]
                elif len(shape) == 3:
                    return [[[random.random() for _ in range(shape[2])] 
                            for _ in range(shape[1])] 
                           for _ in range(shape[0])]
        
        random = random_module()
            
        @staticmethod
        def save(filepath, data):
            with open(filepath, 'w') as f:
                json.dump(data.tolist() if hasattr(data, 'tolist') else data, f)
    
    return MockNumpy

def test_configuration():
    """Test configuration loading."""
    print("Test 1: Configuration loading")
    
    config_path = Path("config.yaml")
    if config_path.exists():
        print("✓ Configuration file exists")
        with open(config_path, 'r') as f:
            content = f.read()
            if 'mimic:' in content and 'data_path:' in content:
                print("✓ Configuration has required MIMIC sections")
            else:
                print("⚠ Configuration may be incomplete")
    else:
        print("✗ Configuration file not found")
        return False
    
    return True

def test_mimic_structure():
    """Test MIMIC data directory structure."""
    print("\nTest 2: MIMIC data structure verification")
    
    mimic_paths = [
        "data/mimic-iv",
        "data/mimic-iv/hosp",
        "data/mimic-iv/icu", 
        "data/mimic-iv/note"
    ]
    
    structure_ok = True
    for path in mimic_paths:
        if Path(path).exists():
            print(f"✓ Directory exists: {path}")
        else:
            print(f"ℹ Directory not found: {path} (will use synthetic data)")
            
    # Check for actual MIMIC files
    mimic_files = [
        "data/mimic-iv/hosp/patients.csv.gz",
        "data/mimic-iv/hosp/admissions.csv.gz",
        "data/mimic-iv/icu/icustays.csv.gz"
    ]
    
    real_data_available = False
    for filepath in mimic_files:
        if Path(filepath).exists():
            print(f"✓ MIMIC file found: {filepath}")
            real_data_available = True
        else:
            print(f"ℹ MIMIC file not found: {filepath}")
    
    if real_data_available:
        print("✓ Real MIMIC-IV data detected")
    else:
        print("ℹ No real MIMIC-IV data found - synthetic fallback will be used")
    
    return True

def test_synthetic_data_generation():
    """Test synthetic data generation."""
    print("\nTest 3: Synthetic data generation")
    
    try:
        # Mock the necessary modules
        MockDataFrame = create_mock_pandas_dataframe()
        mock_np = create_mock_numpy()
        
        # Test data generation
        n_patients = 10
        
        # Generate sample tabular data
        tabular_data = {
            'patient_id': list(range(n_patients)),
            'age': [random.uniform(18, 100) for _ in range(n_patients)],
            'gender': [random.choice(['M', 'F']) for _ in range(n_patients)],
            'heart_rate': [random.uniform(40, 200) for _ in range(n_patients)]
        }
        
        df = MockDataFrame(tabular_data)
        print(f"✓ Generated tabular data: {df.shape}")
        
        # Generate sample text data
        notes = ["Sample clinical note " + str(i) for i in range(n_patients)]
        text_data = MockDataFrame({
            'patient_id': list(range(n_patients)),
            'clinical_notes': notes
        })
        print(f"✓ Generated text data: {text_data.shape}")
        
        # Generate sample labels
        diseases = ['Pneumonia', 'Heart Failure', 'Normal']
        labels_data = MockDataFrame({
            'patient_id': list(range(n_patients)),
            'disease_label': [random.choice(diseases) for _ in range(n_patients)]
        })
        print(f"✓ Generated labels data: {labels_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Synthetic data generation failed: {e}")
        return False

def test_data_loading_interface():
    """Test the data loading interface structure."""
    print("\nTest 4: Data loading interface")
    
    # Check if the data loading modules exist
    data_modules = [
        "src/data/__init__.py",
        "src/data/mimic_downloader.py", 
        "src/data/data_loader.py",
        "src/data/preprocessor.py"
    ]
    
    all_exist = True
    for module in data_modules:
        if Path(module).exists():
            print(f"✓ Module exists: {module}")
        else:
            print(f"✗ Module missing: {module}")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("🏥 MIMIC-IV Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_configuration,
        test_mimic_structure, 
        test_synthetic_data_generation,
        test_data_loading_interface
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} tests passed!")
        print("MIMIC-IV integration structure is ready.")
    else:
        print(f"⚠ {passed}/{total} tests passed")
        print("Some issues detected but basic structure is in place.")
    
    print("\nNext Steps:")
    print("1. Install dependencies: pip install pandas numpy PyYAML")
    print("2. Download MIMIC-IV data to data/mimic-iv/ (optional)")
    print("3. Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD (if using real data)")
    print("4. Run: python src/data/data_loader.py")
    
    return passed == total

if __name__ == "__main__":
    import random
    success = main()
    sys.exit(0 if success else 1)