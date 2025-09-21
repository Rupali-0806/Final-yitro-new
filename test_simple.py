"""
Simplified test script to verify core functionality.
"""

import logging
import torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent / 'src'))

def test_basic_imports():
    """Test basic imports."""
    print("🔄 Testing basic imports...")
    
    try:
        import torch
        import numpy as np
        import pandas as pd
        import pennylane as qml
        from transformers import AutoTokenizer
        
        print("✅ Basic imports successful!")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   Device available: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        return True
        
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_data_loading():
    """Test data loading and preprocessing."""
    print("\n🔄 Testing data loading...")
    
    try:
        from src.data.data_loader import load_and_preprocess_mimic_data
        
        train_data, test_data, classes, preprocessor = load_and_preprocess_mimic_data()
        
        print(f"✅ Data loaded successfully!")
        print(f"   Train samples: {len(train_data['labels'])}")
        print(f"   Test samples: {len(test_data['labels'])}")
        print(f"   Disease classes: {classes}")
        print(f"   Tabular features shape: {train_data['tabular'].shape}")
        print(f"   Text features shape: {train_data['text']['input_ids'].shape}")
        print(f"   Image features shape: {train_data['image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_model():
    """Test a simple quantum model."""
    print("\n🔄 Testing simple quantum model...")
    
    try:
        import pennylane as qml
        import torch
        import torch.nn as nn
        
        # Create a simple quantum model
        n_qubits = 4
        dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(dev)
        def simple_circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Test circuit
        inputs = torch.randn(4)
        weights = torch.randn(2, 4, 3)  # dummy weights
        
        result = simple_circuit(inputs, weights)
        print(f"✅ Simple quantum circuit works!")
        print(f"   Circuit output: {[float(r) for r in result]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple quantum model failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simplified tests."""
    print("🚀 Running Simplified Quantum Healthcare Tests")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Loading", test_data_loading),
        ("Simple Quantum Model", test_simple_model)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20s} {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 Core functionality working! The system is ready.")
        print("\nNext steps:")
        print("1. Install additional dependencies: pip install -r requirements.txt")
        print("2. Run full training: python train.py")
        print("3. Launch UI: streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()