"""
Quick test script to verify data loading and model initialization.
"""

import logging
import torch
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.data_loader import load_and_preprocess_mimic_data
from src.models.quantum_nlp import build_quantum_nlp_model
from src.models.quantum_tabular import build_quantum_tabular_model
from src.models.hybrid_cnn_quantum import build_hybrid_cnn_quantum_model
from src.models.fusion_layer import create_fusion_layer

def test_data_loading():
    """Test data loading and preprocessing."""
    print("🔄 Testing data loading...")
    
    try:
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
        return False

def test_model_building():
    """Test model building and forward pass."""
    print("\n🔄 Testing model building...")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using device: {device}")
        
        # Build models
        text_model = build_quantum_nlp_model(n_qubits=8, n_layers=2, output_dim=9)
        tabular_model = build_quantum_tabular_model(input_dim=16, n_qubits=8, n_layers=2, output_dim=9)
        image_model = build_hybrid_cnn_quantum_model(n_qubits=8, n_layers=2, output_dim=9)
        fusion_model = create_fusion_layer(text_dim=8, tabular_dim=8, image_dim=8, output_dim=9)
        
        print("✅ Models built successfully!")
        
        # Test forward pass with dummy data
        batch_size = 2
        
        # Text data
        input_ids = torch.randint(0, 1000, (batch_size, 128))
        attention_mask = torch.ones(batch_size, 128)
        text_output = text_model(input_ids, attention_mask)
        print(f"   Text model output shape: {text_output.shape}")
        
        # Tabular data
        tabular_input = torch.randn(batch_size, 16)
        tabular_output = tabular_model(tabular_input)
        print(f"   Tabular model output shape: {tabular_output.shape}")
        
        # Image data
        image_input = torch.randn(batch_size, 1, 224, 224)
        image_output = image_model(image_input)
        print(f"   Image model output shape: {image_output.shape}")
        
        # Fusion
        text_features = text_model.extract_quantum_features(input_ids, attention_mask)
        tabular_features = tabular_model.extract_quantum_features(tabular_input)
        image_features = image_model.extract_quantum_features(image_input)
        
        fusion_output = fusion_model(text_features, tabular_features, image_features)
        print(f"   Fusion model output shape: {fusion_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model building failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_imports():
    """Test UI module imports."""
    print("\n🔄 Testing UI imports...")
    
    try:
        from src.ui.streamlit_app import main as streamlit_main
        from src.ui.gradio_interface import create_gradio_interface
        
        print("✅ UI modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ UI import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running Quantum Healthcare System Tests")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Building", test_model_building),
        ("UI Imports", test_ui_imports)
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
        print("\n🎉 All tests passed! The system is ready for training.")
        print("\nNext steps:")
        print("1. Run training: python train.py")
        print("2. Launch UI: streamlit run src/ui/streamlit_app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()