# 🏥 Quantum Healthcare: Multimodal Disease Prediction

A cutting-edge quantum machine learning system for disease prediction using multimodal medical data. This project combines quantum computing with deep learning to process clinical text, vital signs, and medical images for accurate disease diagnosis.

## 🌟 Key Features

- **🧬 Quantum NLP Model**: BERT + quantum circuits for clinical text analysis
- **📊 Quantum Tabular Model**: Quantum feature maps for vital signs processing
- **🖼️ Hybrid CNN-Quantum Model**: Classical-quantum hybrid for medical imaging
- **🔗 Multimodal Fusion**: Advanced quantum interference fusion layer
- **🌐 Interactive Web UI**: Real-time disease prediction interface
- **📈 Comprehensive Evaluation**: Detailed performance metrics and analysis
- **⚡ Quick Demo**: Offline demonstration of core functionality

## 🚀 Quick Start

### Option 1: Quick Demo with Real MIMIC-IV Data
```bash
git clone https://github.com/Rupali-0806/Quantum_project.git
cd Quantum_project

# Check MIMIC-IV integration
python test_mimic_minimal.py

# Download MIMIC-IV data (requires PhysioNet access)
# See MIMIC_INTEGRATION.md for detailed instructions

# Run with real data
python src/data/data_loader.py
```

### Option 2: Demo with Synthetic Data (No Setup Required)
```bash
git clone https://github.com/Rupali-0806/Quantum_project.git
cd Quantum_project
pip install torch pennylane matplotlib numpy pandas
python demo.py
```

### Option 3: Full Installation
```bash
git clone https://github.com/Rupali-0806/Quantum_project.git
cd Quantum_project
pip install -r requirements.txt
python test_simple.py
```

### Option 4: Web Interface
```bash
# Streamlit (Recommended)
streamlit run app.py

# Gradio Alternative  
python app_gradio.py
```

## Project Structure

```
quantum_healthcare/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   └── mimic_downloader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── quantum_nlp.py
│   │   ├── quantum_tabular.py
│   │   ├── hybrid_cnn_quantum.py
│   │   └── fusion_layer.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── streamlit_app.py
│   │   └── gradio_interface.py
│   └── utils/
│       ├── __init__.py
│       ├── quantum_utils.py
│       └── medical_utils.py
├── data/
├── models/
├── notebooks/
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## 🎯 Disease Prediction Capabilities

The system can predict 9 major disease conditions:
- **Pneumonia** - Respiratory infection
- **Heart Failure** - Cardiac dysfunction  
- **Sepsis** - Systemic infection
- **COPD** - Chronic lung disease
- **Diabetes** - Metabolic disorder
- **Acute Kidney Injury** - Renal dysfunction
- **Stroke** - Cerebrovascular event
- **Myocardial Infarction** - Heart attack
- **Normal** - Healthy condition

## 🔬 Technical Architecture

### Quantum Models
- **Text Processing**: 8-qubit quantum circuits with PennyLane
- **Tabular Data**: Quantum feature maps for vital signs
- **Image Analysis**: Hybrid CNN-quantum processing
- **Fusion**: Quantum interference-based multimodal combination

### Training Pipeline
- **Individual Training**: Separate optimization for each modality
- **Fusion Training**: End-to-end multimodal learning
- **Evaluation**: Comprehensive performance analysis

## 📊 Performance

Expected performance on synthetic medical data:
- **Accuracy**: 75-85%
- **Training Time**: 30-60 minutes (CPU)
- **Inference**: Real-time prediction (<1 second)

## 🛠️ Advanced Usage

### Training Custom Models
```bash
python train.py --epochs 30 --batch-size 16 --learning-rate 0.001
```

### Running with Custom Config
```bash
python train.py --config custom_config.json
```

### Evaluation Only
```bash
python -m src.training.evaluator --model-dir outputs/models
```

## 🎓 Educational Use

Perfect for:
- **Quantum ML Research** - Novel quantum-classical hybrid architectures
- **Medical AI Education** - Multimodal healthcare applications
- **Computer Science Projects** - Advanced ML pipeline implementation
- **Healthcare Innovation** - Next-generation diagnostic tools

## ⚕️ Medical Disclaimer

This is a research prototype for educational and demonstration purposes. Always consult qualified healthcare professionals for medical decisions and diagnosis.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.