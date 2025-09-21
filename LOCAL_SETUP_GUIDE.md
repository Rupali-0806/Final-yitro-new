# 🚀 Complete Local Setup Guide for Quantum Healthcare Project

## 📋 Project Overview

**Quantum Healthcare** is a cutting-edge quantum machine learning system for multimodal disease prediction that combines quantum computing with deep learning to process clinical text, vital signs, and medical images for accurate disease diagnosis.

### 🌟 Key Features
- **🧬 Quantum NLP Model**: BERT + quantum circuits for clinical text analysis
- **📊 Quantum Tabular Model**: Quantum feature maps for vital signs processing  
- **🖼️ Hybrid CNN-Quantum Model**: Classical-quantum hybrid for medical imaging
- **🔗 Multimodal Fusion Layer**: Advanced quantum interference fusion
- **🌐 Interactive Web Interfaces**: Real-time disease prediction (Streamlit & Gradio)
- **📈 Comprehensive Evaluation**: Detailed performance metrics and analysis
- **⚡ Quick Demo Mode**: Offline demonstration of core functionality

### 🎯 Disease Prediction Capabilities
The system can predict **9 major disease conditions**:
1. **Pneumonia** - Respiratory infection
2. **Heart Failure** - Cardiac dysfunction
3. **Sepsis** - Systemic infection
4. **COPD** - Chronic lung disease
5. **Diabetes** - Metabolic disorder
6. **Acute Kidney Injury** - Renal dysfunction
7. **Stroke** - Cerebrovascular event
8. **Myocardial Infarction (MI)** - Heart attack
9. **Normal** - Healthy condition

---

## ⚙️ Prerequisites

- **Python 3.8+** (tested with 3.8, 3.9, 3.10, 3.12)
- **4+ GB RAM** (8+ GB recommended for training)
- **2+ GB free disk space** (more if using real MIMIC-IV data)
- **Internet connection** (for initial dependency downloads)

---

## 🚀 Installation Options

### Option 1: Quick Demo (⚡ Fastest Start)

**Best for**: First-time exploration, understanding concepts, demonstration

```bash
# 1. Clone the repository
git clone https://github.com/Rupali-0806/Quantum_project.git
cd Quantum_project

# 2. Install minimal dependencies
pip install torch pennylane matplotlib numpy pandas

# 3. Run offline demo
python demo.py
```

**✅ What you get:**
- Quantum circuit demonstrations with medical data
- Multimodal fusion examples  
- Clinical risk assessment calculations
- Architecture overview and explanations
- No internet required after initial install

**⏱️ Time**: ~5 minutes setup, ~30 seconds demo runtime

---

### Option 2: Web Interface (🌐 Recommended)

**Best for**: Interactive exploration, testing with custom inputs, full UI experience

```bash
# 1. Clone the repository
git clone https://github.com/Rupali-0806/Quantum_project.git
cd Quantum_project

# 2. Install web interface dependencies
pip install torch pennylane streamlit scikit-learn transformers gradio

# 3. Launch Streamlit interface (Primary UI)
streamlit run app.py

# OR launch Gradio interface (Alternative UI)
python app_gradio.py
```

**🌐 Access URLs:**
- **Streamlit**: http://localhost:8501
- **Gradio**: http://localhost:7860

**✅ What you get:**
- Interactive web interface with forms
- Real-time quantum processing
- Custom patient data input
- Disease prediction with confidence scores
- Risk assessment visualization
- Pre-loaded sample cases and examples
- Export functionality for results

**⏱️ Time**: ~10 minutes setup, ~10 seconds UI load time

---

### Option 3: Full Installation (🏥 Complete System)

**Best for**: Research, development, full functionality, training custom models

```bash
# 1. Clone the repository
git clone https://github.com/Rupali-0806/Quantum_project.git
cd Quantum_project

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Run system tests
python test_simple.py

# 4. Optional: Setup real MIMIC-IV data
python setup_credentials.py  # Set PhysioNet credentials
python download_mimic.py     # Download dataset (~10 GB)

# 5. Optional: Train models
python train.py --epochs 10  # Quick training
python train.py              # Full training (50 epochs)

# 6. Launch web interface
streamlit run app.py
```

**✅ What you get:**
- Complete quantum healthcare system
- Real MIMIC-IV medical data support
- Full training pipeline with customization
- Model performance evaluation
- Research-ready environment
- All 9 disease prediction models trained
- Export and analysis tools

**⏱️ Time**: ~20 minutes setup, 30-60 minutes training

---

### Option 4: Development Setup (🔬 For Contributors)

**Best for**: Contributing code, modifying algorithms, research experiments

```bash
# 1. Clone the repository
git clone https://github.com/Rupali-0806/Quantum_project.git
cd Quantum_project

# 2. Create virtual environment (recommended)
python -m venv quantum_env
source quantum_env/bin/activate  # On Windows: quantum_env\Scripts\activate

# 3. Install development dependencies
pip install -r requirements.txt
pip install pytest black isort flake8 jupyter

# 4. Run all tests
python test_system.py      # Full system tests
python test_simple.py      # Basic functionality
python test_code_syntax.py # Code quality

# 5. Code quality checks
black src/                 # Format code
isort src/                 # Sort imports
flake8 src/                # Lint code

# 6. Start development
jupyter notebook           # For experimentation
```

---

## 🧪 Testing Your Installation

### 1. Basic Functionality Test
```bash
python test_simple.py
```

**Expected Output:**
```
🚀 Running Simplified Quantum Healthcare Tests
==================================================
📋 Basic Imports                    ✅ PASSED
📋 Simple Quantum Model             ✅ PASSED  
📋 Data Loading                     ⚠️ May fail without internet (normal)

Results: 2/3 tests passed
```

### 2. Demo Test
```bash
python demo.py
```

**Expected Output:**
```
🏥 Quantum Healthcare AI - Demonstration
==================================================
🔬 Quantum Circuit Demonstration
Patient Data → Quantum Features:
  Healthy Patient: [-0.082, 0.322, -0.017, -0.125]
  Heart Disease  : [-0.094, -0.188, 0.058, -0.03]
  ...
🎉 DEMONSTRATION SUMMARY
==================================================
✅ Quantum circuits for medical data processing
✅ Multimodal fusion (text + tabular + image)
✅ Clinical risk assessment algorithms
```

### 3. Web Interface Test
```bash
# Start the interface
streamlit run app.py

# Check in browser: http://localhost:8501
# You should see the Quantum Healthcare interface
```

---

## 🖥️ Using the Web Interface

### Streamlit Interface (Primary)

1. **Open**: http://localhost:8501
2. **Input Patient Data:**
   - **Clinical Notes**: Enter text like "Patient presents with chest pain, shortness of breath"
   - **Vital Signs**: 
     - Heart Rate: 95 bpm
     - Blood Pressure: 140/90 mmHg  
     - Temperature: 98.6°F
     - Oxygen Saturation: 94%
   - **Chest X-ray**: Upload image (optional)
3. **Click "Predict Disease"**
4. **View Results:**
   - Disease prediction with confidence
   - Risk assessment scores
   - Clinical interpretation
   - Downloadable report

### Sample Patient Cases

**Case 1 - Heart Disease:**
- Clinical Notes: "Chest pain, elevated troponin, ECG changes"
- Heart Rate: 105 bpm
- Blood Pressure: 160/95 mmHg
- Temperature: 98.6°F
- Oxygen Saturation: 96%

**Case 2 - Pneumonia:**
- Clinical Notes: "Fever, productive cough, chest X-ray infiltrates"
- Heart Rate: 95 bpm  
- Blood Pressure: 120/80 mmHg
- Temperature: 102.1°F
- Oxygen Saturation: 92%

**Case 3 - Normal:**
- Clinical Notes: "Routine check-up, no acute complaints"
- Heart Rate: 72 bpm
- Blood Pressure: 118/75 mmHg
- Temperature: 98.6°F
- Oxygen Saturation: 98%

---

## 🏋️‍♂️ Training Custom Models

### Quick Training (Testing)
```bash
# 5 epochs, good for testing the pipeline
python train.py --epochs 5 --batch-size 8
```

### Standard Training (Research)
```bash
# Default: 20 epochs individual + 15 epochs fusion
python train.py
```

### Advanced Training (Custom)
```bash
# Full customization
python train.py \
    --epochs 50 \
    --fusion-epochs 30 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --use-wandb \
    --output-dir my_models
```

### Training with Configuration File
```bash
# Create custom config
cp config.yaml my_config.yaml
# Edit my_config.yaml as needed
python train.py --config my_config.yaml
```

---

## 📊 Expected Performance

### Hardware Requirements
| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **RAM** | 4 GB | 8 GB | 16+ GB |
| **CPU** | Dual-core | Quad-core | 8+ cores |
| **GPU** | None | Any CUDA | RTX 3080+ |
| **Storage** | 2 GB | 10 GB | 50+ GB |
| **Internet** | Setup only | Setup only | Setup only |

### Performance Expectations
| Metric | Expected Value |
|--------|----------------|
| **Demo Runtime** | ~30 seconds |
| **Web Interface Load** | ~10 seconds |
| **Prediction Time** | <1 second |
| **Training Time (CPU)** | 30-60 minutes |
| **Training Time (GPU)** | 10-20 minutes |
| **Accuracy (Synthetic)** | 75-85% |
| **Accuracy (Real MIMIC)** | 80-90% |

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Error: No module named 'X'
# Solution: Install missing dependencies
pip install torch pennylane transformers streamlit scikit-learn
```

#### 2. Internet Connection Issues
```bash
# Error: Cannot connect to huggingface.co
# Solution: Use offline mode
export TRANSFORMERS_OFFLINE=1
python demo.py  # Use demo instead
```

#### 3. Memory Issues
```bash
# Error: Out of memory
# Solution: Reduce batch size
python train.py --batch-size 4

# Or edit config.yaml:
# training:
#   batch_size: 4
```

#### 4. Port Conflicts
```bash
# Error: Port 8501 already in use
# Solution: Use different port
streamlit run app.py --server.port 8502
```

#### 5. GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU if needed
export CUDA_VISIBLE_DEVICES=""
python train.py
```

#### 6. Slow Performance
```bash
# Reduce quantum circuit complexity
# Edit config.yaml:
# quantum:
#   n_qubits: 4  # Instead of 8
#   n_layers: 1  # Instead of 2
```

---

## 📚 Using Real MIMIC-IV Data

### Prerequisites
1. **PhysioNet Account**: Register at https://physionet.org/
2. **CITI Training**: Complete required training modules
3. **MIMIC-IV Access**: Apply and get approved for MIMIC-IV dataset
4. **Credentials**: Note your PhysioNet username and password

### Setup Process
```bash
# 1. Setup credentials
python setup_credentials.py
# Enter your PhysioNet username and password when prompted

# 2. Download data (~10 GB)
python download_mimic.py
# This downloads core, ICU, and notes modules

# 3. Test integration
python test_mimic_integration.py

# 4. Train with real data
python train.py --data-dir data/mimic-iv
```

### Data Modules Downloaded
- **Core**: Admissions, patients, transfers
- **ICU**: ICU stays, chart events, output events  
- **Notes**: Discharge summaries, radiology reports

---

## 🎯 Next Steps

After successful installation:

1. **🔍 Explore the Demo**: Run `python demo.py` to understand quantum concepts
2. **🌐 Try Web Interface**: Use `streamlit run app.py` for interactive testing
3. **📖 Read Documentation**: Check `USAGE.md` for detailed usage instructions
4. **🏋️‍♂️ Train Models**: Use `python train.py` to train on your data
5. **⚙️ Customize**: Modify `config.yaml` and source code for your needs
6. **🔬 Experiment**: Try different quantum architectures and parameters
7. **📊 Analyze**: Use the evaluation tools to understand model performance

---

## 📖 Additional Documentation

- **Main README**: `README.md` - Project overview and quick start
- **Usage Guide**: `USAGE.md` - Detailed usage instructions  
- **MIMIC Integration**: `MIMIC_INTEGRATION.md` - Real data setup
- **PhysioNet Guide**: `PHYSIONET_DOWNLOAD_GUIDE.md` - Data access guide
- **Configuration**: `config.yaml` - System configuration options

---

## 🆘 Support

- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Documentation**: Comprehensive guides available
- **Code Comments**: Well-documented source code

---

## ⚕️ Important Disclaimer

This is a **research prototype for educational purposes**. The quantum healthcare system is designed for:
- ✅ **Research and Development**
- ✅ **Educational Demonstrations**  
- ✅ **Proof of Concept Studies**
- ✅ **Algorithm Development**

**❌ NOT for clinical use or medical decisions**

Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.

---

## 🏁 Quick Start Summary

**For a 5-minute demo:**
```bash
git clone https://github.com/Rupali-0806/Quantum_project.git
cd Quantum_project
pip install torch pennylane matplotlib numpy pandas
python demo.py
```

**For interactive web interface:**
```bash
git clone https://github.com/Rupali-0806/Quantum_project.git
cd Quantum_project  
pip install torch pennylane streamlit scikit-learn transformers
streamlit run app.py
# Open: http://localhost:8501
```

**For full research setup:**
```bash
git clone https://github.com/Rupali-0806/Quantum_project.git
cd Quantum_project
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Enjoy exploring the future of quantum machine learning in healthcare! 🚀🏥