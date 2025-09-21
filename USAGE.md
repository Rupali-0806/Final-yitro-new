# Quantum Healthcare Project

## Overview

This project implements a comprehensive quantum machine learning pipeline for multimodal disease prediction using clinical text, vital signs, and medical imaging data.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Rupali-0806/Quantum_project.git
cd Quantum_project

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Test the System

```bash
python test_system.py
```

### Train Models

```bash
python train.py --epochs 20 --batch-size 8
```

### Launch Web Interface

#### Streamlit (Recommended)
```bash
streamlit run app.py
```

#### Gradio Alternative
```bash
python app_gradio.py
```

## Features

- **🧬 Quantum NLP**: BERT + quantum circuits for clinical text
- **📊 Quantum Tabular**: Quantum feature maps for vital signs
- **🖼️ Hybrid CNN-Quantum**: Classical-quantum image processing
- **🔗 Multimodal Fusion**: Quantum interference for data fusion
- **🌐 Web Interface**: Interactive disease prediction system
- **📈 Comprehensive Evaluation**: Detailed performance metrics

## Project Structure

```
quantum_healthcare/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Quantum ML models
│   ├── training/       # Training and evaluation
│   ├── ui/            # Web interfaces
│   └── utils/         # Utilities
├── data/              # Dataset storage
├── outputs/           # Training outputs
├── train.py          # Training script
├── test_system.py    # System tests
├── app.py            # Streamlit launcher
└── app_gradio.py     # Gradio launcher
```

## Usage Examples

### Command Line Training

```bash
# Basic training
python train.py

# Advanced training with custom parameters
python train.py --epochs 30 --batch-size 16 --learning-rate 0.001 --use-wandb

# Training with custom config
python train.py --config config.json
```

### Programmatic Usage

```python
from src.training.trainer import QuantumHealthcareTrainer, create_default_config

# Initialize trainer
config = create_default_config()
trainer = QuantumHealthcareTrainer(config)

# Train models
trainer.train_complete_pipeline()
```

### Web Interface Usage

1. **Launch Interface**: Run `streamlit run app.py`
2. **Enter Clinical Data**: 
   - Clinical notes (text)
   - Vital signs (tabular)
   - Chest X-ray (image, optional)
3. **Get Prediction**: View disease prediction with confidence and risk assessment

## Model Architecture

### Quantum NLP Model
- BERT embeddings → Quantum processing (8 qubits)
- Variational quantum circuits for text understanding
- Clinical text → Disease features

### Quantum Tabular Model  
- Quantum feature maps for vital signs
- Multi-layer quantum encoding
- Tabular data → Quantum features

### Hybrid CNN-Quantum Model
- Classical CNN backbone
- Quantum convolutional layers
- Medical images → Hybrid features

### Multimodal Fusion
- Attention-based fusion
- Quantum interference fusion
- Combined → Disease prediction

## Dataset

The project now supports both **real MIMIC-IV data** and synthetic fallback:

### Real MIMIC-IV Data (Recommended for Research)
- **Patient Demographics**: Age, gender, admission details from PhysioNet  
- **Clinical Notes**: Real discharge summaries and clinical text
- **Vital Signs**: ICU measurements, lab values, physiological data
- **Disease Labels**: Derived from ICD codes and admission diagnoses
- **Data Source**: PhysioNet MIMIC-IV Database v2.2

**Setup**: See [MIMIC_INTEGRATION.md](MIMIC_INTEGRATION.md) for detailed instructions.

### Synthetic MIMIC-IV-style Data (Development/Demo)
- **Clinical Notes**: Generated medical text based on templates
- **Vital Signs**: Simulated ICU measurements with realistic distributions  
- **Chest X-rays**: Synthetic medical images for demonstration
- **Disease Labels**: 9 common conditions with realistic prevalence

**Auto-fallback**: System automatically uses synthetic data when real MIMIC-IV data is unavailable.

For real MIMIC-IV data, obtain access from PhysioNet and follow the setup guide in MIMIC_INTEGRATION.md.

## Configuration

### Training Configuration

```json
{
  "batch_size": 8,
  "learning_rate": 0.001,
  "individual_epochs": 20,
  "fusion_epochs": 15,
  "models": {
    "text_qubits": 8,
    "tabular_qubits": 8,
    "image_qubits": 8,
    "fusion_type": "hybrid"
  }
}
```

### Model Configurations

- **Qubits**: Number of quantum bits (4-16 recommended)
- **Layers**: Quantum circuit depth (2-5 recommended)
- **Fusion Type**: "attention", "quantum", "hybrid", or "concat"

## Performance

Expected performance on synthetic data:
- **Accuracy**: 75-85%
- **F1-Score**: 0.70-0.80
- **Training Time**: 30-60 minutes (CPU)

Real-world performance depends on data quality and quantum hardware access.

## Development

### Running Tests

```bash
# System tests
python test_system.py

# Unit tests (if implemented)
pytest tests/

# Code quality
black src/
isort src/
flake8 src/
```

### Adding New Models

1. Create model in `src/models/`
2. Add to training pipeline in `src/training/trainer.py`
3. Update fusion layer if needed
4. Add UI components

### Extending Data Modalities

1. Add preprocessor in `src/data/preprocessor.py`
2. Create quantum model for new modality
3. Update fusion layer
4. Modify UI to accept new inputs

## Deployment

### Local Deployment

```bash
# Run with Docker (if Dockerfile created)
docker build -t quantum-healthcare .
docker run -p 8501:8501 quantum-healthcare

# Or run directly
streamlit run app.py --server.port 8501
```

### Cloud Deployment

- **Streamlit Cloud**: Deploy directly from GitHub
- **Hugging Face Spaces**: Use Gradio interface
- **AWS/GCP**: Deploy with container services

## Research & Development

### Quantum Advantages

- **Quantum Superposition**: Parallel processing of medical features
- **Quantum Entanglement**: Complex feature interactions
- **Quantum Interference**: Novel fusion mechanisms
- **Quantum Parallelism**: Efficient high-dimensional processing

### Future Directions

- **Real Quantum Hardware**: IBM Quantum, Google Quantum AI
- **Advanced Architectures**: Quantum transformers, QNNs
- **More Modalities**: Genomics, laboratory results, ECG
- **Federated Learning**: Privacy-preserving quantum ML

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Memory Issues**: Reduce batch size or model size
3. **Slow Training**: Use GPU if available
4. **Model Loading**: Check model file paths

### Performance Issues

- **CPU Training**: Expected for quantum simulation
- **Memory Usage**: Monitor RAM with large models
- **Convergence**: Adjust learning rates and epochs

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MIMIC-IV Dataset**: PhysioNet Critical Care Database
- **Qiskit**: IBM Quantum computing framework
- **PennyLane**: Xanadu quantum ML library
- **Transformers**: Hugging Face NLP library

## Citations

```bibtex
@misc{quantum-healthcare-2024,
  title={Quantum Healthcare: Multimodal Disease Prediction with MIMIC-IV Integration},
  author={Quantum Healthcare Team},
  year={2024},
  url={https://github.com/Rupali-0806/Quantum_project}
}

@article{johnson2023mimic,
  title={MIMIC-IV, a freely accessible electronic health record dataset},
  author={Johnson, Alistair EW and Bulgarelli, Lucas and Shen, Lu and Gayles, Alvin and Shammout, Ayad and Horng, Steven and Pollard, Tom J and Hao, Sicheng and Moody, Benjamin and Gow, Brian and others},
  journal={Scientific data},
  volume={10},
  number={1},
  pages={1},
  year={2023},
  publisher={Nature Publishing Group}
}
```

## Contact

For questions, issues, or collaboration:
- GitHub Issues: [Report Issues](https://github.com/Rupali-0806/Quantum_project/issues)
- Email: contact@quantumhealthcare.ai

---

⚕️ **Medical Disclaimer**: This is a research prototype for educational and research purposes. Always consult qualified healthcare professionals for medical decisions.