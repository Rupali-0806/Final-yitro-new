# Quantum Healthcare: Multimodal Disease Prediction

A comprehensive quantum machine learning pipeline for disease prediction using multiple data modalities including text, tabular, and image data from MIMIC-IV dataset.

## Features

- **Quantum NLP Model**: Quantum-enhanced natural language processing for clinical notes
- **Quantum Tabular Model**: Quantum feature encoding for structured medical data
- **Hybrid CNN-Quantum Model**: Classical-quantum hybrid model for medical imaging
- **Multimodal Fusion**: Advanced fusion layer combining all modalities
- **Interactive UI**: Web-based chatbot interface for disease prediction
- **MIMIC-IV Integration**: Comprehensive data loading and preprocessing

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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rupali-0806/Quantum_project.git
cd Quantum_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up MIMIC-IV access (requires PhysioNet credentials)

## Usage

### Training Models
```bash
python src/training/trainer.py --modality all
```

### Running the UI
```bash
streamlit run src/ui/streamlit_app.py
```

## License

This project is licensed under the MIT License.