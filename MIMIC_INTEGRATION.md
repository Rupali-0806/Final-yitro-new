# MIMIC-IV Dataset Integration Guide

This guide explains how to use real MIMIC-IV data from PhysioNet instead of synthetic data in the Quantum Healthcare project.

## Overview

The project now supports both:
- **Real MIMIC-IV data** from PhysioNet (preferred for research)
- **Synthetic data fallback** for demonstration and development

## Quick Start (Synthetic Data)

For immediate testing without real data:

```bash
# Clone and setup
git clone https://github.com/Rupali-0806/Quantum_project.git
cd Quantum_project

# Test the integration
python test_mimic_minimal.py

# Run with synthetic data
python src/data/data_loader.py
```

## Using Real MIMIC-IV Data

### 1. Obtain MIMIC-IV Access

1. **Register at PhysioNet**: Go to https://physionet.org/
2. **Complete CITI training**: Required for MIMIC access
3. **Apply for MIMIC-IV access**: https://physionet.org/content/mimiciv/
4. **Download credentials**: Note your username and password

### 2. Download MIMIC-IV Data

**NEW: Automatic Download Available!**

You can now automatically download MIMIC-IV data using the implemented PhysioNet downloader:

```bash
# Quick automatic download
python setup_credentials.py  # Sets up your PhysioNet credentials
python download_mimic.py     # Downloads all required MIMIC-IV files

# Test the download system
python test_download.py
```

**Manual Download (Alternative)**

```bash
# Create data directory
mkdir -p data/mimic-iv

# Option A: Manual download from PhysioNet website
# Visit https://physionet.org/content/mimiciv/2.2/
# Download required files to data/mimic-iv/

# Option B: Using wget (requires PhysioNet credentials)
# See PhysioNet documentation for wget commands
```

For detailed instructions, see: **[PHYSIONET_DOWNLOAD_GUIDE.md](PHYSIONET_DOWNLOAD_GUIDE.md)**

### 3. Required MIMIC-IV Files

Minimum required files for the integration:

```
data/mimic-iv/
├── hosp/
│   ├── patients.csv.gz      # Patient demographics
│   ├── admissions.csv.gz    # Hospital admissions
│   └── transfers.csv.gz     # Patient transfers
├── icu/
│   ├── icustays.csv.gz      # ICU stay information
│   └── chartevents.csv.gz   # Vital signs and measurements
└── note/
    ├── discharge.csv.gz     # Discharge summaries
    └── radiology.csv.gz     # Radiology reports
```

### 4. Set Environment Variables

```bash
# Set PhysioNet credentials (optional, for future API access)
export PHYSIONET_USERNAME="your_username"
export PHYSIONET_PASSWORD="your_password"
```

### 5. Configure the System

Edit `config.yaml`:

```yaml
mimic:
  data_path: "data/mimic-iv"
  sample_size: 1000  # Set to -1 for all patients
  use_synthetic_fallback: true  # Keep true for development
```

### 6. Run with Real Data

```bash
# Test data availability
python test_mimic_integration.py

# Load and preprocess real MIMIC-IV data
python src/data/data_loader.py

# Train models with real data
python train.py
```

## Configuration Options

### MIMIC-IV Settings (`config.yaml`)

```yaml
mimic:
  # Local path to MIMIC-IV data
  data_path: "data/mimic-iv"
  
  # Dataset version
  version: "2.2"
  
  # Number of patients to load (-1 for all)
  sample_size: 1000
  
  # Fall back to synthetic data if real data unavailable
  use_synthetic_fallback: true
  
  # Data modules to load
  modules:
    core:
      admissions: "hosp/admissions.csv.gz"
      patients: "hosp/patients.csv.gz"
      transfers: "hosp/transfers.csv.gz"
    icu:
      icustays: "icu/icustays.csv.gz"
      chartevents: "icu/chartevents.csv.gz"
    notes:
      discharge: "note/discharge.csv.gz"
      radiology: "note/radiology.csv.gz"
```

### Preprocessing Settings

```yaml
preprocessing:
  text:
    max_length: 512
    tokenizer: "bert-base-uncased"
  image:
    size: [224, 224]
    channels: 1
  tabular:
    normalize: true
    handle_missing: "mean"
```

## Data Flow

```
1. Check for real MIMIC-IV data
   ├── Found → Load and process real data
   └── Not found → Use synthetic fallback

2. Process each modality:
   ├── Tabular: ICU measurements, demographics
   ├── Text: Clinical notes, discharge summaries  
   └── Images: Synthetic (MIMIC-IV has no images)

3. Apply preprocessing:
   ├── Normalize tabular features
   ├── Tokenize text with BERT
   └── Prepare for quantum models

4. Create train/test splits
```

## Data Verification

Check what data is available:

```python
from src.data.mimic_downloader import MIMICDownloader

downloader = MIMICDownloader()

# Check data source
print(downloader.get_data_source())

# Verify file availability
verification = downloader.verify_real_data()
for module, files in verification.items():
    print(f"{module}:")
    for name, available in files.items():
        status = "✓" if available else "✗"
        print(f"  {name}: {status}")
```

## Troubleshooting

### Issue: "No MIMIC-IV data found"
**Solution**: Ensure files are in correct location (`data/mimic-iv/`) and properly named.

### Issue: "Permission denied" accessing files
**Solution**: Check file permissions and PhysioNet access status.

### Issue: "Memory error" loading data
**Solution**: Reduce `sample_size` in config.yaml or use a machine with more RAM.

### Issue: Missing clinical notes
**Solution**: The system will generate synthetic notes if real ones aren't available.

## Performance Considerations

### For Development
```yaml
mimic:
  sample_size: 100  # Small sample for fast iteration
  use_synthetic_fallback: true
```

### For Research
```yaml
mimic:
  sample_size: -1  # All available patients
  use_synthetic_fallback: false  # Ensure real data only
```

### For Production
```yaml
mimic:
  sample_size: 10000  # Large but manageable sample
  use_synthetic_fallback: false
```

## Data Privacy and Ethics

⚠️ **Important**: MIMIC-IV contains real patient data (de-identified)
- Only use for approved research purposes
- Follow PhysioNet data use agreements
- Never attempt to re-identify patients
- Secure all data files and results

## Integration Examples

### Load Real Data
```python
from src.data.data_loader import load_and_preprocess_mimic_data

# Load with real MIMIC-IV data
train_data, test_data, classes, preprocessor = load_and_preprocess_mimic_data(
    download_fresh=True,
    config_path="config.yaml"
)

print(f"Loaded {len(train_data['labels'])} training samples")
print(f"Disease classes: {classes}")
```

### Check Data Source
```python
from src.data.mimic_downloader import MIMICDownloader

downloader = MIMICDownloader()
print(f"Using: {downloader.get_data_source()}")

# Verify real data availability
real_data_status = downloader.verify_real_data()
if any(any(files.values()) for files in real_data_status.values()):
    print("Real MIMIC-IV data detected!")
else:
    print("Using synthetic data fallback")
```

## Contributing

To extend MIMIC-IV support:

1. **Add new data modules**: Update `config.yaml` modules section
2. **Enhance preprocessing**: Modify `_process_*_data` methods
3. **Add new data types**: Extend the processing pipeline
4. **Improve real data integration**: Add PhysioNet API support

## References

- [MIMIC-IV Documentation](https://mimic.mit.edu/docs/iv/)
- [PhysioNet](https://physionet.org/)
- [MIMIC-IV on PhysioNet](https://physionet.org/content/mimiciv/)
- [Data Use Agreement](https://physionet.org/content/mimiciv/view-dua/2.2/)