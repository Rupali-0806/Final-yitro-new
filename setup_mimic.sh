#!/bin/bash
# MIMIC-IV Environment Setup Script

echo "🏥 Quantum Healthcare - MIMIC-IV Setup"
echo "======================================"

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 not found. Please install Python 3.8+"; exit 1; }

# Create virtual environment (optional but recommended)
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || { echo "Warning: Could not activate virtual environment"; }

# Install basic dependencies
echo "Installing basic dependencies..."
pip install --upgrade pip
pip install pandas numpy PyYAML scikit-learn

echo "✓ Basic dependencies installed"

# Test MIMIC integration
echo "Testing MIMIC-IV integration..."
python test_mimic_minimal.py

if [ $? -eq 0 ]; then
    echo "✓ MIMIC-IV integration test passed!"
else
    echo "✗ MIMIC-IV integration test failed"
    exit 1
fi

# Check for MIMIC-IV data
echo ""
echo "Checking for MIMIC-IV data..."
if [ -d "data/mimic-iv" ]; then
    echo "✓ MIMIC-IV data directory found"
    
    # Check for key files
    if [ -f "data/mimic-iv/hosp/patients.csv.gz" ]; then
        echo "✓ Patient data found"
    else
        echo "ℹ Patient data not found"
    fi
    
    if [ -f "data/mimic-iv/hosp/admissions.csv.gz" ]; then
        echo "✓ Admissions data found"
    else
        echo "ℹ Admissions data not found"
    fi
    
    if [ -f "data/mimic-iv/icu/icustays.csv.gz" ]; then
        echo "✓ ICU stays data found"
    else
        echo "ℹ ICU stays data not found"
    fi
else
    echo "ℹ MIMIC-IV data directory not found (will use synthetic data)"
fi

echo ""
echo "📋 Setup Summary:"
echo "=================="
echo "✓ Python environment ready"
echo "✓ Basic dependencies installed"
echo "✓ MIMIC-IV integration tested"

if [ -d "data/mimic-iv" ] && [ -f "data/mimic-iv/hosp/patients.csv.gz" ]; then
    echo "✓ Real MIMIC-IV data available"
    echo ""
    echo "🎉 Ready to use with real MIMIC-IV data!"
    echo "Run: python src/data/data_loader.py"
else
    echo "ℹ Using synthetic data fallback"
    echo ""
    echo "🎉 Ready to use with synthetic data!"
    echo "Run: python src/data/data_loader.py"
    echo ""
    echo "📥 To use real MIMIC-IV data:"
    echo "1. Register at https://physionet.org/"
    echo "2. Apply for MIMIC-IV access"
    echo "3. Download data to data/mimic-iv/"
    echo "4. See MIMIC_INTEGRATION.md for details"
fi

echo ""
echo "📚 Next Steps:"
echo "- Read MIMIC_INTEGRATION.md for detailed setup"
echo "- Run python test_mimic_integration.py for full test"
echo "- Run python demo.py for quick demonstration"
echo "- Run streamlit run app.py for web interface"