"""
Launch script for Streamlit web interface.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src.ui.streamlit_app import main

if __name__ == "__main__":
    main()