"""
Launch script for Gradio web interface.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src.ui.gradio_interface import create_gradio_interface

def main():
    """Launch Gradio interface."""
    print("🚀 Launching Quantum Healthcare AI - Gradio Interface")
    
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()