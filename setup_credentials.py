#!/usr/bin/env python3
"""
Environment Setup for MIMIC-IV Download
Sets up environment variables for PhysioNet credentials securely.
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup environment variables for PhysioNet credentials."""
    print("🔐 PhysioNet Credentials Setup")
    print("=" * 40)
    
    # Set the credentials as environment variables
    username = "uragul500@gmail.com"
    password = "Ragul@4321"
    
    # Set environment variables for current session
    os.environ['PHYSIONET_USERNAME'] = username
    os.environ['PHYSIONET_PASSWORD'] = password
    
    print(f"✅ PHYSIONET_USERNAME set to: {username}")
    print("✅ PHYSIONET_PASSWORD set (hidden for security)")
    
    # Create .env file for persistent storage (optional)
    env_file = Path(".env")
    with open(env_file, "w") as f:
        f.write(f"PHYSIONET_USERNAME={username}\n")
        f.write(f"PHYSIONET_PASSWORD={password}\n")
    
    print(f"✅ Environment variables saved to {env_file}")
    print("\nNote: The .env file contains sensitive credentials.")
    print("Make sure it's included in .gitignore to avoid committing credentials.")
    
    return username, password

if __name__ == "__main__":
    setup_environment()
    print("\nEnvironment setup complete! You can now run:")
    print("  python download_mimic.py")