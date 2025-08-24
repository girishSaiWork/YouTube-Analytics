#!/usr/bin/env python3
"""
Installation script for YouTube Analytics project
This script will install the project in development mode
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Main installation process"""
    print("🚀 YouTube Analytics Project Installation")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "setup.py").exists():
        print("❌ Error: setup.py not found. Please run this script from the project root directory.")
        return False
    
    print(f"📁 Installing from: {current_dir}")
    
    # Install in development mode
    success = run_command(
        f"{sys.executable} -m pip install -e .",
        "Installing project in development mode"
    )
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 INSTALLATION COMPLETE!")
        print("=" * 50)
        print("✅ Project installed in development mode")
        print("✅ You can now import modules from anywhere:")
        print("   from config.settings import Config")
        print("   from src.utils.spark_utils import SparkUtils")
        print("\n📝 Next steps:")
        print("1. Copy .env.example to .env and configure your settings")
        print("2. Test the installation by running: python -c 'from config.settings import Config; print(Config.COUNTRIES)'")
        print("3. Open notebooks/01_data_exploration.ipynb to start exploring!")
        
        return True
    else:
        print("\n❌ Installation failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
