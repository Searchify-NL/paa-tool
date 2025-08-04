#!/usr/bin/env python3
"""
SEO SERP Visibility Tool - Setup Script
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = ["data", ".streamlit"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_secrets():
    """Check if secrets file exists."""
    secrets_file = Path(".streamlit/secrets.toml")
    if secrets_file.exists():
        print("✅ Secrets file found: .streamlit/secrets.toml")
        return True
    else:
        print("⚠️  Secrets file not found: .streamlit/secrets.toml")
        print("Please create it with your API keys:")
        print("""
[general]
app_password = "your_password"

[searchapi]
api_key = "your_searchapi_key"
        """)
        return False

def main():
    """Run the setup process."""
    print("🔧 Setting up SEO SERP Visibility Tool...")
    print("="*50)
    
    # Create directories
    create_directories()
    
    # Check secrets
    secrets_ok = check_secrets()
    
    # Install requirements
    requirements_ok = install_requirements()
    
    print("\n" + "="*50)
    if requirements_ok and secrets_ok:
        print("✅ Setup complete! You can now run the app:")
        print("   python run.py")
        print("   or")
        print("   streamlit run app.py")
    else:
        print("⚠️  Setup incomplete. Please check the issues above.")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 