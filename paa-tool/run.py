#!/usr/bin/env python3
"""
SEO SERP Visibility Tool - Local Development Runner
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app locally."""
    print("🚀 Starting SEO SERP Visibility Tool...")
    print("📊 App will be available at: http://localhost:8501")
    print("🔐 Using secrets from .streamlit/secrets.toml")
    print("💾 Database will be created at: data/serp.db")
    print("\n" + "="*50)
    
    # Check if requirements are installed
    try:
        import streamlit
        import pandas
        import requests
        import sqlalchemy
        import sentence_transformers
        print("✅ All dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")
        return

if __name__ == "__main__":
    main() 