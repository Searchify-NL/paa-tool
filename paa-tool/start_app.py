#!/usr/bin/env python3
"""
Startup script for PAA Tool with proper process management and resource cleanup
"""

import subprocess
import sys
import os
import signal
import time
import psutil

def kill_existing_processes():
    """Kill any existing streamlit processes."""
    print("🔍 Checking for existing processes...")
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if 'streamlit' in cmdline and 'app.py' in cmdline:
                print(f"🔄 Killing existing process: {proc.info['pid']}")
                proc.terminate()
                proc.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass
    
    # Wait a moment for processes to fully terminate
    time.sleep(2)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit', 'pandas', 'requests', 'sqlalchemy', 
        'sentence_transformers', 'torch', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def start_app():
    """Start the Streamlit app with proper configuration."""
    print("🚀 Starting PAA Tool...")
    print("📊 App will be available at: http://localhost:8501")
    print("💾 Database will be created at: data/serp.db")
    print("\n" + "="*50)
    
    try:
        # Set environment variables for better memory management
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '200'
        env['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'true'
        
        # Start streamlit with optimized settings
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.maxUploadSize", "200",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ], env=env, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")
        return False
    
    return True

def main():
    """Main startup function."""
    print("🔧 PAA Tool Startup Script")
    print("="*50)
    
    # Kill existing processes
    kill_existing_processes()
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Start the app
    return start_app()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 