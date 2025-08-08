#!/usr/bin/env python3
"""
NewsBot 2.0 - One-Click Launch Script
Unified launcher for the complete NewsBot Intelligence System 2.0

This is the MAIN entry point that starts the unified app.py application
which contains all integrated modules (A, B, C, D) in a single system.
"""

import os
import sys
import subprocess
import webbrowser
import time
import socket
from pathlib import Path

def check_port_available(port):
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result != 0  # Return True if port is available
    except:
        return True  # Assume available if we can't check

def find_available_port():
    """Find an available port from common alternatives"""
    ports_to_try = [8080, 3000, 8000, 9000, 8888, 7000, 8081, 3001]
    
    for port in ports_to_try:
        if check_port_available(port):
            return port
    
    # If none of the common ports work, try a random high port
    for port in range(8082, 8100):
        if check_port_available(port):
            return port
    
    return 8080  # Default fallback

def main():
    """One-click launch for NewsBot 2.0"""
    
    print("🚀 NewsBot 2.0 - One-Click Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found. Please run this from the NewsBot directory.")
        sys.exit(1)
    
    # Check if data exists
    if not os.path.exists('data/processed/newsbot_dataset.csv'):
        print("❌ Error: Dataset not found. Please ensure data is available.")
        sys.exit(1)
    
    print("✅ NewsBot directory found")
    print("✅ Dataset found")
    
    # Check if models are trained, if not train them
    required_models = [
        'data/models/best_classifier.pkl',
        'data/models/topic_model.pkl'
    ]
    
    missing_models = [model for model in required_models if not os.path.exists(model)]
    
    if missing_models:
        print(f"🔄 Training NewsBot models... (missing: {', '.join([os.path.basename(m) for m in missing_models])})")
        print("⏱️  This may take a few minutes on first run...")
        try:
            python_path = '/Users/martin.demel/myenv3.10/bin/python'
            result = subprocess.run([python_path, 'train_models.py'], 
                                  capture_output=True, text=True, check=True)
            print("✅ Models trained successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error training models: {e}")
            print("Error output:", e.stderr)
            print("💡 Please check dependencies and try again")
            sys.exit(1)
    else:
        print("✅ All trained models found - skipping training")
    
    # Find an available port
    port = find_available_port()
    print(f"🔌 Using port: {port}")
    
    # Try to start the Flask application
    try:
        print("\n🌟 Starting NewsBot 2.0 Web Application...")
        print("⏳ Initializing ML models... (this may take 30-60 seconds)")
        print("📱 Browser will open automatically once ready...")
        print(f"🔗 URL: http://localhost:{port}")
        print("\n" + "=" * 50)
        print("📋 Available Features:")
        print(f"   • Dashboard: http://localhost:{port}")
        print(f"   • Article Analysis: http://localhost:{port}/analyze")
        print(f"   • Batch Processing: http://localhost:{port}/batch")
        print(f"   • Natural Language Queries: http://localhost:{port}/query")
        print(f"   • Visualizations: http://localhost:{port}/visualization")
        print(f"   • Translation: http://localhost:{port}/translate")
        print(f"   • Real-time Monitoring: http://localhost:{port}/realtime")
        print("=" * 50)
        print("\n💡 Press Ctrl+C to stop the application")
        print("-" * 50)
        
        # Wait for app to fully start then open browser
        def open_browser():
            print("🔄 Waiting for application to start...")
            time.sleep(45)  # Give ML models time to load
            try:
                webbrowser.open(f'http://localhost:{port}')
                print("🌐 Browser opened automatically")
            except:
                print("⚠️  Could not open browser automatically")
                print(f"   Please manually open: http://localhost:{port}")
        
        # Start browser opening in background
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start the Flask application with custom port
        env = os.environ.copy()
        env['PORT'] = str(port)
        # Use the correct Python executable from myenv3.10
        python_path = '/Users/martin.demel/myenv3.10/bin/python'
        subprocess.run([python_path, 'app.py'], env=env, check=True)
        
    except KeyboardInterrupt:
        print("\n\n🛑 NewsBot application stopped by user")
        print("👋 Thank you for using NewsBot 2.0!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting application: {e}")
        print("💡 Try running manually: python3 app.py")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Try running manually: python3 app.py")

if __name__ == "__main__":
    main()