#!/usr/bin/env python3
"""
NewsBot 2.0 Production Startup Script
Ensures proper environment setup and system initialization
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / 'src'))

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = current_dir / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Environment variables loaded from {env_path}")
    else:
        print(f"⚠️ No .env file found at {env_path}")
except ImportError:
    print("⚠️ python-dotenv not installed. Using system environment variables.")

# Verify critical environment variables
openai_key = os.getenv('OPENAI_API_KEY')
if not openai_key:
    print("❌ OPENAI_API_KEY not found in environment variables!")
    print("Please set your OpenAI API key in the .env file or environment.")
    sys.exit(1)
else:
    print(f"✅ OpenAI API key found: {openai_key[:10]}...")

# Verify OpenAI library
try:
    import openai
    from openai import OpenAI
    print(f"✅ OpenAI library available (version {openai.__version__})")
    
    # Test client creation
    client = OpenAI(api_key=openai_key)
    print("✅ OpenAI client created successfully")
except ImportError:
    print("❌ OpenAI library not installed. Please run: pip install openai")
    sys.exit(1)
except Exception as e:
    print(f"❌ Failed to create OpenAI client: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)

print("🚀 Starting NewsBot 2.0 Production System...")

# Import and run the Flask app
try:
    from app import app, newsbot_system
    
    # Verify system initialization
    if not newsbot_system or not newsbot_system.is_initialized:
        print("❌ NewsBot system not properly initialized!")
        sys.exit(1)
    
    print("✅ NewsBot system initialized successfully")
    print(f"✅ Articles loaded: {len(newsbot_system.article_database) if newsbot_system.article_database is not None else 0}")
    print(f"✅ OpenAI handler available: {newsbot_system.openai_chat_handler and newsbot_system.openai_chat_handler.is_available}")
    
    # Start the Flask app
    print("🌐 Starting web interface...")
    app.run(host='0.0.0.0', port=5000, debug=False)
    
except Exception as e:
    print(f"❌ Failed to start NewsBot system: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
