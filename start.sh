#!/bin/bash
# NewsBot 2.0 - Simple Shell Launcher

echo "🚀 NewsBot 2.0 - Quick Start"
echo "================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Please run from NewsBot directory."
    exit 1
fi

echo "✅ Starting NewsBot 2.0..."
echo "🌐 Web interface will be available at: http://localhost:[auto-detected]"
echo "💡 Press Ctrl+C to stop"
echo "--------------------------------"

# Start the application
python3 start_newsbot.py