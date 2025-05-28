#!/bin/sh
# Setup script for Tinfoil proxy

echo "Setting up Tinfoil proxy environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install tinfoil fastapi uvicorn[standard] pydantic python-multipart aiohttp

echo "Setup complete! To run the proxy:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Set your API key: export TINFOIL_API_KEY='your-key-here'"
echo "3. Run the proxy: python tinfoil_proxy.py"