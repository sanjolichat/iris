#!/bin/bash
# Iris setup + run script
# Usage: bash run.sh

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   IRIS — Environmental Scanner       ║"
echo "║   FutureHacks 8                      ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install from python.org"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install eyetrax opencv-python numpy requests websockets --quiet

echo ""
echo "Starting Iris..."
echo ""

# Run with API key from env or prompt
python3 iris.py
