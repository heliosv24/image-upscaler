#!/bin/bash
# Image Upscaler - One-command setup and launch
set -e

cd "$(dirname "$0")"

VENV_PYTHON="./venv/bin/python3"
VENV_PIP="./venv/bin/pip"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Install dependencies if needed
if [ ! -f "venv/.deps_installed" ]; then
    echo "Installing dependencies (this may take a minute on first run)..."
    $VENV_PIP install --upgrade pip -q
    $VENV_PIP install -r requirements.txt -q
    touch venv/.deps_installed
    echo "Dependencies installed."
fi

# Enable MPS fallback for operations not yet supported on Metal
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo ""
echo "╔══════════════════════════════════════╗"
echo "║       Image Upscaler (Real-ESRGAN)   ║"
echo "║   Open: http://localhost:8080        ║"
echo "╚══════════════════════════════════════╝"
echo ""

$VENV_PYTHON app.py
