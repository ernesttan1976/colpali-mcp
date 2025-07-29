#!/bin/bash
# Apple Silicon setup script for ColPali

echo "🍎 Setting up ColPali for Apple Silicon..."

# Check if we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ This script is designed for Apple Silicon (arm64) machines"
    exit 1
fi

# Check if Python 3.9+ is available
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [[ $(echo "$python_version >= 3.9" | bc -l) -ne 1 ]]; then
    echo "❌ Python 3.9+ required, found $python_version"
    exit 1
fi

echo "✅ Running on Apple Silicon with Python $python_version"

# Install poppler for PDF processing
if ! command -v pdftoppm &> /dev/null; then
    echo "📦 Installing poppler for PDF processing..."
    if command -v brew &> /dev/null; then
        brew install poppler
    else
        echo "❌ Please install Homebrew first: https://brew.sh"
        exit 1
    fi
else
    echo "✅ Poppler already installed"
fi

# Create virtual environment
if [ ! -d "venv_apple_silicon" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv_apple_silicon
fi

# Activate virtual environment
source venv_apple_silicon/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Apple Silicon optimized packages
echo "📦 Installing Apple Silicon optimized packages..."
pip install -r requirements-apple-silicon.txt

# Create necessary directories
mkdir -p ./data/embeddings_db
mkdir -p ./data/image_cache
mkdir -p ./models/colqwen2

echo "🎉 Apple Silicon setup complete!"
echo "💡 To activate the environment: source venv_apple_silicon/bin/activate"
echo "🚀 To test: python test_apple_silicon.py"
