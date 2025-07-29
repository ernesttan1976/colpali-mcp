#!/bin/bash
# Startup script for the ColPali Docker application

# Check if running on Apple Silicon
if [[ "$(uname -m)" == "arm64" && "$(uname)" == "Darwin" ]]; then
    echo "=== Apple Silicon detected ==="
    # Check if PyTorch with MPS is installed
    PYTORCH_INSTALLED=$(pip list | grep torch || echo "not installed")
    
    if [[ "$PYTORCH_INSTALLED" == "not installed" ]]; then
        echo "Installing PyTorch for Apple Silicon..."
        pip install torch torchvision torchaudio
    fi
else
    echo "=== Non-Apple Silicon platform detected ==="
    # Use the CUDA version if running on a machine with NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected, using CUDA version of PyTorch"
        pip install -r requirements.txt
    else
        echo "No NVIDIA GPU detected, using CPU version of PyTorch"
        pip install torch torchvision torchaudio
        pip install -r requirements.txt
    fi
fi

# Check environment
echo "=== System Information ==="
uname -a
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "NVIDIA drivers not found or not accessible"
fi

# Verify CUDA
echo "=== Running CUDA Verification ==="
python verify_cuda.py

# Create model directories on host if they don't exist
echo "=== Ensuring Model Directories Exist ==="
mkdir -p ./models/colqwen2/model
mkdir -p ./models/colqwen2/processor
chmod -R 777 ./models

# List current model files
echo "=== Current Model Files ==="
ls -la ./models/colqwen2/model || echo "Model directory is empty"
ls -la ./models/colqwen2/processor || echo "Processor directory is empty"

# Handle flash-attn installation
echo "=== Checking Flash Attention Requirement ==="
if grep -q "install_fa2" app.py; then
  echo "Flash Attention is used in the app, but we'll let the app handle its installation if needed"
  echo "If the app tries to install flash-attn automatically, it might take several minutes"
  echo "You can check app.py if it actually needs flash-attn or if it can be disabled"
fi

# Verify torch version
echo "=== PyTorch Configuration ==="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Create necessary directories
mkdir -p ./data/embeddings_db

# Launch application
echo "=== Starting ColPali Application ==="
python app.py