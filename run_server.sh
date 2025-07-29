#!/bin/bash

cd /Volumes/myssd/colpali-mcp
source venv_apple_silicon/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
python colpali_streaming_server.py