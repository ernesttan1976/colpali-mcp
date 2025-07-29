"""
CUDA verification script - run this to check if CUDA is properly configured
"""

import torch
import os

print("=== CUDA Environment Verification ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i} name: {torch.cuda.get_device_name(i)}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
else:
    print("CUDA is not available. GPU acceleration will not be used.")

print("\n=== Environment Variables ===")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'Not set')}")

# Try a simple CUDA operation
if torch.cuda.is_available():
    print("\n=== Testing CUDA Operation ===")
    try:
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print("CUDA operation successful!")
    except Exception as e:
        print(f"CUDA operation failed: {e}")