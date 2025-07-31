#!/usr/bin/env python3
"""
Quick Device Test for ColPali Apple Silicon Fix
"""

import torch

def test_device_detection():
    print("🧪 Testing device detection and handling...")
    
    # Test Apple Silicon detection
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_str = "mps"
        print("✅ Apple Silicon MPS detected")
    elif torch.cuda.is_available():
        device_str = "cuda"
        print("✅ CUDA detected")
    else:
        device_str = "cpu"
        print("⚠️  Using CPU")
    
    # Test device object conversion
    device_obj = torch.device(device_str)
    print(f"📱 Device string: '{device_str}' -> Device object: {device_obj}")
    
    # Test tensor operations
    try:
        test_tensor = torch.randn(2, 3)
        test_tensor = test_tensor.to(device_obj)
        print(f"✅ Tensor successfully moved to {device_obj}")
        print(f"📊 Tensor device: {test_tensor.device}")
        
        # Test basic operations
        result = test_tensor * 2
        print(f"✅ Basic operations work on {device_obj}")
        
        # Move back to CPU
        cpu_result = result.cpu()
        print(f"✅ Successfully moved tensor back to CPU")
        
    except Exception as e:
        print(f"❌ Device test failed: {e}")
        return False
    
    print("🎉 Device handling test passed!")
    return True

if __name__ == "__main__":
    test_device_detection()
