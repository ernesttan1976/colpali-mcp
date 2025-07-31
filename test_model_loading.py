#!/usr/bin/env python3
"""
Simple ColPali model test to debug the loading issue
"""

import torch
from transformers import AutoProcessor, AutoModel

def test_colpali_loading():
    print("🧪 Testing ColPali model loading...")
    
    # Test device detection
    print("\n1️⃣ Testing device detection:")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA available: {device}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps") 
        print(f"✅ MPS available: {device}")
    else:
        device = torch.device("cpu")
        print(f"✅ Using CPU: {device}")
    
    # Test model loading with transformers
    print("\n2️⃣ Testing model loading with transformers:")
    try:
        model_name = "vidore/colpali-v1.2"
        print(f"Loading processor from {model_name}...")
        processor = AutoProcessor.from_pretrained(model_name)
        print("✅ Processor loaded successfully")
        
        print(f"Loading model from {model_name}...")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            trust_remote_code=True
        )
        print("✅ Model loaded successfully")
        
        print(f"Moving model to device: {device}")
        model.to(device)
        model.eval()
        print("✅ Model moved to device and set to eval mode")
        
        print("\n🎉 ColPali model loading test successful!")
        print(f"Model device: {next(model.parameters()).device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with colpali-engine (if available)
    print("\n3️⃣ Testing with colpali-engine:")
    try:
        from colpali_engine.models import ColPali, ColPaliProcessor
        
        print("Loading with ColPali engine...")
        processor = ColPaliProcessor.from_pretrained(model_name)
        model = ColPali.from_pretrained(model_name)
        model.to(device)
        model.eval()
        print("✅ ColPali engine loading successful!")
        
    except ImportError:
        print("⚠️  colpali-engine not available, using transformers")
    except Exception as e:
        print(f"❌ ColPali engine loading failed: {e}")

if __name__ == "__main__":
    test_colpali_loading()
