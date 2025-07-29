"""
Simple test script for Apple Silicon ColPali
Tests basic functionality without full application overhead
"""

import os
import sys
import gc
import psutil
import torch
from pathlib import Path

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)

def test_environment():
    """Test Apple Silicon environment"""
    print("🍎 Testing Apple Silicon Environment")
    print(f"📊 Initial memory: {get_memory_usage():.2f}GB")
    print(f"🖥️  System: {os.uname().machine}")
    print(f"🐍 Python: {sys.version}")
    
    # Test PyTorch
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🎯 MPS available: {torch.backends.mps.is_available()}")
    
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Apple Silicon GPU ready")
    else:
        device = "cpu"
        print("⚠️  Using CPU (MPS not available)")
    
    return device

def test_basic_model_loading():
    """Test basic model loading"""
    print("\n🔄 Testing Model Loading...")
    
    try:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        
        # Test processor first (lighter)
        print("📥 Loading processor...")
        processor = ColQwen2Processor.from_pretrained(
            "vidore/colqwen2-v1.0", 
            trust_remote_code=True
        )
        print("✅ Processor loaded successfully")
        print(f"📊 Memory after processor: {get_memory_usage():.2f}GB")
        
        # Test model loading
        print("📥 Loading model...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        model.eval()
        print("✅ Model loaded successfully")
        print(f"📊 Memory after model: {get_memory_usage():.2f}GB")
        
        return True, model, processor
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False, None, None

def test_pdf_processing():
    """Test PDF processing"""
    print("\n📄 Testing PDF Processing...")
    
    test_pdf = "/Users/ernest/Documents/Scribd/282739699-Flight-Training-Manual-A330-pdf.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found: {test_pdf}")
        return False
    
    try:
        from pdf2image import convert_from_path
        
        print(f"🔄 Converting first 3 pages of {os.path.basename(test_pdf)}...")
        
        # Convert only first 3 pages for testing
        images = convert_from_path(
            test_pdf, 
            first_page=1, 
            last_page=3,
            thread_count=2
        )
        
        print(f"✅ Converted {len(images)} pages successfully")
        print(f"📊 Image sizes: {[img.size for img in images]}")
        print(f"📊 Memory after conversion: {get_memory_usage():.2f}GB")
        
        return True, images
        
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return False, None

def test_embedding_generation(model, processor, images):
    """Test embedding generation"""
    print("\n⚙️ Testing Embedding Generation...")
    
    if not model or not processor or not images:
        print("❌ Missing required components")
        return False
    
    try:
        device = next(model.parameters()).device
        print(f"🎯 Model device: {device}")
        
        # Process just the first image
        test_image = images[0]
        print(f"🖼️ Processing image size: {test_image.size}")
        
        with torch.no_grad():
            # Process image
            batch_doc = processor.process_images([test_image]).to(device)
            print(f"📊 Batch input shape: {batch_doc['pixel_values'].shape}")
            
            # Generate embedding
            batch_embeddings = model(**batch_doc)
            print(f"📊 Batch embedding shape: {batch_embeddings.shape}")
            print(f"📊 Batch embedding dtype: {batch_embeddings.dtype}")
            
            # Unbind to get individual embeddings (like in the main app)
            embedding_list = list(torch.unbind(batch_embeddings.to("cpu")))
            embedding = embedding_list[0]  # Get the first (and only) embedding
            
            print(f"📊 Final embedding shape: {embedding.shape}")
            print(f"✅ Embedding generated successfully")
            print(f"📊 Memory after embedding: {get_memory_usage():.2f}GB")
            
        return True, embedding
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_query_processing(model, processor, embedding):
    """Test query processing"""
    print("\n🔍 Testing Query Processing...")
    
    if not model or not processor or embedding is None:
        print("❌ Missing required components")
        return False
    
    try:
        device = next(model.parameters()).device
        test_query = "flight controls and systems"
        
        print(f"🔄 Processing query: '{test_query}'")
        
        with torch.no_grad():
            # Process query
            batch_query = processor.process_queries([test_query]).to(device)
            query_embedding = model(**batch_query)
            query_embedding_cpu = query_embedding.to("cpu")
            
            print(f"📊 Query embedding shape: {query_embedding_cpu.shape}")
            print(f"📊 Document embedding shape: {embedding.shape}")
            
            # Fix the tensor dimensions for scoring
            # Both query and document embeddings need to be unbinded
            query_list = list(torch.unbind(query_embedding_cpu))
            doc_list = [embedding]  # Document embedding is already 2D
            
            print(f"📊 Query list length: {len(query_list)}")
            print(f"📊 Query item shape: {query_list[0].shape}")
            print(f"📊 Doc list length: {len(doc_list)}")
            print(f"📊 Doc item shape: {doc_list[0].shape}")
            
            # Test scoring with proper format
            score = processor.score(query_list, doc_list, device="cpu")
            print(f"📊 Similarity score: {score[0].item():.4f}")
            
            print("✅ Query processing successful")
            
        return True
        
    except Exception as e:
        print(f"❌ Query processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 Apple Silicon ColPali Test Suite")
    print("=" * 50)
    
    # Test environment
    device = test_environment()
    
    # Test memory constraint
    if get_memory_usage() > 8.0:
        print("⚠️  Warning: Already using > 8GB memory")
    
    # Test model loading
    success, model, processor = test_basic_model_loading()
    if not success:
        print("💥 Model loading failed - stopping tests")
        return False
    
    # Test PDF processing
    success, images = test_pdf_processing()
    if not success:
        print("💥 PDF processing failed - stopping tests")
        return False
    
    # Test embedding generation
    success, embedding = test_embedding_generation(model, processor, images)
    if not success:
        print("💥 Embedding generation failed - stopping tests")
        return False
    
    # Test query processing
    success = test_query_processing(model, processor, embedding)
    if not success:
        print("💥 Query processing failed - stopping tests")
        return False
    
    # Final memory check
    final_memory = get_memory_usage()
    print(f"\n📊 Final memory usage: {final_memory:.2f}GB")
    
    if final_memory > 8.0:
        print("⚠️  Warning: Exceeded 8GB memory limit")
    else:
        print("✅ Memory usage within 8GB limit")
    
    print("\n🎉 All tests passed successfully!")
    print("🚀 Apple Silicon ColPali is ready!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
