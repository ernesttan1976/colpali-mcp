"""
Apple Silicon optimized ColPali application
Designed for 16GB RAM with max 8GB usage constraint
"""

import os
import sys
import gc
import psutil
import torch
import time
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image

# Memory monitoring
def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)

def check_memory_limit(limit_gb=8.0):
    """Check if memory usage exceeds limit"""
    current = get_memory_usage()
    if current > limit_gb:
        print(f"⚠️  Memory usage {current:.2f}GB exceeds limit {limit_gb}GB")
        return False
    return True

# Set memory optimization flags for Apple Silicon
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable high watermark
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable CPU fallback

print("🍎 ColPali Apple Silicon - Optimized for 16GB RAM")
print(f"📊 Initial memory usage: {get_memory_usage():.2f}GB")

# Core imports
from pdf2image import convert_from_path
from torch.utils.data import DataLoader
from tqdm import tqdm
from colpali_engine.models import ColQwen2, ColQwen2Processor
from db_apple_silicon import AppleSiliconEmbeddingDB
from image_cache import convert_files

# Initialize database
print("🗄️ Initializing Apple Silicon database...")
os.makedirs("./data/embeddings_db", exist_ok=True)
db = AppleSiliconEmbeddingDB(db_path="./data/embeddings_db", max_memory_gb=3.5)

class AppleSiliconColPali:
    """Apple Silicon optimized ColPali implementation"""
    
    def __init__(self, max_memory_gb=3.5):
        self.max_memory_gb = max_memory_gb
        self.model = None
        self.processor = None
        self.device = None
        self._setup_device()
        
    def _setup_device(self):
        """Setup optimal device for Apple Silicon"""
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ Apple Silicon MPS detected and enabled")
        else:
            self.device = "cpu"
            print("⚠️  MPS not available, using CPU (will be slower)")
            
        print(f"🎯 Target device: {self.device}")
        print(f"💾 Memory limit: {self.max_memory_gb}GB (STRICT)")
        
    def _optimize_model_for_memory(self):
        """Apply memory optimizations for Apple Silicon"""
        if self.model is None:
            return
            
        # Enable memory efficient attention if available
        if hasattr(self.model.config, 'use_memory_efficient_attention'):
            self.model.config.use_memory_efficient_attention = True
            
        # Use gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            
        print("🔧 Applied memory optimizations")
        
    def load_model(self):
        """Load model with Apple Silicon optimizations"""
        print("📥 Loading ColQwen2 model for Apple Silicon...")
        
        # Check memory before loading
        if not check_memory_limit(self.max_memory_gb - 2.5):  # Reserve 2.5GB for model
            print("❌ Insufficient memory to load model")
            return False
            
        try:
            # Load with aggressive memory optimizations
            print("🔄 Loading model...")
            self.model = ColQwen2.from_pretrained(
                "vidore/colqwen2-v1.0",
                torch_dtype=torch.float16,  # Use float16 for memory efficiency
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Minimize CPU memory usage
                max_memory={self.device: "2GB"},  # Strict GPU memory limit
            )
            
            # Apply memory optimizations
            self._optimize_model_for_memory()
            
            print("🔄 Loading processor...")
            self.processor = ColQwen2Processor.from_pretrained(
                "vidore/colqwen2-v1.0", 
                trust_remote_code=True
            )
            
            # Set to eval mode and move to device
            self.model.eval()
            
            print(f"✅ Model loaded successfully on {self.device}")
            print(f"📊 Memory usage after model load: {get_memory_usage():.2f}GB")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
            
    def process_pdf_batch(self, pdf_path: str, batch_size: int = 1) -> Tuple[List[torch.Tensor], List[Image.Image]]:
        """Process PDF in batches to manage memory"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
        
        # Check if embeddings exist
        filename = os.path.basename(pdf_path)
        if db.embeddings_exist(filename):
            print("📋 Loading existing embeddings...")
            existing_embeddings = db.load_embeddings(filename)
            if existing_embeddings:
                # Also need to load images for this file
                images = convert_files([pdf_path])
                return existing_embeddings, images
                
        # Convert PDF to images with caching
        print("🖼️ Converting PDF to images...")
        images = convert_files([pdf_path])
        print(f"📊 Converted {len(images)} pages")
        print(f"📊 Memory usage after conversion: {get_memory_usage():.2f}GB")
        
        # Process embeddings in batches
        embeddings = []
        total_batches = (len(images) + batch_size - 1) // batch_size
        
        print(f"⚙️ Processing {total_batches} batches (batch_size={batch_size})")
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            print(f"🔄 Processing batch {i//batch_size + 1}/{total_batches}")
            
            # Check memory before processing batch
            if not check_memory_limit(self.max_memory_gb - 0.5):  # Reserve 0.5GB buffer
                print("⚠️  Memory limit reached, reducing batch size")
                batch_size = 1  # Force single image processing
                batch_images = batch_images[:1]
            
            # Process batch
            batch_embeddings = self._process_image_batch(batch_images)
            embeddings.extend(batch_embeddings)
            
            # Force garbage collection every image for 3.5GB limit
            if i % batch_size == 0:  # After every batch
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()
                print(f"🧹 Cleaned cache, memory: {get_memory_usage():.2f}GB")
        
        # Save embeddings to database
        print("💾 Saving embeddings to database...")
        db.save_embeddings(filename, embeddings, len(images))
        
        print(f"✅ Processing complete: {len(embeddings)} embeddings generated")
        print(f"📊 Final memory usage: {get_memory_usage():.2f}GB")
        
        return embeddings, images
        
    def _process_image_batch(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """Process a batch of images to embeddings"""
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")
            
        embeddings = []
        
        # Create dataloader with small batch size for memory efficiency
        dataloader = DataLoader(
            images,
            batch_size=1,  # Process one at a time for Apple Silicon
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x).to(self.device),
        )
        
        with torch.no_grad():
            for batch_doc in dataloader:
                # Move to device
                batch_doc = {k: v.to(self.device) for k, v in batch_doc.items()}
                
                # Generate embeddings
                batch_embeddings = self.model(**batch_doc)
                
                # Move to CPU immediately to free GPU memory
                embeddings.extend(list(torch.unbind(batch_embeddings.to("cpu"))))
                
                # Clear GPU cache
                if self.device == "mps":
                    torch.mps.empty_cache()
                    
        return embeddings
        
    def search(self, query: str, embeddings: List[torch.Tensor], images: List[Image.Image], k: int = 5) -> List[Tuple[Image.Image, str]]:
        """Search for relevant images using query"""
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")
            
        print(f"🔍 Searching for: '{query}'")
        
        k = min(k, len(embeddings))
        
        # Process query
        with torch.no_grad():
            batch_query = self.processor.process_queries([query]).to(self.device)
            query_embedding = self.model(**batch_query)
            query_embedding = query_embedding.to("cpu")
            
        # Score against document embeddings
        scores = self.processor.score([query_embedding], embeddings, device="cpu")
        
        # Get top k results
        top_k_indices = scores[0].topk(k).indices.tolist()
        
        results = []
        for idx in top_k_indices:
            results.append((images[idx], f"Page {idx + 1}"))
            
        print(f"✅ Found {len(results)} relevant pages")
        return results

def test_apple_silicon_colpali():
    """Test function for Apple Silicon ColPali"""
    print("🧪 Testing Apple Silicon ColPali...")
    
    # Test PDF path
    test_pdf = "/Users/ernest/Documents/Scribd/282739699-Flight-Training-Manual-A330-pdf.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found: {test_pdf}")
        return False
        
    try:
        # Initialize ColPali with 3.5GB limit
        colpali = AppleSiliconColPali(max_memory_gb=3.5)
        
        # Load model
        if not colpali.load_model():
            print("❌ Failed to load model")
            return False
            
        # Process PDF with smaller batch size for 3.5GB limit
        embeddings, images = colpali.process_pdf_batch(test_pdf, batch_size=1)
        
        # Test search
        test_queries = [
            "flight controls",
            "emergency procedures",
            "landing configuration",
            "engine parameters"
        ]
        
        for query in test_queries:
            results = colpali.search(query, embeddings, images, k=3)
            print(f"📋 Query: '{query}' -> {len(results)} results")
            
        print("✅ Apple Silicon ColPali test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_apple_silicon_colpali()
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Tests failed!")
        sys.exit(1)
