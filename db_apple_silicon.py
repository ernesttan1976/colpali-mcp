"""
Apple Silicon optimized database module for ColPali
Designed for memory efficiency with 8GB constraint
"""

import os
import torch
import numpy as np
import json
import psutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib
import gc

class AppleSiliconEmbeddingDB:
    """
    Memory-optimized embedding database for Apple Silicon
    Uses direct file storage to avoid PyArrow/LanceDB memory overhead
    """
    
    def __init__(self, db_path: str = "./data/embeddings_db", max_memory_gb: float = 8.0):
        """Initialize the database with memory constraints"""
        self.db_path = Path(db_path)
        self.max_memory_gb = max_memory_gb
        self.embeddings_dir = self.db_path / "embeddings"
        
        # Create directories
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)
        
        print(f"🗄️ Apple Silicon DB initialized at {db_path}")
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
        
    def _check_memory_limit(self, buffer_gb: float = 1.0) -> bool:
        """Check if we have enough memory available"""
        current = self._get_memory_usage()
        available = self.max_memory_gb - buffer_gb
        return current < available
        
    def _get_doc_id(self, filename: str) -> str:
        """Generate a consistent document ID from filename"""
        # Use just the filename without path or extension
        base_name = Path(filename).stem
        # Create a clean identifier
        doc_id = ''.join(c if c.isalnum() else '_' for c in base_name)
        return f"doc_{doc_id}"
        
    def _get_doc_dir(self, filename: str) -> Path:
        """Get the directory path for a document"""
        doc_id = self._get_doc_id(filename)
        return self.embeddings_dir / doc_id
        
    def embeddings_exist(self, filename: str) -> bool:
        """Check if embeddings exist for a file"""
        doc_dir = self._get_doc_dir(filename)
        metadata_file = doc_dir / "metadata.json"
        
        exists = metadata_file.exists()
        if exists:
            print(f"📋 Found existing embeddings for {Path(filename).name}")
        return exists
        
    def save_embeddings(self, filename: str, embeddings: List[torch.Tensor], page_count: int) -> bool:
        """Save embeddings with memory optimization"""
        try:
            if not self._check_memory_limit(2.0):  # Need 2GB buffer for saving
                print("⚠️  Insufficient memory to save embeddings")
                return False
                
            doc_dir = self._get_doc_dir(filename)
            doc_dir.mkdir(exist_ok=True)
            
            print(f"💾 Saving {len(embeddings)} embeddings for {Path(filename).name}")
            
            # Save metadata
            metadata = {
                "filename": Path(filename).name,
                "original_path": str(filename),
                "page_count": page_count,
                "embedding_count": len(embeddings),
                "embedding_dimension": embeddings[0].shape[0] if embeddings else 0,
                "dtype": str(embeddings[0].dtype) if embeddings else "unknown"
            }
            
            with open(doc_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Save embeddings in chunks to manage memory
            chunk_size = 50  # Process 50 embeddings at a time
            for i in range(0, len(embeddings), chunk_size):
                chunk = embeddings[i:i + chunk_size]
                
                for j, embedding in enumerate(chunk):
                    idx = i + j
                    
                    # Convert to float32 numpy array for consistent storage
                    if torch.is_tensor(embedding):
                        if embedding.dtype in [torch.bfloat16, torch.float16]:
                            embedding = embedding.to(torch.float32)
                        embedding_np = embedding.cpu().detach().numpy().astype(np.float32)
                    else:
                        embedding_np = np.array(embedding, dtype=np.float32)
                    
                    # Save individual embedding
                    embedding_file = doc_dir / f"embedding_{idx:04d}.npy"
                    np.save(embedding_file, embedding_np)
                
                # Cleanup after each chunk
                if i % (chunk_size * 2) == 0:  # Every 100 embeddings
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        
            print(f"✅ Saved {len(embeddings)} embeddings successfully")
            print(f"📊 Memory usage: {self._get_memory_usage():.2f}GB")
            return True
            
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def load_embeddings(self, filename: str) -> Optional[List[torch.Tensor]]:
        """Load embeddings with memory optimization"""
        try:
            doc_dir = self._get_doc_dir(filename)
            
            if not doc_dir.exists():
                return None
                
            # Load metadata
            metadata_file = doc_dir / "metadata.json"
            if not metadata_file.exists():
                return None
                
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            embedding_count = metadata.get("embedding_count", 0)
            if embedding_count == 0:
                return None
                
            print(f"📥 Loading {embedding_count} embeddings for {metadata['filename']}")
            
            # Check memory before loading
            if not self._check_memory_limit(3.0):
                print("⚠️  Insufficient memory to load embeddings")
                return None
                
            # Determine target dtype
            original_dtype_str = metadata.get("dtype", "torch.float32")
            if "bfloat16" in original_dtype_str:
                target_dtype = torch.bfloat16
            elif "float16" in original_dtype_str:
                target_dtype = torch.float16
            else:
                target_dtype = torch.float32
                
            # Load embeddings in chunks to manage memory
            embeddings = []
            chunk_size = 50  # Load 50 at a time
            
            for i in range(0, embedding_count, chunk_size):
                chunk_embeddings = []
                
                for j in range(chunk_size):
                    idx = i + j
                    if idx >= embedding_count:
                        break
                        
                    embedding_file = doc_dir / f"embedding_{idx:04d}.npy"
                    if not embedding_file.exists():
                        print(f"⚠️  Missing embedding file: {embedding_file}")
                        continue
                        
                    # Load numpy array
                    embedding_np = np.load(embedding_file)
                    
                    # Convert to tensor with target dtype
                    embedding = torch.tensor(embedding_np, dtype=torch.float32)
                    if target_dtype != torch.float32:
                        embedding = embedding.to(target_dtype)
                        
                    chunk_embeddings.append(embedding)
                    
                embeddings.extend(chunk_embeddings)
                
                # Cleanup after each chunk
                if i % (chunk_size * 2) == 0:  # Every 100 embeddings
                    gc.collect()
                    
            if len(embeddings) != embedding_count:
                print(f"⚠️  Loaded {len(embeddings)} embeddings, expected {embedding_count}")
                
            print(f"✅ Loaded {len(embeddings)} embeddings successfully")
            print(f"📊 Memory usage: {self._get_memory_usage():.2f}GB")
            return embeddings
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def delete_embeddings(self, filename: str) -> bool:
        """Delete embeddings for a file"""
        try:
            doc_dir = self._get_doc_dir(filename)
            
            if doc_dir.exists():
                import shutil
                shutil.rmtree(doc_dir)
                print(f"🗑️  Deleted embeddings for {Path(filename).name}")
                return True
                
            return False
            
        except Exception as e:
            print(f"❌ Error deleting embeddings: {e}")
            return False
            
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents with embeddings"""
        try:
            documents = []
            
            if not self.embeddings_dir.exists():
                return documents
                
            for doc_dir in self.embeddings_dir.iterdir():
                if not doc_dir.is_dir():
                    continue
                    
                metadata_file = doc_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        metadata["doc_id"] = doc_dir.name
                        documents.append(metadata)
                else:
                    # Create basic entry if metadata missing
                    documents.append({
                        "filename": doc_dir.name.replace("doc_", ""),
                        "doc_id": doc_dir.name,
                        "page_count": 0,
                        "embedding_count": 0
                    })
                    
            return documents
            
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
            return []
            
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            docs = self.list_documents()
            total_embeddings = sum(doc.get("embedding_count", 0) for doc in docs)
            total_pages = sum(doc.get("page_count", 0) for doc in docs)
            
            return {
                "total_documents": len(docs),
                "total_embeddings": total_embeddings,
                "total_pages": total_pages,
                "memory_usage_gb": self._get_memory_usage(),
                "memory_limit_gb": self.max_memory_gb,
                "db_path": str(self.db_path)
            }
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {}
            
    def cleanup_cache(self):
        """Force cleanup of memory caches"""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"🧹 Cache cleaned, memory: {self._get_memory_usage():.2f}GB")
