"""
Database module for storing and retrieving document embeddings using LanceDB.
"""

import os
import torch
import lancedb
import numpy as np
import pyarrow as pa
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import hashlib


class DocumentEmbeddingDatabase:
    """
    A class to handle storage and retrieval of document embeddings using LanceDB.
    """
    
    def __init__(self, db_path: str = "./data/embeddings_db"):
        """Initialize the database connection."""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = lancedb.connect(db_path)
        print(f"Connected to LanceDB at {db_path}")
        
    def get_table_name_for_file(self, file_path: str) -> str:
        """
        Generate a consistent table name for a given file path.
        Only uses the filename, ignoring the path.
        """
        file_name = os.path.basename(file_path)
        file_name = Path(file_name).stem
        table_name = ''.join(c if c.isalnum() else '_' for c in file_name)
        return f"doc_{table_name}"
        
    def embeddings_exist(self, file_path: str) -> bool:
        """Check if embeddings for a file already exist in the database."""
        table_name = self.get_table_name_for_file(file_path)
        try:
            # Check if the table exists in LanceDB
            exists = table_name in self.db.table_names()
            
            # Also check the direct storage method
            doc_dir = os.path.join(self.db_path, "embeddings", table_name)
            exists_direct = os.path.exists(doc_dir)
            
            if exists or exists_direct:
                file_name = os.path.basename(file_path)
                print(f"Found existing embeddings for {file_name}")
                return True
            return False
        except Exception as e:
            print(f"Error checking if embeddings exist: {e}")
            return False
    
    def save_embeddings_direct(self, file_path: str, embeddings: List[torch.Tensor], page_count: int) -> bool:
        """Save embeddings using direct file writing to bypass LanceDB and PyArrow issues."""
        try:
            # Only use the filename for consistent table naming
            table_name = self.get_table_name_for_file(file_path)
            file_name = os.path.basename(file_path)
            
            print(f"Saving {len(embeddings)} embeddings for {file_name} using direct method")
            
            # Create a directory for this document
            doc_dir = os.path.join(self.db_path, "embeddings", table_name)
            os.makedirs(doc_dir, exist_ok=True)
            
            # Save metadata - store both filename and original path
            embedding_dim = embeddings[0].shape[0] if torch.is_tensor(embeddings[0]) else len(embeddings[0])
            metadata = {
                "filename": file_name,
                "original_path": file_path,
                "page_count": page_count,
                "embedding_dimension": embedding_dim,
                "dtype": str(embeddings[0].dtype) if torch.is_tensor(embeddings[0]) else "unknown"
            }
            
            import json
            with open(os.path.join(doc_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f)
                
            # Save each embedding as a separate file
            for i, embedding in enumerate(embeddings):
                if torch.is_tensor(embedding):
                    # Convert tensor to float32 numpy array
                    if embedding.dtype in [torch.bfloat16, torch.float16]:
                        embedding = embedding.to(torch.float32)
                    embedding_np = embedding.cpu().detach().numpy().astype(np.float32)
                else:
                    embedding_np = np.array(embedding, dtype=np.float32)
                
                # Save as numpy file
                np.save(os.path.join(doc_dir, f"embedding_{i:04d}.npy"), embedding_np)
            
            print(f"Successfully saved {len(embeddings)} embeddings for {file_name}")
            return True
        except Exception as e:
            print(f"Error in direct save: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_embeddings(self, file_path: str, embeddings: List[torch.Tensor], page_count: int) -> bool:
        """Save document embeddings to the database."""
        # Use direct file saving as a more reliable method
        return self.save_embeddings_direct(file_path, embeddings, page_count)
        
    def load_embeddings_direct(self, file_path: str) -> Optional[List[torch.Tensor]]:
        """Load embeddings using direct file reading."""
        try:
            table_name = self.get_table_name_for_file(file_path)
            file_name = os.path.basename(file_path)
            doc_dir = os.path.join(self.db_path, "embeddings", table_name)
            
            if not os.path.exists(doc_dir):
                return None
                
            print(f"Loading embeddings for {file_name} using direct method")
            
            # Load metadata
            import json
            try:
                with open(os.path.join(doc_dir, "metadata.json"), 'r') as f:
                    metadata = json.load(f)
                    print(f"Loaded metadata: {metadata['filename']} with {metadata.get('page_count', 0)} pages")
            except:
                metadata = {"dtype": "torch.float32"}
            
            # Determine target dtype
            original_dtype_str = metadata.get("dtype", "torch.float32")
            if "bfloat16" in original_dtype_str:
                target_dtype = torch.bfloat16
            elif "float16" in original_dtype_str:
                target_dtype = torch.float16
            else:
                target_dtype = torch.float32
            
            # Find all embedding files
            import glob
            embedding_files = sorted(glob.glob(os.path.join(doc_dir, "embedding_*.npy")))
            
            if not embedding_files:
                return None
                
            # Load each embedding
            embeddings = []
            for emb_file in embedding_files:
                embedding_np = np.load(emb_file)
                # Create as float32 then convert if needed
                embedding = torch.tensor(embedding_np, dtype=torch.float32)
                if target_dtype != torch.float32:
                    embedding = embedding.to(target_dtype)
                embeddings.append(embedding)
            
            print(f"Successfully loaded {len(embeddings)} embeddings for {file_name}")
            return embeddings
        except Exception as e:
            print(f"Error in direct load: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_embeddings(self, file_path: str) -> Optional[List[torch.Tensor]]:
        """Load document embeddings from the database."""
        # Use direct file loading as a more reliable method
        return self.load_embeddings_direct(file_path)
    
    def delete_embeddings(self, file_path: str) -> bool:
        """Delete embeddings for a file from the database."""
        try:
            table_name = self.get_table_name_for_file(file_path)
            doc_dir = os.path.join(self.db_path, "embeddings", table_name)
            
            if os.path.exists(doc_dir):
                import shutil
                shutil.rmtree(doc_dir)
                print(f"Deleted embeddings for {os.path.basename(file_path)}")
                return True
                
            # Also try to delete from LanceDB if it exists
            if table_name in self.db.table_names():
                self.db.drop_table(table_name)
                return True
                
            return False
        except Exception as e:
            print(f"Error deleting embeddings: {e}")
            return False
            
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents with embeddings in the database."""
        try:
            documents = []
            
            # Check direct storage
            embeddings_dir = os.path.join(self.db_path, "embeddings")
            if os.path.exists(embeddings_dir):
                import os
                for table_name in os.listdir(embeddings_dir):
                    doc_dir = os.path.join(embeddings_dir, table_name)
                    if os.path.isdir(doc_dir):
                        # Load metadata
                        import json
                        try:
                            with open(os.path.join(doc_dir, "metadata.json"), 'r') as f:
                                metadata = json.load(f)
                                metadata["table_name"] = table_name
                                documents.append(metadata)
                        except:
                            # If metadata file is missing, create a basic entry
                            documents.append({
                                "filename": table_name.replace("doc_", ""),
                                "table_name": table_name,
                                "page_count": 0,
                                "embedding_dimension": 0
                            })
            
            return documents
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []