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
import json
import glob
import shutil


class DocumentEmbeddingDatabase:
    """
    A class to handle storage and retrieval of document embeddings using LanceDB.
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the database connection."""
        # Use environment variable or default to the data directory
        if db_path is None:
            db_path = os.getenv("COLPALI_DB_PATH", "/Volumes/myssd/colpali-mcp/data")

        self.db_path = db_path

        # Ensure the database directory exists and is writable
        try:
            os.makedirs(self.db_path, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(self.db_path, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print(f"Database directory is writable: {self.db_path}")
        except (OSError, PermissionError) as e:
            raise RuntimeError(
                f"Cannot create or write to database directory {db_path}: {e}"
            )

        # Initialize LanceDB connection
        try:
            self.db = lancedb.connect(self.db_path)
            print(f"Connected to LanceDB at {self.db_path}")
        except Exception as e:
            print(f"Warning: Failed to connect to LanceDB: {e}")
            print("Falling back to file-based storage only")
            self.db = None

    def get_table_name_for_file(self, file_path: str) -> str:
        """
        Generate a consistent table name for a given file path.
        Only uses the filename, ignoring the path.
        """
        file_name = os.path.basename(file_path)
        file_name = Path(file_name).stem
        table_name = "".join(c if c.isalnum() else "_" for c in file_name)
        return f"doc_{table_name}"

    def embeddings_exist(self, file_path: str) -> bool:
        """Check if embeddings for a file already exist in the database."""
        table_name = self.get_table_name_for_file(file_path)
        file_name = os.path.basename(file_path)

        try:
            # Check if the table exists in LanceDB (if available)
            if self.db is not None:
                lancedb_exists = table_name in self.db.table_names()
            else:
                lancedb_exists = False

            # Check the direct storage method
            doc_dir = os.path.join(self.db_path, "embeddings", table_name)
            file_exists = os.path.exists(doc_dir) and os.path.exists(
                os.path.join(doc_dir, "metadata.json")
            )

            if lancedb_exists or file_exists:
                print(f"Found existing embeddings for {file_name}")
                return True
            return False
        except Exception as e:
            print(f"Error checking if embeddings exist: {e}")
            return False

    def save_embeddings_direct(
        self, file_path: str, embeddings: List[torch.Tensor], page_count: int
    ) -> bool:
        """Save embeddings using direct file writing to bypass LanceDB and PyArrow issues."""
        try:
            # Only use the filename for consistent table naming
            table_name = self.get_table_name_for_file(file_path)
            file_name = os.path.basename(file_path)

            print(
                f"Saving {len(embeddings)} embeddings for {file_name} using direct method"
            )

            # Create a directory for this document
            doc_dir = os.path.join(self.db_path, "embeddings", table_name)
            os.makedirs(doc_dir, exist_ok=True)

            # Save metadata - store both filename and original path
            embedding_dim = (
                embeddings[0].shape[0]
                if torch.is_tensor(embeddings[0])
                else len(embeddings[0])
            )
            metadata = {
                "filename": file_name,
                "original_path": file_path,
                "page_count": page_count,
                "embedding_count": len(embeddings),
                "embedding_dimension": embedding_dim,
                "dtype": str(embeddings[0].dtype)
                if torch.is_tensor(embeddings[0])
                else "unknown",
                "created_at": str(Path(file_path).stat().st_mtime)
                if os.path.exists(file_path)
                else None,
                "file_size": os.path.getsize(file_path)
                if os.path.exists(file_path)
                else None,
            }

            # Write metadata
            with open(os.path.join(doc_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            # Save each embedding as a separate file
            for i, embedding in enumerate(embeddings):
                if torch.is_tensor(embedding):
                    # Convert tensor to float32 numpy array
                    if embedding.dtype in [torch.bfloat16, torch.float16]:
                        embedding = embedding.to(torch.float32)
                    embedding_np = embedding.cpu().detach().numpy().astype(np.float32)
                else:
                    embedding_np = np.array(embedding, dtype=np.float32)

                # Ensure 2D shape for consistency
                if embedding_np.ndim == 1:
                    embedding_np = embedding_np.reshape(1, -1)

                # Save as numpy file
                np.save(os.path.join(doc_dir, f"embedding_{i:04d}.npy"), embedding_np)

            print(f"Successfully saved {len(embeddings)} embeddings for {file_name}")
            return True

        except Exception as e:
            print(f"Error in direct save: {e}")
            import traceback

            traceback.print_exc()
            return False

    def save_embeddings_lancedb(
        self, file_path: str, embeddings: List[torch.Tensor], page_count: int
    ) -> bool:
        """Save embeddings to LanceDB (if available)."""
        if self.db is None:
            return False

        try:
            table_name = self.get_table_name_for_file(file_path)
            file_name = os.path.basename(file_path)

            print(f"Saving {len(embeddings)} embeddings for {file_name} to LanceDB")

            # Prepare data for LanceDB
            data = []
            for i, embedding in enumerate(embeddings):
                if torch.is_tensor(embedding):
                    if embedding.dtype in [torch.bfloat16, torch.float16]:
                        embedding = embedding.to(torch.float32)
                    embedding_np = embedding.cpu().detach().numpy().astype(np.float32)
                else:
                    embedding_np = np.array(embedding, dtype=np.float32)

                # Ensure 1D for LanceDB
                if embedding_np.ndim > 1:
                    embedding_np = embedding_np.flatten()

                data.append(
                    {
                        "id": f"{table_name}_page_{i:04d}",
                        "page_number": i,
                        "filename": file_name,
                        "embedding": embedding_np.tolist(),
                    }
                )

            # Create or replace table
            if table_name in self.db.table_names():
                self.db.drop_table(table_name)

            table = self.db.create_table(table_name, data)
            print(
                f"Successfully saved {len(embeddings)} embeddings to LanceDB table {table_name}"
            )
            return True

        except Exception as e:
            print(f"Error saving to LanceDB: {e}")
            return False

    def save_embeddings(
        self, file_path: str, embeddings: List[torch.Tensor], page_count: int
    ) -> bool:
        """Save document embeddings to the database using both methods for redundancy."""
        # Always use direct file saving as primary method
        direct_success = self.save_embeddings_direct(file_path, embeddings, page_count)

        # Try LanceDB as backup if available
        lancedb_success = False
        if self.db is not None:
            try:
                lancedb_success = self.save_embeddings_lancedb(
                    file_path, embeddings, page_count
                )
            except Exception as e:
                print(f"LanceDB save failed (using direct storage): {e}")

        return direct_success or lancedb_success

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
            metadata_path = os.path.join(doc_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    print(
                        f"Loaded metadata: {metadata['filename']} with {metadata.get('page_count', 0)} pages"
                    )
            else:
                metadata = {"dtype": "torch.float32"}
                print(f"No metadata file found, using defaults")

            # Determine target dtype
            original_dtype_str = metadata.get("dtype", "torch.float32")
            if "bfloat16" in original_dtype_str:
                target_dtype = torch.bfloat16
            elif "float16" in original_dtype_str:
                target_dtype = torch.float16
            else:
                target_dtype = torch.float32

            # Find all embedding files
            embedding_files = sorted(
                glob.glob(os.path.join(doc_dir, "embedding_*.npy"))
            )

            if not embedding_files:
                print(f"No embedding files found in {doc_dir}")
                return None

            # Load each embedding
            embeddings = []
            for emb_file in embedding_files:
                try:
                    embedding_np = np.load(emb_file)
                    # Create as float32 then convert if needed
                    embedding = torch.tensor(embedding_np, dtype=torch.float32)
                    if target_dtype != torch.float32:
                        embedding = embedding.to(target_dtype)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error loading embedding file {emb_file}: {e}")
                    continue

            if embeddings:
                print(
                    f"Successfully loaded {len(embeddings)} embeddings for {file_name}"
                )
                return embeddings
            else:
                print(f"No valid embeddings could be loaded for {file_name}")
                return None

        except Exception as e:
            print(f"Error in direct load: {e}")
            import traceback

            traceback.print_exc()
            return None

    def load_embeddings_lancedb(self, file_path: str) -> Optional[List[torch.Tensor]]:
        """Load embeddings from LanceDB."""
        if self.db is None:
            return None

        try:
            table_name = self.get_table_name_for_file(file_path)

            if table_name not in self.db.table_names():
                return None

            table = self.db.open_table(table_name)
            results = table.to_pandas().sort_values("page_number")

            embeddings = []
            for _, row in results.iterrows():
                embedding_data = row["embedding"]
                if isinstance(embedding_data, list):
                    embedding = torch.tensor(embedding_data, dtype=torch.float32)
                else:
                    embedding = torch.tensor(embedding_data, dtype=torch.float32)
                embeddings.append(embedding)

            print(f"Successfully loaded {len(embeddings)} embeddings from LanceDB")
            return embeddings

        except Exception as e:
            print(f"Error loading from LanceDB: {e}")
            return None

    def load_embeddings(self, file_path: str) -> Optional[List[torch.Tensor]]:
        """Load document embeddings from the database, trying direct method first."""
        # Try direct file loading first (more reliable)
        embeddings = self.load_embeddings_direct(file_path)

        # Fall back to LanceDB if direct method fails and LanceDB is available
        if embeddings is None and self.db is not None:
            embeddings = self.load_embeddings_lancedb(file_path)

        return embeddings

    def delete_embeddings(self, file_path: str) -> bool:
        """Delete embeddings for a file from the database."""
        success = False
        table_name = self.get_table_name_for_file(file_path)
        file_name = os.path.basename(file_path)

        try:
            # Delete from direct storage
            doc_dir = os.path.join(self.db_path, "embeddings", table_name)
            if os.path.exists(doc_dir):
                shutil.rmtree(doc_dir)
                print(f"Deleted direct storage embeddings for {file_name}")
                success = True

            # Delete from LanceDB if available
            if self.db is not None and table_name in self.db.table_names():
                self.db.drop_table(table_name)
                print(f"Deleted LanceDB table for {file_name}")
                success = True

            return success

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
                for table_name in os.listdir(embeddings_dir):
                    doc_dir = os.path.join(embeddings_dir, table_name)
                    if os.path.isdir(doc_dir):
                        # Load metadata
                        metadata_path = os.path.join(doc_dir, "metadata.json")
                        try:
                            with open(metadata_path, "r") as f:
                                metadata = json.load(f)
                                metadata["table_name"] = table_name
                                metadata["storage_method"] = "direct"
                                documents.append(metadata)
                        except:
                            # If metadata file is missing, create a basic entry
                            embedding_files = glob.glob(
                                os.path.join(doc_dir, "embedding_*.npy")
                            )
                            documents.append(
                                {
                                    "filename": table_name.replace("doc_", ""),
                                    "table_name": table_name,
                                    "page_count": len(embedding_files),
                                    "embedding_dimension": 0,
                                    "storage_method": "direct",
                                }
                            )

            # Check LanceDB tables if available
            if self.db is not None:
                try:
                    for table_name in self.db.table_names():
                        if table_name.startswith("doc_"):
                            # Check if we already have this document from direct storage
                            existing = next(
                                (d for d in documents if d["table_name"] == table_name),
                                None,
                            )
                            if existing:
                                existing["storage_method"] = "both"
                            else:
                                # Add LanceDB-only entry
                                try:
                                    table = self.db.open_table(table_name)
                                    count = table.count_rows()
                                    documents.append(
                                        {
                                            "filename": table_name.replace("doc_", ""),
                                            "table_name": table_name,
                                            "page_count": count,
                                            "embedding_dimension": 0,
                                            "storage_method": "lancedb",
                                        }
                                    )
                                except:
                                    pass
                except Exception as e:
                    print(f"Error reading LanceDB tables: {e}")

            return documents

        except Exception as e:
            print(f"Error listing documents: {e}")
            return []

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database."""
        info = {
            "db_path": self.db_path,
            "lancedb_available": self.db is not None,
            "storage_methods": ["direct"],
        }

        if self.db is not None:
            info["storage_methods"].append("lancedb")
            try:
                info["lancedb_tables"] = self.db.table_names()
            except:
                info["lancedb_tables"] = []

        # Check direct storage
        embeddings_dir = os.path.join(self.db_path, "embeddings")
        if os.path.exists(embeddings_dir):
            info["direct_storage_docs"] = len(
                [
                    d
                    for d in os.listdir(embeddings_dir)
                    if os.path.isdir(os.path.join(embeddings_dir, d))
                ]
            )
        else:
            info["direct_storage_docs"] = 0

        return info

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the database."""
        health = {
            "status": "healthy",
            "issues": [],
            "database_writable": False,
            "lancedb_functional": False,
        }

        # Test write permissions
        try:
            test_file = os.path.join(self.db_path, ".health_check")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            health["database_writable"] = True
        except Exception as e:
            health["issues"].append(f"Database not writable: {e}")
            health["status"] = "degraded"

        # Test LanceDB functionality
        if self.db is not None:
            try:
                # Try a simple operation
                tables = self.db.table_names()
                health["lancedb_functional"] = True
            except Exception as e:
                health["issues"].append(f"LanceDB not functional: {e}")
                health["status"] = "degraded"
        else:
            health["issues"].append("LanceDB not available")

        # If we have issues but direct storage works, we're degraded not broken
        if health["issues"] and health["database_writable"]:
            health["status"] = "degraded"
        elif health["issues"]:
            health["status"] = "unhealthy"

        return health
