#!/usr/bin/env python3
"""
ColPali Long-Running HTTP Server
Supports Apple Silicon M4 with MPS acceleration
"""

import asyncio
import json
import logging
import time
import uuid
import io
import gc  # For garbage collection
import psutil  # For memory monitoring
import urllib.parse  # For URL encoding
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Any

import torch
import lancedb
from PIL import Image
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# ColPali imports for production
try:
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    COLPALI_ENGINE_AVAILABLE = True
    print("✅ Using ColQwen2 for Apple Silicon optimization")
except ImportError:
    try:
        from colpali_engine.models import ColPali, ColPaliProcessor

        COLPALI_ENGINE_AVAILABLE = True
        print("✅ Using ColPali (fallback)")
    except ImportError:
        COLPALI_ENGINE_AVAILABLE = False
        print("⚠️  colpali-engine not available, falling back to transformers")

from transformers import AutoProcessor, AutoModel
from torch.utils.data import DataLoader
import torch.nn.functional as F


@dataclass
class StreamingProgress:
    task_id: str
    progress: float  # 0.0 to 100.0
    current_step: str
    step_num: int
    total_steps: int
    details: str = ""
    eta_seconds: Optional[int] = None
    throughput: Optional[str] = None
    error: Optional[str] = None
    results: Optional[List[Dict]] = None  # For storing search results

    def to_dict(self):
        return asdict(self)


@dataclass
class SearchResult:
    page_num: int
    doc_name: str
    score: float
    snippet: str
    image_path: Optional[str] = None
    full_text: Optional[str] = None


# Pydantic models for API
class IngestRequest(BaseModel):
    file_path: str
    doc_name: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class TaskProgressResponse(BaseModel):
    task_id: str
    progress: Dict[str, Any]


class ColPaliModelManager:
    """Handles ColPali model loading and inference"""

    def __init__(self, device: str = "auto", model_name: str = "vidore/colqwen2-v1.0"):
        # Handle device selection properly for Apple Silicon
        if device == "auto":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                # Remove memory restrictions for maximum performance
                import os

                os.environ.pop(
                    "PYTORCH_MPS_HIGH_WATERMARK_RATIO", None
                )  # Remove memory limit
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
                    "1"  # Keep fallback for compatibility
                )
                print("🧠 Apple Silicon MPS detected - SMART PERFORMANCE MODE")
                print("💾 Intelligent memory management enabled")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print("⚡ CUDA detected - MAXIMUM PERFORMANCE MODE")
            else:
                self.device = "cpu"
                print("⚠️  Using CPU (MPS/CUDA not available)")
        else:
            self.device = device
            print(f"🎯 Using specified device: {device}")

        self.model_name = model_name
        self.model = None
        self.processor = None
        self.model_loaded = False

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"ColPali ModelManager initialized - device: {self.device}, model: {model_name}"
        )

    def _get_device_obj(self):
        """Convert device string to torch.device object if needed"""
        return (
            torch.device(self.device) if isinstance(self.device, str) else self.device
        )

    async def load_model(self) -> AsyncGenerator[StreamingProgress, None]:
        """Load ColPali model with streaming progress"""
        task_id = "model_load"

        yield StreamingProgress(
            task_id=task_id,
            progress=0.0,
            current_step="Initializing model loading",
            step_num=1,
            total_steps=3,
            details=f"Loading ColPali model: {self.model_name}",
        )

        try:
            # Load processor first
            yield StreamingProgress(
                task_id=task_id,
                progress=20.0,
                current_step="Loading ColPali processor",
                step_num=2,
                total_steps=3,
                details="Loading tokenizer and image processor",
            )

            self.logger.info(f"Loading ColPali processor from {self.model_name}")

            # Use the correct processor class
            if "colqwen2" in self.model_name.lower():
                self.processor = ColQwen2Processor.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                self.logger.info("Using ColQwen2Processor for Apple Silicon")
            elif COLPALI_ENGINE_AVAILABLE:
                self.processor = ColPaliProcessor.from_pretrained(self.model_name)
                self.logger.info("Using ColPaliProcessor")
            else:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.logger.info("Using AutoProcessor fallback")

            self.logger.info("ColPali processor loaded successfully")

            yield StreamingProgress(
                task_id=task_id,
                progress=60.0,
                current_step="Loading ColPali model weights",
                step_num=3,
                total_steps=3,
                details=f"Loading to {self.device} device",
            )

            # Load model
            self.logger.info(f"Loading ColPali model from {self.model_name}")

            # Use the correct model class and settings for Apple Silicon
            if "colqwen2" in self.model_name.lower():
                # Use ColQwen2 for Apple Silicon optimization - MAXIMUM PERFORMANCE
                self.model = ColQwen2.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,  # Use float16 for speed
                    device_map=self.device,
                    trust_remote_code=True,
                    # Remove memory restrictions for maximum performance
                    # low_cpu_mem_usage=True  # Removed for speed
                )
                self.logger.info("Using ColQwen2 for Apple Silicon - SMART PERFORMANCE")
            else:
                # Fallback to standard ColPali - MAXIMUM PERFORMANCE
                if COLPALI_ENGINE_AVAILABLE:
                    self.model = ColPali.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16
                        if self.device in ["mps", "cuda"]
                        else torch.float32,
                        device_map=self.device,
                        # No memory restrictions for maximum performance
                    )
                    self.logger.info("Using ColPali engine - MAXIMUM PERFORMANCE")
                else:
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16
                        if self.device in ["mps", "cuda"]
                        else torch.float32,
                        trust_remote_code=True,
                    )
                    self.model.to(self.device)
                    self.logger.info(
                        "Using transformers AutoModel - MAXIMUM PERFORMANCE"
                    )

            self.model.eval()  # Set to evaluation mode
            self.logger.info(f"ColPali model loaded successfully on {self.device}")

            self.model_loaded = True

            yield StreamingProgress(
                task_id=task_id,
                progress=100.0,
                current_step="Model loaded successfully",
                step_num=3,
                total_steps=3,
                details=f"ColPali ready on {self.device}",
                throughput="Ready for inference",
            )

        except Exception as e:
            self.logger.error(f"Failed to load ColPali model: {str(e)}", exc_info=True)
            yield StreamingProgress(
                task_id=task_id,
                progress=0.0,
                current_step="Model loading failed",
                step_num=0,
                total_steps=1,
                error=f"Failed to load ColPali model: {str(e)}",
            )
            raise

    async def encode_pages(
        self, images: List[Image.Image]
    ) -> AsyncGenerator[StreamingProgress, None]:
        """Encode PDF pages with streaming progress and memory management"""
        task_id = f"encode_{uuid.uuid4().hex[:8]}"
        total_pages = len(images)
        start_time = time.time()

        yield StreamingProgress(
            task_id=task_id,
            progress=0.0,
            current_step="Starting page encoding",
            step_num=1,
            total_steps=total_pages,
            details=f"Processing {total_pages} pages with ColPali",
        )

        if not self.model_loaded:
            raise RuntimeError("ColPali model not loaded. Call load_model() first.")

        embeddings = []

        # MEMORY-OPTIMIZED batch sizing
        initial_batch_size = 1  # Start very conservative
        max_batch_size = 4  # Never exceed this
        current_batch_size = initial_batch_size

        # Memory monitoring
        memory_threshold = 85  # Reduce batch if memory > 85%

        self.logger.info(f"Starting with conservative batch size: {current_batch_size}")

        try:
            with torch.no_grad():
                for i in range(0, total_pages, current_batch_size):
                    # Monitor memory before each batch
                    try:
                        memory_percent = psutil.virtual_memory().percent
                        if memory_percent > memory_threshold:
                            # Force garbage collection
                            gc.collect()
                            if self.device == "mps":
                                torch.mps.empty_cache()
                            elif self.device == "cuda":
                                torch.cuda.empty_cache()

                            # Reduce batch size if memory is high
                            if current_batch_size > 1:
                                current_batch_size = max(1, current_batch_size - 1)
                                self.logger.warning(
                                    f"High memory usage ({memory_percent:.1f}%), reducing batch size to {current_batch_size}"
                                )
                    except Exception:
                        pass  # Continue if memory monitoring fails

                    batch_images = images[i : i + current_batch_size]
                    batch_start = i
                    batch_end = min(i + current_batch_size, total_pages)

                    current_progress = (batch_start / total_pages) * 100
                    elapsed = time.time() - start_time
                    pages_per_sec = batch_start / elapsed if elapsed > 0 else 0

                    yield StreamingProgress(
                        task_id=task_id,
                        progress=current_progress,
                        current_step=f"Encoding batch {batch_start + 1}-{batch_end}/{total_pages}",
                        step_num=batch_start + 1,
                        total_steps=total_pages,
                        details=f"Memory-safe batch size: {len(batch_images)}",
                        throughput=f"{pages_per_sec:.1f} pages/sec",
                    )

                    # Process images with retry logic
                    max_retries = 3
                    retry_count = 0
                    batch_success = False

                    while retry_count < max_retries and not batch_success:
                        try:
                            # Process batch of images
                            if COLPALI_ENGINE_AVAILABLE and hasattr(
                                self.processor, "process_images"
                            ):
                                batch_inputs = self.processor.process_images(
                                    batch_images
                                )
                            else:
                                batch_inputs = self.processor(
                                    images=batch_images,
                                    return_tensors="pt",
                                    padding=True,
                                    max_length=512,  # Limit sequence length
                                    truncation=True,
                                )

                            # Move to device with memory check
                            device_obj = self._get_device_obj()
                            for key in batch_inputs:
                                if isinstance(batch_inputs[key], torch.Tensor):
                                    batch_inputs[key] = batch_inputs[key].to(device_obj)

                            # Model inference with memory management
                            try:
                                batch_embeddings = self.model(**batch_inputs)
                                batch_success = True

                                # Process embeddings immediately to free memory
                                for emb in batch_embeddings:
                                    # Move to CPU immediately and convert to float32 for consistency
                                    embeddings.append(emb.cpu().detach().float())

                                # Clean up GPU/MPS memory immediately
                                del batch_embeddings
                                del batch_inputs

                                if self.device == "mps":
                                    torch.mps.empty_cache()
                                elif self.device == "cuda":
                                    torch.cuda.empty_cache()

                                gc.collect()

                            except RuntimeError as inference_error:
                                error_msg = str(inference_error).lower()
                                if (
                                    "buffer size" in error_msg
                                    or "memory" in error_msg
                                    or "out of memory" in error_msg
                                ):
                                    self.logger.error(
                                        f"Memory error during inference: {inference_error}"
                                    )

                                    # Clean up memory
                                    del batch_inputs
                                    if self.device == "mps":
                                        torch.mps.empty_cache()
                                    elif self.device == "cuda":
                                        torch.cuda.empty_cache()
                                    gc.collect()

                                    # If we're already at batch size 1, we can't reduce further
                                    if len(batch_images) == 1:
                                        raise RuntimeError(
                                            f"Cannot process even a single page due to memory constraints. "
                                            f"Try closing other applications or reducing image resolution. "
                                            f"Original error: {inference_error}"
                                        )
                                    else:
                                        # Split the batch and retry
                                        batch_images = batch_images[
                                            : len(batch_images) // 2
                                        ]
                                        current_batch_size = len(batch_images)
                                        retry_count += 1
                                        self.logger.warning(
                                            f"Retrying with smaller batch size: {len(batch_images)} (attempt {retry_count})"
                                        )
                                        continue
                                else:
                                    raise

                        except Exception as process_error:
                            retry_count += 1
                            if retry_count >= max_retries:
                                raise RuntimeError(
                                    f"Failed to process batch after {max_retries} attempts: {process_error}"
                                )

                            self.logger.warning(
                                f"Batch processing failed, retry {retry_count}: {process_error}"
                            )
                            # Reduce batch size for retry
                            if len(batch_images) > 1:
                                batch_images = batch_images[: len(batch_images) // 2]
                            await asyncio.sleep(0.5)  # Brief pause before retry

                    # Brief pause between batches to allow system recovery
                    await asyncio.sleep(0.1)

            final_progress = (total_pages / total_pages) * 100
            elapsed = time.time() - start_time
            avg_throughput = total_pages / elapsed if elapsed > 0 else 0

            yield StreamingProgress(
                task_id=task_id,
                progress=100.0,
                current_step="Page encoding complete",
                step_num=total_pages,
                total_steps=total_pages,
                details=f"Generated {len(embeddings)} embeddings with memory optimization",
                throughput=f"Average: {avg_throughput:.1f} pages/sec",
            )

            self.logger.info(
                f"Successfully encoded {len(embeddings)} pages with memory optimization"
            )

        except Exception as e:
            self.logger.error(f"Failed to encode pages: {str(e)}", exc_info=True)
            yield StreamingProgress(
                task_id=task_id,
                progress=0.0,
                current_step="Encoding failed",
                step_num=0,
                total_steps=1,
                error=f"Failed to encode pages: {str(e)}",
            )
            raise

    async def encode_query(self, query: str) -> AsyncGenerator[StreamingProgress, None]:
        """Encode search query with streaming progress"""
        task_id = f"query_{uuid.uuid4().hex[:8]}"

        yield StreamingProgress(
            task_id=task_id,
            progress=0.0,
            current_step="Processing query",
            step_num=1,
            total_steps=3,
            details=f"Query: '{query[:50]}...'",
        )

        if not self.model_loaded:
            raise RuntimeError("ColPali model not loaded. Call load_model() first.")

        try:
            yield StreamingProgress(
                task_id=task_id,
                progress=40.0,
                current_step="Tokenizing query",
                step_num=2,
                total_steps=3,
                details="Converting text to tokens",
            )

            # Process query with ColPali processor
            with torch.no_grad():
                if COLPALI_ENGINE_AVAILABLE and hasattr(
                    self.processor, "process_queries"
                ):
                    query_inputs = self.processor.process_queries([query])
                else:
                    # Fallback for transformers processor
                    query_inputs = self.processor(
                        text=[query], return_tensors="pt", padding=True
                    )

                # Move to device (handle string device properly)
                device_obj = self._get_device_obj()
                for key in query_inputs:
                    if isinstance(query_inputs[key], torch.Tensor):
                        query_inputs[key] = query_inputs[key].to(device_obj)

                yield StreamingProgress(
                    task_id=task_id,
                    progress=80.0,
                    current_step="Generating embedding",
                    step_num=3,
                    total_steps=3,
                    details="ColPali query encoding",
                )

                # Get query embedding from ColPali model
                query_embedding = self.model(**query_inputs)

                yield StreamingProgress(
                    task_id=task_id,
                    progress=100.0,
                    current_step="Query encoded successfully",
                    step_num=3,
                    total_steps=3,
                    details="Ready for similarity search",
                )

        except Exception as e:
            self.logger.error(f"Failed to encode query: {str(e)}", exc_info=True)
            yield StreamingProgress(
                task_id=task_id,
                progress=0.0,
                current_step="Query encoding failed",
                step_num=0,
                total_steps=1,
                error=f"Failed to encode query: {str(e)}",
            )
            raise

    async def encode_query_simple(self, query: str) -> torch.Tensor:
        """Simple query encoding that returns the embedding directly"""
        if not self.model_loaded:
            raise RuntimeError("ColPali model not loaded. Call load_model() first.")

        try:
            with torch.no_grad():
                # Process query with ColPali processor
                if COLPALI_ENGINE_AVAILABLE and hasattr(
                    self.processor, "process_queries"
                ):
                    query_inputs = self.processor.process_queries([query])
                else:
                    # Fallback for transformers processor
                    query_inputs = self.processor(
                        text=[query], return_tensors="pt", padding=True
                    )

                # Move to device (handle string device properly)
                device_obj = self._get_device_obj()
                for key in query_inputs:
                    if isinstance(query_inputs[key], torch.Tensor):
                        query_inputs[key] = query_inputs[key].to(device_obj)

                # Get query embedding from ColPali model
                query_embedding = self.model(**query_inputs)

                # Return first embedding (for single query) and ensure consistent dtype
                result_embedding = (
                    query_embedding[0].cpu().float()
                )  # Convert to float32 for consistency
                return result_embedding

        except Exception as e:
            self.logger.error(
                f"Failed to encode query '{query}': {str(e)}", exc_info=True
            )
            raise

    def cleanup_memory(self):
        """Force memory cleanup"""
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def maxsim_score(
        self, query_embedding: torch.Tensor, doc_embedding: torch.Tensor
    ) -> float:
        """Compute MaxSim score between query and document embeddings for ColPali"""
        try:
            # Ensure both embeddings are 2D [patches, dim]
            if query_embedding.dim() != 2:
                raise ValueError(
                    f"Query embedding must be 2D [patches, dim], got shape {query_embedding.shape}"
                )
            if doc_embedding.dim() != 2:
                raise ValueError(
                    f"Document embedding must be 2D [patches, dim], got shape {doc_embedding.shape}"
                )

            # Compute similarity matrix: [query_patches, doc_patches]
            sim_matrix = torch.mm(query_embedding, doc_embedding.t())

            # MaxSim: for each query patch, find max similarity across doc patches
            max_sims = torch.max(sim_matrix, dim=1)[0]  # [query_patches]

            # Sum the max similarities (ColPali approach)
            maxsim_score = torch.sum(max_sims).item()

            return maxsim_score

        except Exception as e:
            self.logger.error(f"MaxSim scoring failed: {e}")
            return 0.0

    def should_use_maxsim(self, embedding: torch.Tensor) -> bool:
        """Determine if we should use MaxSim scoring based on embedding shape"""
        return embedding.dim() == 2 and embedding.shape[0] > 1  # Multiple patches


class LanceDBManager:
    """Handles LanceDB operations"""

    def __init__(self, db_path: str = "./data/embeddings_db"):
        self.db_path = db_path
        self.db = None
        self.table = None

    async def initialize(self):
        """Initialize LanceDB connection"""
        logger = logging.getLogger(__name__)
        logger.info(f"Initializing LanceDB connection at: {self.db_path}")

        try:
            self.db = lancedb.connect(self.db_path)
            logger.info(f"LanceDB connection established successfully")

            # Check if documents table exists
            try:
                existing_tables = self.db.table_names()
                logger.info(f"Existing tables in database: {existing_tables}")

                if "documents" in existing_tables:
                    self.table = self.db.open_table("documents")
                    row_count = len(self.table)
                    logger.info(
                        f"Opened existing 'documents' table with {row_count} rows"
                    )
                else:
                    logger.info(
                        "No 'documents' table found - will be created when first document is ingested"
                    )

            except Exception as table_check_error:
                logger.warning(f"Could not check existing tables: {table_check_error}")

        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {str(e)}", exc_info=True)
            raise

    async def search_embeddings(self, query_embedding, limit=10, score_threshold=0.8):
        """
        Search for similar embeddings in the database using ColPali MaxSim scoring

        Args:
            query_embedding: The query embedding vector (patch-level for ColPali)
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold

        Yields:
            StreamingProgress objects
        """
        task_id = f"search_{uuid.uuid4().hex[:8]}"
        logger = logging.getLogger(__name__)

        logger.info(
            f"Task {task_id}: Starting ColPali search with limit={limit}, score_threshold={score_threshold}"
        )

        try:
            # Check database and table state
            if self.db is None:
                await self.initialize()

            if self.table is None:
                try:
                    self.table = self.db.open_table("documents")
                except Exception as table_error:
                    yield StreamingProgress(
                        task_id=task_id,
                        progress=0.0,
                        current_step="Database table not found",
                        step_num=0,
                        total_steps=1,
                        error=f"Database table not initialized: {str(table_error)}",
                    )
                    return

            # Check if table has data
            try:
                table_count = len(self.table)
                logger.info(f"Task {task_id}: Table contains {table_count} rows")

                if table_count == 0:
                    yield StreamingProgress(
                        task_id=task_id,
                        progress=100.0,
                        current_step="Search completed",
                        step_num=3,
                        total_steps=3,
                        details="No documents in database",
                        results=[],
                    )
                    return
            except Exception as debug_error:
                logger.error(f"Task {task_id}: Error checking table: {debug_error}")

            yield StreamingProgress(
                task_id=task_id,
                progress=25.0,
                current_step="Query encoding complete",
                step_num=1,
                total_steps=4,
                details="Starting ColPali MaxSim search",
            )

            # Get all documents for ColPali MaxSim scoring
            logger.info(f"Task {task_id}: Retrieving all documents for MaxSim scoring")
            all_docs_df = self.table.to_pandas()
            all_docs = all_docs_df.to_dict("records")

            logger.info(
                f"Task {task_id}: Processing {len(all_docs)} documents with ColPali MaxSim"
            )

            yield StreamingProgress(
                task_id=task_id,
                progress=50.0,
                current_step="Computing ColPali MaxSim scores",
                step_num=2,
                total_steps=4,
                details=f"Scoring {len(all_docs)} document pages",
            )

            # Compute MaxSim scores for all documents
            scored_results = []

            for i, doc in enumerate(all_docs):
                try:
                    # Reconstruct the original embedding shape from stored data
                    stored_vector = doc.get("vector", [])
                    embedding_shape_str = doc.get("embedding_shape", "unknown")

                    # Parse the stored shape (e.g., "torch.Size([704, 128])")
                    if (
                        "torch.Size([" in embedding_shape_str
                        and "])" in embedding_shape_str
                    ):
                        # Extract dimensions from string like "torch.Size([704, 128])"
                        shape_str = embedding_shape_str.replace(
                            "torch.Size([", ""
                        ).replace("])", "")
                        dims = [int(d.strip()) for d in shape_str.split(",")]
                        if len(dims) == 2:
                            patches, dim = dims

                            # Reshape flattened vector back to [patches, dim]
                            if len(stored_vector) == patches * dim:
                                import numpy as np

                                doc_embedding_np = np.array(stored_vector).reshape(
                                    patches, dim
                                )
                                doc_embedding = torch.from_numpy(
                                    doc_embedding_np
                                ).float()

                                # Compute MaxSim score
                                maxsim_score = self.compute_maxsim_score(
                                    query_embedding, doc_embedding
                                )

                                scored_results.append(
                                    {
                                        "doc": doc,
                                        "score": maxsim_score,
                                        "doc_name": doc.get("doc_name", ""),
                                        "page_num": doc.get("page_num", 0),
                                    }
                                )

                                logger.debug(
                                    f"Task {task_id}: Doc {i} MaxSim score: {maxsim_score:.4f}"
                                )
                            else:
                                logger.warning(
                                    f"Task {task_id}: Vector length mismatch for doc {i}"
                                )
                        else:
                            logger.warning(
                                f"Task {task_id}: Invalid shape format for doc {i}: {embedding_shape_str}"
                            )
                    else:
                        logger.warning(
                            f"Task {task_id}: Could not parse embedding shape for doc {i}: {embedding_shape_str}"
                        )

                except Exception as scoring_error:
                    logger.error(
                        f"Task {task_id}: Error scoring doc {i}: {scoring_error}"
                    )
                    continue

            logger.info(
                f"Task {task_id}: Computed MaxSim scores for {len(scored_results)} documents"
            )

            yield StreamingProgress(
                task_id=task_id,
                progress=80.0,
                current_step="Ranking and filtering results",
                step_num=3,
                total_steps=4,
                details=f"Found {len(scored_results)} scored results",
            )

            # Sort by MaxSim score (higher is better)
            scored_results.sort(key=lambda x: x["score"], reverse=True)

            # Apply score threshold and limit
            filtered_results = []
            for result in scored_results:
                if result["score"] >= score_threshold:
                    filtered_results.append(result)
                if len(filtered_results) >= limit:
                    break

            logger.info(
                f"Task {task_id}: After filtering: {len(filtered_results)} results (threshold: {score_threshold})"
            )

            # Format final results
            search_results = []
            for result in filtered_results:
                doc = result["doc"]
                text_content = doc.get("text_content", "")
                snippet = text_content[:200] if text_content else "No text content"

                # Create HTTP-accessible image URL instead of file path
                image_url = None
                if doc.get("image_path"):
                    doc_name = doc.get("doc_name", "")
                    page_num = doc.get("page_num", 0)
                    # URL encode the document name to handle special characters
                    encoded_doc_name = urllib.parse.quote(doc_name, safe="")
                    image_url = (
                        f"http://127.0.0.1:8000/image/{encoded_doc_name}/{page_num}"
                    )

                search_result = SearchResult(
                    page_num=doc.get("page_num", 0),
                    doc_name=doc.get("doc_name", ""),
                    score=result["score"],
                    snippet=snippet,
                    image_path=image_url,  # Use HTTP URL instead of file path
                    full_text=text_content,  # Include full text in response
                )
                search_results.append(search_result)

            logger.info(f"Task {task_id}: Final results: {len(search_results)} matches")

            yield StreamingProgress(
                task_id=task_id,
                progress=100.0,
                current_step="ColPali search completed",
                step_num=4,
                total_steps=4,
                details=f"Found {len(search_results)} relevant matches using MaxSim",
                results=[result.__dict__ for result in search_results],
            )

        except Exception as e:
            logger.error(
                f"Task {task_id}: ColPali search failed: {str(e)}", exc_info=True
            )
            yield StreamingProgress(
                task_id=task_id,
                progress=0.0,
                current_step="Search failed",
                step_num=0,
                total_steps=1,
                error=f"ColPali search failed: {str(e)}",
            )

    def compute_maxsim_score(
        self, query_embedding: torch.Tensor, doc_embedding: torch.Tensor
    ) -> float:
        """
        Compute MaxSim score between query and document embeddings for ColPali
        """
        try:
            # Ensure both embeddings are 2D [patches, dim]
            if query_embedding.dim() != 2:
                raise ValueError(
                    f"Query embedding must be 2D [patches, dim], got shape {query_embedding.shape}"
                )
            if doc_embedding.dim() != 2:
                raise ValueError(
                    f"Document embedding must be 2D [patches, dim], got shape {doc_embedding.shape}"
                )

            # Ensure both embeddings have the same dtype (fix for Half vs float mismatch)
            if query_embedding.dtype != doc_embedding.dtype:
                logging.getLogger(__name__).info(
                    f"Converting dtype mismatch: query {query_embedding.dtype} -> doc {doc_embedding.dtype}"
                )
                # Convert both to float32 for consistency
                query_embedding = query_embedding.float()
                doc_embedding = doc_embedding.float()

            # Normalize embeddings (important for cosine similarity)
            query_norm = F.normalize(
                query_embedding, p=2, dim=1
            )  # [query_patches, dim]
            doc_norm = F.normalize(doc_embedding, p=2, dim=1)  # [doc_patches, dim]

            # Compute similarity matrix: [query_patches, doc_patches]
            sim_matrix = torch.mm(query_norm, doc_norm.t())

            # MaxSim: for each query patch, find max similarity across doc patches
            max_sims = torch.max(sim_matrix, dim=1)[0]  # [query_patches]

            # Sum the max similarities (ColPali approach)
            maxsim_score = torch.sum(max_sims).item()

            return maxsim_score

        except Exception as e:
            logging.getLogger(__name__).error(f"MaxSim scoring failed: {e}")
            return 0.0

    async def store_embeddings(
        self, embeddings: List[torch.Tensor], metadata: List[Dict]
    ) -> AsyncGenerator[StreamingProgress, None]:
        """Store embeddings with streaming progress"""
        task_id = f"store_{uuid.uuid4().hex[:8]}"
        total_embeddings = len(embeddings)
        logger = logging.getLogger(__name__)  # Add logger reference

        yield StreamingProgress(
            task_id=task_id,
            progress=0.0,
            current_step="Preparing data for storage",
            step_num=1,
            total_steps=3,
            details=f"Processing {total_embeddings} embeddings",
        )

        # Prepare data with timestamp
        import datetime

        current_time = datetime.datetime.now().isoformat()

        data = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            # Log embedding format for debugging
            embedding_shape = getattr(embedding, "shape", "no shape")
            flattened_vector = embedding.numpy().flatten().tolist()
            logger.info(
                f"Storing embedding {i}: original shape {embedding_shape}, flattened length {len(flattened_vector)}"
            )

            data.append(
                {
                    "id": f"{meta['doc_name']}_page_{meta['page_num']}",
                    "vector": flattened_vector,
                    "doc_name": meta["doc_name"],
                    "page_num": meta["page_num"],
                    "text_content": meta.get("text_content", ""),
                    "image_path": meta.get("image_path", ""),  # Store image path
                    "created_at": current_time,
                    "file_size": meta.get("file_size", "unknown"),
                    "embedding_shape": str(
                        embedding_shape
                    ),  # Store original shape for reference
                }
            )

        yield StreamingProgress(
            task_id=task_id,
            progress=30.0,
            current_step="Creating/updating table",
            step_num=2,
            total_steps=3,
            details="Setting up vector index",
        )

        await asyncio.sleep(1)

        try:
            # Initialize database if not already done
            if self.db is None:
                await self.initialize()

            # Try to open existing table or create new one
            try:
                # First, try to open existing table
                self.table = self.db.open_table("documents")
                # Add new data to existing table
                self.table.add(data)
                logger.info(f"Added {total_embeddings} embeddings to existing table")
            except Exception as open_error:
                logger.info(f"Could not open existing table: {open_error}")
                try:
                    # If opening fails, try to create new table
                    self.table = self.db.create_table("documents", data)
                    logger.info(f"Created new table with {total_embeddings} embeddings")
                except Exception as create_error:
                    logger.error(f"Could not create table: {create_error}")
                    # If both fail, try alternative approaches
                    try:
                        # Check if table exists in database
                        existing_tables = self.db.table_names()
                        if "documents" in existing_tables:
                            # Table exists, try to open it again with different approach
                            self.table = self.db.open_table("documents")
                            self.table.add(data)
                            logger.info(
                                f"Successfully added to existing table on retry"
                            )
                        else:
                            # Table doesn't exist, create it
                            self.table = self.db.create_table("documents", data)
                            logger.info(f"Created new table on retry")
                    except Exception as final_error:
                        raise Exception(
                            f"Failed all storage attempts: open_error={open_error}, create_error={create_error}, final_error={final_error}"
                        )

            yield StreamingProgress(
                task_id=task_id,
                progress=100.0,
                current_step="Storage complete",
                step_num=3,
                total_steps=3,
                details=f"Stored {total_embeddings} embeddings successfully",
            )

        except Exception as e:
            yield StreamingProgress(
                task_id=task_id,
                progress=0.0,
                current_step="Storage failed",
                step_num=3,
                total_steps=3,
                error=str(e),
            )


class PDFProcessor:
    """Handles PDF processing and page extraction"""

    @staticmethod
    async def extract_pages(file_path: str) -> AsyncGenerator[StreamingProgress, None]:
        """Extract pages from PDF with streaming progress"""
        task_id = f"extract_{uuid.uuid4().hex[:8]}"

        yield StreamingProgress(
            task_id=task_id,
            progress=0.0,
            current_step="Opening PDF file",
            step_num=1,
            total_steps=3,
            details=f"Loading {file_path}",
        )

        await asyncio.sleep(0.5)

        # Open PDF
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
        except Exception as e:
            yield StreamingProgress(
                task_id=task_id,
                progress=0.0,
                current_step="Error opening PDF",
                step_num=1,
                total_steps=3,
                error=str(e),
            )
            return

        yield StreamingProgress(
            task_id=task_id,
            progress=20.0,
            current_step=f"Processing {total_pages} pages",
            step_num=2,
            total_steps=3,
            details="Extracting images and text",
        )

        images = []
        metadata = []

        for page_num in range(total_pages):
            page = doc[page_num]

            # Extract image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            images.append(image)

            # Extract text
            text_content = page.get_text()

            metadata.append(
                {
                    "page_num": page_num + 1,
                    "doc_name": Path(file_path).stem,
                    "text_content": text_content,
                }
            )

            # Update progress
            page_progress = 20.0 + (page_num + 1) / total_pages * 70.0
            yield StreamingProgress(
                task_id=task_id,
                progress=page_progress,
                current_step=f"Extracted page {page_num + 1}/{total_pages}",
                step_num=2,
                total_steps=3,
                details=f"Image: {image.size}, Text: {len(text_content)} chars",
            )

            await asyncio.sleep(0.1)  # Yield control

        doc.close()

        yield StreamingProgress(
            task_id=task_id,
            progress=100.0,
            current_step="PDF extraction complete",
            step_num=3,
            total_steps=3,
            details=f"Extracted {len(images)} pages successfully",
        )


class ColPaliHTTPServer:
    """Main HTTP Server - Long Running"""

    def __init__(self):
        self.app = FastAPI(title="ColPali HTTP Server", version="1.0.0")
        self.model_manager = ColPaliModelManager()
        self.db_manager = LanceDBManager()
        self.pdf_processor = PDFProcessor()

        # Progress tracking - persistent across requests
        self.active_tasks: Dict[str, AsyncGenerator] = {}
        self.latest_progress: Dict[str, StreamingProgress] = {}

        # Setup enhanced logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),  # Console output
                logging.FileHandler("colpali_server.log", mode="a"),  # File logging
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("ColPali HTTP Server initialized")

        # Setup routes
        self.setup_routes()

    def save_page_image(self, image: Image.Image, doc_name: str, page_num: int) -> str:
        """Save page image to disk and return path"""
        images_dir = Path("./data/extracted_images")
        images_dir.mkdir(parents=True, exist_ok=True)

        # Create safe filename from doc_name
        safe_doc_name = "".join(
            c for c in doc_name if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_doc_name = safe_doc_name.replace(" ", "_")

        image_filename = f"{safe_doc_name}_page_{page_num}.png"
        image_path = images_dir / image_filename

        try:
            image.save(image_path, "PNG")
            self.logger.info(f"Saved page image: {image_path}")
            return str(image_path.absolute())
        except Exception as e:
            self.logger.error(f"Failed to save page image {image_path}: {e}")
            return ""

    def setup_routes(self):
        """Setup FastAPI routes"""

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify your domain
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "model_loaded": self.model_manager.model_loaded,
                "model_name": self.model_manager.model_name
                if hasattr(self.model_manager, "model_name")
                else "unknown",
                "device": str(self.model_manager.device)
                if hasattr(self.model_manager, "device")
                else "unknown",
                "active_tasks": len(self.active_tasks),
            }

        @self.app.get("/image/{doc_name}/{page_num}")
        async def serve_page_image(doc_name: str, page_num: int):
            """Serve page images via HTTP"""
            try:
                # Construct the image path
                images_dir = Path("./data/extracted_images")
                safe_doc_name = "".join(c for c in doc_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_doc_name = safe_doc_name.replace(' ', '_')
                image_filename = f"{safe_doc_name}_page_{page_num}.png"
                image_path = images_dir / image_filename
                
                if not image_path.exists():
                    raise HTTPException(status_code=404, detail="Image not found")
                
                # Read and return the image file
                return FileResponse(
                    image_path,
                    media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"}  # Cache for 1 hour
                )
                
            except Exception as e:
                self.logger.error(f"Error serving image {doc_name}/page_{page_num}: {e}")
                raise HTTPException(status_code=500, detail="Failed to serve image")

        @self.app.get("/document/{doc_name}/page/{page_num}/text")
        async def get_page_text(doc_name: str, page_num: int):
            """Get the full text content for a specific page"""
            try:
                # Initialize database if needed
                if self.db_manager.db is None:
                    await self.db_manager.initialize()

                if self.db_manager.table is None:
                    raise HTTPException(status_code=404, detail="No documents table found")

                # Query for the specific document and page
                all_docs = self.db_manager.table.to_pandas()
                
                # URL decode the document name to handle special characters
                decoded_doc_name = urllib.parse.unquote(doc_name)
                
                self.logger.info(f"Looking for doc_name='{decoded_doc_name}', page_num={page_num}")
                
                # Find the matching document and page
                matching_docs = all_docs[
                    (all_docs["doc_name"] == decoded_doc_name) & 
                    (all_docs["page_num"] == page_num)
                ]
                
                if len(matching_docs) == 0:
                    # Log available documents for debugging
                    available = all_docs[["doc_name", "page_num"]].drop_duplicates().head(5)
                    self.logger.warning(f"No match found. Available docs: {available.to_dict('records')}")
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Page {page_num} of document '{decoded_doc_name}' not found"
                    )
                
                # Get the text content
                doc_row = matching_docs.iloc[0]
                text_content = doc_row.get("text_content", "")
                
                self.logger.info(f"Found text content: {len(text_content)} characters")
                
                if not text_content or text_content.strip() == "":
                    text_content = "No text content was extracted from this page during ingestion."
                
                return {
                    "doc_name": decoded_doc_name,
                    "page_num": page_num,
                    "text_content": text_content,
                    "character_count": len(text_content),
                    "status": "success"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error retrieving text for {doc_name}/page_{page_num}: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to retrieve text content: {str(e)}"
                )

        @self.app.get("/debug/document/{doc_name}/page/{page_num}")
        async def debug_document_page(doc_name: str, page_num: int):
            """Debug endpoint to see what's stored for a specific page"""
            try:
                if self.db_manager.db is None:
                    await self.db_manager.initialize()

                if self.db_manager.table is None:
                    return {"error": "No documents table found"}

                # Get all data for debugging
                all_docs = self.db_manager.table.to_pandas()
                decoded_doc_name = urllib.parse.unquote(doc_name)
                
                # Find matching documents
                matching_docs = all_docs[
                    (all_docs["doc_name"] == decoded_doc_name) & 
                    (all_docs["page_num"] == page_num)
                ]
                
                if len(matching_docs) == 0:
                    return {
                        "error": f"No matching documents found",
                        "searched_for": {
                            "doc_name": decoded_doc_name,
                            "page_num": page_num
                        },
                        "available_docs": all_docs[["doc_name", "page_num", "text_content"]].head(10).to_dict("records")
                    }
                
                doc_row = matching_docs.iloc[0]
                return {
                    "doc_name": doc_row.get("doc_name"),
                    "page_num": doc_row.get("page_num"),
                    "text_content": doc_row.get("text_content", ""),
                    "text_length": len(str(doc_row.get("text_content", ""))),
                    "all_fields": list(doc_row.keys()),
                    "raw_data": doc_row.to_dict()
                }
                
            except Exception as e:
                return {"error": str(e)}

        @self.app.get("/image/{doc_name}/{page_num}")
        async def serve_page_image(doc_name: str, page_num: int):
            """Serve page images via HTTP"""
            try:
                # Construct the image path
                images_dir = Path("./data/extracted_images")
                safe_doc_name = "".join(
                    c for c in doc_name if c.isalnum() or c in (" ", "-", "_")
                ).rstrip()
                safe_doc_name = safe_doc_name.replace(" ", "_")
                image_filename = f"{safe_doc_name}_page_{page_num}.png"
                image_path = images_dir / image_filename

                if not image_path.exists():
                    raise HTTPException(status_code=404, detail="Image not found")

                # Read and return the image file
                return FileResponse(
                    image_path,
                    media_type="image/png",
                    headers={
                        "Cache-Control": "public, max-age=3600"
                    },  # Cache for 1 hour
                )

            except Exception as e:
                self.logger.error(
                    f"Error serving image {doc_name}/page_{page_num}: {e}"
                )
                raise HTTPException(status_code=500, detail="Failed to serve image")

        @self.app.get("/document/{doc_name}/page/{page_num}/text")
        async def get_page_text(doc_name: str, page_num: int):
            """Get the full text content for a specific page"""
            try:
                # Initialize database if needed
                if self.db_manager.db is None:
                    await self.db_manager.initialize()

                if self.db_manager.table is None:
                    raise HTTPException(
                        status_code=404, detail="No documents table found"
                    )

                # Query for the specific document and page
                all_docs = self.db_manager.table.to_pandas()

                # URL decode the document name to handle special characters
                decoded_doc_name = urllib.parse.unquote(doc_name)

                # Find the matching document and page
                matching_docs = all_docs[
                    (all_docs["doc_name"] == decoded_doc_name)
                    & (all_docs["page_num"] == page_num)
                ]

                if len(matching_docs) == 0:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Page {page_num} of document '{decoded_doc_name}' not found",
                    )

                # Get the text content
                doc_row = matching_docs.iloc[0]
                text_content = doc_row.get("text_content", "")

                if not text_content:
                    text_content = "No text content available for this page."

                return {
                    "doc_name": decoded_doc_name,
                    "page_num": page_num,
                    "text_content": text_content,
                    "status": "success",
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(
                    f"Error retrieving text for {doc_name}/page_{page_num}: {e}"
                )
                raise HTTPException(
                    status_code=500, detail=f"Failed to retrieve text content: {str(e)}"
                )

        @self.app.get("/model/status")
        async def model_status():
            """Get detailed model loading status"""
            return {
                "model_loaded": self.model_manager.model_loaded,
                "model_name": getattr(self.model_manager, "model_name", "unknown"),
                "device": str(getattr(self.model_manager, "device", "unknown")),
                "processor_loaded": self.model_manager.processor is not None,
                "model_ready": self.model_manager.model is not None,
            }

        @self.app.post("/initialize")
        async def initialize_model():
            """Initialize ColPali model"""
            task_id = f"init_{uuid.uuid4().hex[:8]}"
            progress_updates = []

            try:
                async for progress in self.model_manager.load_model():
                    self.latest_progress[task_id] = progress
                    progress_updates.append(progress.to_dict())

                return {
                    "task_id": task_id,
                    "status": "completed",
                    "final_progress": progress_updates[-1]
                    if progress_updates
                    else None,
                    "message": "ColPali model initialized successfully",
                }

            except Exception as e:
                error_progress = StreamingProgress(
                    task_id=task_id,
                    progress=0.0,
                    current_step="Initialization failed",
                    step_num=0,
                    total_steps=1,
                    error=str(e),
                )
                self.latest_progress[task_id] = error_progress
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/ingest")
        async def ingest_pdf(file: UploadFile = File(...), doc_name: str = None):
            """Ingest PDF with streaming progress"""
            task_id = f"ingest_{uuid.uuid4().hex[:8]}"
            self.logger.info(
                f"Starting ingestion task {task_id} for file: {file.filename}"
            )

            # Validate file type
            if not file.filename.endswith(".pdf"):
                self.logger.error(
                    f"Invalid file type for {file.filename} - only PDF supported"
                )
                raise HTTPException(
                    status_code=400, detail="Only PDF files are supported"
                )

            # Use provided doc_name or derive from filename
            actual_doc_name = doc_name or Path(file.filename).stem
            self.logger.info(f"Processing document: {actual_doc_name}")

            # Save uploaded file temporarily
            temp_file_path = f"/tmp/{file.filename}"

            try:
                # Save uploaded file
                self.logger.info(f"Saving uploaded file to {temp_file_path}")
                file_size = 0
                with open(temp_file_path, "wb") as buffer:
                    content = await file.read()
                    file_size = len(content)
                    buffer.write(content)
                self.logger.info(
                    f"File saved successfully, size: {file_size / 1024 / 1024:.2f} MB"
                )

                # Initial progress update
                initial_progress = StreamingProgress(
                    task_id=task_id,
                    progress=5.0,
                    current_step="File uploaded successfully",
                    step_num=1,
                    total_steps=6,
                    details=f"Processing {actual_doc_name} ({file_size / 1024 / 1024:.2f} MB)",
                )
                self.latest_progress[task_id] = initial_progress
                self.logger.info(f"Task {task_id}: {initial_progress.current_step}")

                # Start background processing
                asyncio.create_task(
                    self.process_pdf_background(
                        task_id, temp_file_path, actual_doc_name
                    )
                )

                return {
                    "task_id": task_id,
                    "status": "started",
                    "message": f"Processing {actual_doc_name} in background",
                }

            except Exception as e:
                self.logger.error(
                    f"Task {task_id}: FAILED during setup - {str(e)}", exc_info=True
                )
                error_progress = StreamingProgress(
                    task_id=task_id,
                    progress=0.0,
                    current_step="Ingestion failed",
                    step_num=0,
                    total_steps=1,
                    error=str(e),
                )
                self.latest_progress[task_id] = error_progress
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/documents")
        async def list_documents():
            """List all documents in the database"""
            try:
                await self.db_manager.initialize()

                if self.db_manager.table is None:
                    return {"documents": [], "total": 0}

                # Get all documents from the database using LanceDB query
                try:
                    # Use LanceDB's pandas conversion for more reliable results
                    all_docs = self.db_manager.table.to_pandas().to_dict("records")

                    if not all_docs:
                        return {"documents": [], "total": 0}

                    # Group by document name manually
                    docs_by_name = {}
                    for doc in all_docs:
                        doc_name = doc.get("doc_name", "Unknown")
                        if doc_name not in docs_by_name:
                            docs_by_name[doc_name] = {
                                "pages": 0,
                                "max_page": 0,
                                "embeddings_count": 0,
                                "created_at": doc.get("created_at", "Unknown"),
                            }

                        docs_by_name[doc_name]["embeddings_count"] += 1
                        page_num = doc.get("page_num", 0)
                        if page_num > docs_by_name[doc_name]["max_page"]:
                            docs_by_name[doc_name]["max_page"] = page_num

                    # Format response
                    documents = []
                    for doc_name, stats in docs_by_name.items():
                        documents.append(
                            {
                                "id": f"doc_{hash(doc_name) % 10000}",
                                "name": doc_name,
                                "pages": stats["max_page"],
                                "embeddings_count": stats["embeddings_count"],
                                "created_date": stats["created_at"][:10]
                                if stats["created_at"] != "Unknown"
                                else "2024-07-25",
                                "size": "N/A",
                            }
                        )

                    return {"documents": documents, "total": len(documents)}

                except Exception as table_error:
                    self.logger.warning(f"Could not read table data: {table_error}")
                    return {
                        "documents": [],
                        "total": 0,
                        "info": "No documents indexed yet",
                    }

            except Exception as e:
                self.logger.error(f"Error listing documents: {str(e)}")
                return {"documents": [], "total": 0, "error": str(e)}

        @self.app.post("/search")
        async def search_documents(request: SearchRequest):
            """Search documents"""
            task_id = f"search_{uuid.uuid4().hex[:8]}"
            self.logger.info(
                f"Starting search task {task_id} for query: '{request.query}' (top_k={request.top_k})"
            )

            # Initial progress
            initial_progress = StreamingProgress(
                task_id=task_id,
                progress=5.0,
                current_step="Search initiated",
                step_num=1,
                total_steps=4,
                details=f"Query: '{request.query[:50]}...'",
            )
            self.latest_progress[task_id] = initial_progress
            self.logger.info(f"Task {task_id}: {initial_progress.current_step}")

            # Start background processing
            asyncio.create_task(self.process_search_background(task_id, request))

            return {
                "task_id": task_id,
                "status": "started",
                "message": f"Processing search for '{request.query}'",
            }

        @self.app.get("/progress/{task_id}")
        async def get_task_progress(task_id: str):
            """Get latest progress for a task"""
            if task_id in self.latest_progress:
                progress = self.latest_progress[task_id]
                self.logger.debug(
                    f"Progress requested for task {task_id}: {progress.current_step} ({progress.progress:.1f}%)"
                )
                return {
                    "task_id": task_id,
                    "progress": progress.to_dict(),
                }
            else:
                self.logger.warning(f"Progress requested for unknown task: {task_id}")
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        @self.app.get("/tasks")
        async def list_active_tasks():
            """List all active tasks"""
            return {
                "active_tasks": [
                    {
                        "task_id": task_id,
                        "current_step": progress.current_step,
                        "progress": progress.progress,
                    }
                    for task_id, progress in self.latest_progress.items()
                    if progress.progress < 100.0 and not progress.error
                ]
            }

        @self.app.delete("/documents/{doc_name}")
        async def delete_document(doc_name: str):
            """Delete a document and all its embeddings"""
            task_id = f"delete_{uuid.uuid4().hex[:8]}"
            self.logger.info(
                f"Starting document deletion task {task_id} for: {doc_name}"
            )

            try:
                # Initialize database if needed
                if self.db_manager.db is None:
                    await self.db_manager.initialize()

                if self.db_manager.table is None:
                    self.logger.error(f"Task {task_id}: No documents table found")
                    raise HTTPException(
                        status_code=404, detail="No documents table found"
                    )

                # Check if document exists
                try:
                    # Query to find documents with this name
                    all_docs = self.db_manager.table.to_pandas()
                    doc_rows = all_docs[all_docs["doc_name"] == doc_name]

                    if len(doc_rows) == 0:
                        self.logger.warning(
                            f"Task {task_id}: Document '{doc_name}' not found"
                        )
                        raise HTTPException(
                            status_code=404, detail=f"Document '{doc_name}' not found"
                        )

                    self.logger.info(
                        f"Task {task_id}: Found {len(doc_rows)} embeddings to delete for '{doc_name}'"
                    )

                    # Delete rows with this document name
                    # LanceDB doesn't have a direct delete by filter, so we need to filter and recreate
                    remaining_docs = all_docs[all_docs["doc_name"] != doc_name]

                    if len(remaining_docs) == 0:
                        # If no documents remain, drop the table entirely
                        try:
                            self.db_manager.db.drop_table("documents")
                            self.db_manager.table = None
                            self.logger.info(
                                f"Task {task_id}: Dropped empty documents table"
                            )
                        except Exception as drop_error:
                            self.logger.warning(
                                f"Task {task_id}: Could not drop table: {drop_error}"
                            )
                    else:
                        # Recreate table with remaining data
                        try:
                            # Convert back to records format
                            remaining_data = remaining_docs.to_dict("records")

                            # Drop old table and create new one
                            self.db_manager.db.drop_table("documents")
                            self.db_manager.table = self.db_manager.db.create_table(
                                "documents", remaining_data
                            )

                            self.logger.info(
                                f"Task {task_id}: Recreated table with {len(remaining_data)} remaining embeddings"
                            )
                        except Exception as recreate_error:
                            self.logger.error(
                                f"Task {task_id}: Failed to recreate table: {recreate_error}"
                            )
                            raise HTTPException(
                                status_code=500,
                                detail=f"Failed to recreate table: {recreate_error}",
                            )

                    deleted_count = len(doc_rows)
                    remaining_count = len(remaining_docs)

                    self.logger.info(
                        f"Task {task_id}: Successfully deleted {deleted_count} embeddings for '{doc_name}', {remaining_count} embeddings remain"
                    )

                    return {
                        "task_id": task_id,
                        "status": "completed",
                        "message": f"Document '{doc_name}' deleted successfully",
                        "deleted_embeddings": deleted_count,
                        "remaining_embeddings": remaining_count,
                    }

                except Exception as query_error:
                    self.logger.error(
                        f"Task {task_id}: Database query error: {query_error}"
                    )
                    raise HTTPException(
                        status_code=500, detail=f"Database error: {query_error}"
                    )

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(
                    f"Task {task_id}: Deletion failed: {str(e)}", exc_info=True
                )
                raise HTTPException(
                    status_code=500, detail=f"Deletion failed: {str(e)}"
                )

        @self.app.get("/logs")
        async def get_recent_logs():
            """Get recent server logs"""
            try:
                log_file_path = Path("colpali_server.log")
                if log_file_path.exists():
                    # Read last 100 lines of log file
                    with open(log_file_path, "r") as f:
                        lines = f.readlines()
                        recent_lines = lines[-100:] if len(lines) > 100 else lines

                    return {
                        "logs": [
                            {
                                "timestamp": line.split(" - ")[0]
                                if " - " in line
                                else "",
                                "level": line.split(" - ")[2]
                                if len(line.split(" - ")) > 2
                                else "INFO",
                                "message": " - ".join(line.split(" - ")[3:]).strip()
                                if len(line.split(" - ")) > 3
                                else line.strip(),
                            }
                            for line in recent_lines
                            if line.strip()
                        ][-50:]  # Last 50 log entries
                    }
                else:
                    return {"logs": []}
            except Exception as e:
                self.logger.error(f"Failed to read log file: {str(e)}")
                return {"logs": [], "error": "Failed to read server logs"}

    async def start_server(self, host: str = "localhost", port: int = 8000):
        """Start the HTTP server"""
        try:
            await self.db_manager.initialize()
            self.logger.info(f"LanceDB initialized at: {self.db_manager.db_path}")

            self.logger.info(f"🚀 Starting ColPali HTTP Server on {host}:{port}")
            self.logger.info(f"📊 Server health endpoint: http://{host}:{port}/health")
            self.logger.info(f"📝 Server logs being written to: colpali_server.log")

            config = uvicorn.Config(
                self.app,
                host=host,
                port=port,
                log_level="info",
                access_log=True,
            )
            server = uvicorn.Server(config)
            await server.serve()

        except Exception as e:
            self.logger.error(f"Failed to start server: {str(e)}", exc_info=True)
            raise

    async def process_pdf_background(
        self, task_id: str, temp_file_path: str, actual_doc_name: str
    ):
        """Background processing of PDF with real-time progress updates"""
        try:
            # Ensure model is loaded
            if not self.model_manager.model_loaded:
                self.logger.info(
                    f"Task {task_id}: Model not loaded, initializing automatically..."
                )
                async for progress in self.model_manager.load_model():
                    progress.task_id = task_id
                    progress.step_num = 2
                    progress.total_steps = 6
                    # Adjust progress to fit within step 2's range (5-35%)
                    progress.progress = 5.0 + (progress.progress / 100.0) * 30.0
                    progress.current_step = (
                        f"Loading ColPali model: {progress.current_step}"
                    )
                    self.latest_progress[task_id] = progress
                    self.logger.info(
                        f"Task {task_id}: Model loading - {progress.current_step} ({progress.progress:.1f}%)"
                    )
                    await asyncio.sleep(0.1)  # Allow other tasks to run

            # Encode pages - CAPTURE THE ACTUAL DATA
            self.logger.info(f"Task {task_id}: Starting ColPali page encoding")

            # Extract PDF pages directly
            doc = fitz.open(temp_file_path)
            total_pages = len(doc)

            self.logger.info(
                f"Task {task_id}: Processing all {total_pages} pages from PDF"
            )

            images = []
            metadata = []

            for page_num in range(total_pages):
                page = doc[page_num]

                # Use lower resolution for memory savings (adjust zoom as needed)
                zoom_factor = 1.5  # Reduced from 2.0
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))

                # Optionally resize image if too large
                max_size = (1024, 1024)  # Adjust as needed
                if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)

                images.append(image)

                # Extract text with multiple methods for better coverage
                try:
                    # Method 1: Standard text extraction
                    text_content = page.get_text()
                    
                    # Method 2: If no text found, try extracting text blocks
                    if not text_content or len(text_content.strip()) < 10:
                        text_blocks = page.get_text("blocks")
                        text_content = "\n".join([block[4] for block in text_blocks if len(block) > 4])
                    
                    # Method 3: If still no text, try dictionary method
                    if not text_content or len(text_content.strip()) < 10:
                        text_dict = page.get_text("dict")
                        extracted_text = []
                        for block in text_dict.get("blocks", []):
                            if "lines" in block:
                                for line in block["lines"]:
                                    for span in line.get("spans", []):
                                        if "text" in span:
                                            extracted_text.append(span["text"])
                        text_content = " ".join(extracted_text)
                    
                    # Clean up the text
                    text_content = text_content.strip()
                    if not text_content:
                        text_content = f"[No extractable text found on page {page_num + 1}]"
                    
                    self.logger.info(f"Task {task_id}: Page {page_num + 1} extracted {len(text_content)} characters of text")
                    
                except Exception as text_error:
                    self.logger.error(f"Task {task_id}: Text extraction failed for page {page_num + 1}: {text_error}")
                    text_content = f"[Text extraction failed for page {page_num + 1}]"

                # Save page image to disk
                image_path = self.save_page_image(image, actual_doc_name, page_num + 1)

                metadata.append(
                    {
                        "page_num": page_num + 1,
                        "doc_name": actual_doc_name,
                        "text_content": text_content,
                        "file_size": len(img_data),
                        "image_path": image_path,  # Store the saved image path
                    }
                )

                # Update progress
                page_progress = 20.0 + ((page_num + 1) / total_pages) * 20.0
                current_progress = StreamingProgress(
                    task_id=task_id,
                    progress=page_progress,
                    current_step=f"Extracted page {page_num + 1}/{total_pages}",
                    step_num=3,
                    total_steps=6,
                    details=f"Image: {image.size}, Text: {len(text_content)} chars (extracted with robust method)",
                )
                self.latest_progress[task_id] = current_progress
                await asyncio.sleep(0.05)

            doc.close()

            # Memory cleanup after PDF processing
            self.model_manager.cleanup_memory()

            # The encode_pages method is now memory-optimized and returns embeddings as a generator
            # We need to collect the actual embeddings
            self.logger.info(
                f"Task {task_id}: Starting memory-optimized ColPali encoding"
            )

            # Process images in very small batches with the memory-safe approach
            embeddings = []
            batch_size = 1  # Very conservative for memory safety

            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]

                # Update progress
                encode_progress = 40.0 + ((i + 1) / len(images)) * 30.0
                progress_update = StreamingProgress(
                    task_id=task_id,
                    progress=encode_progress,
                    current_step=f"Encoding page {i + 1}/{len(images)} with memory optimization",
                    step_num=4,
                    total_steps=6,
                    details=f"Memory-safe batch processing",
                )
                self.latest_progress[task_id] = progress_update

                try:
                    # Process with processor
                    batch_inputs = self.model_manager.processor.process_images(
                        batch_images
                    )

                    # Move to device
                    device_obj = self.model_manager._get_device_obj()
                    for key in batch_inputs:
                        if isinstance(batch_inputs[key], torch.Tensor):
                            batch_inputs[key] = batch_inputs[key].to(device_obj)

                    # Get embeddings with memory management
                    with torch.no_grad():
                        batch_embeddings = self.model_manager.model(**batch_inputs)

                    # Immediately move to CPU and store
                    for emb in batch_embeddings:
                        embeddings.append(emb.cpu().detach())

                    # Clean up immediately after each batch
                    del batch_embeddings, batch_inputs
                    self.model_manager.cleanup_memory()

                except Exception as e:
                    self.logger.error(
                        f"Task {task_id}: Encoding error for batch {i}: {e}"
                    )
                    # Continue with next batch rather than failing completely
                    continue

                # Brief pause for memory recovery
                await asyncio.sleep(0.1)

            self.logger.info(
                f"Task {task_id}: Generated {len(embeddings)} embeddings successfully"
            )

            # Store in database
            self.logger.info(f"Task {task_id}: Storing embeddings in LanceDB")
            async for progress in self.db_manager.store_embeddings(
                embeddings, metadata
            ):
                progress.task_id = task_id
                progress.step_num = 5
                progress.total_steps = 6
                progress.progress = 70.0 + (progress.progress / 100.0) * 25.0
                self.latest_progress[task_id] = progress
                await asyncio.sleep(0.1)

            # Final completion
            final_progress = StreamingProgress(
                task_id=task_id,
                progress=100.0,
                current_step="Ingestion completed successfully",
                step_num=6,
                total_steps=6,
                details=f"Document '{actual_doc_name}' ready for search - {len(embeddings)} pages indexed",
            )
            self.latest_progress[task_id] = final_progress
            self.logger.info(f"Task {task_id}: COMPLETED successfully")

        except Exception as e:
            self.logger.error(f"Task {task_id}: FAILED - {str(e)}", exc_info=True)
            error_progress = StreamingProgress(
                task_id=task_id,
                progress=0.0,
                current_step="Processing failed",
                step_num=0,
                total_steps=1,
                error=str(e),
            )
            self.latest_progress[task_id] = error_progress
        finally:
            # Clean up
            try:
                Path(temp_file_path).unlink(missing_ok=True)
                self.model_manager.cleanup_memory()
            except Exception as cleanup_error:
                self.logger.warning(f"Cleanup error: {cleanup_error}")

    async def process_search_background(self, task_id: str, request: SearchRequest):
        """Background processing of search with real-time progress updates"""
        try:
            self.logger.info(
                f"Task {task_id}: Starting search background processing for query: '{request.query}'"
            )

            # Check database manager initialization
            self.logger.info(
                f"Task {task_id}: Database manager exists: {hasattr(self, 'db_manager')}"
            )
            if hasattr(self, "db_manager"):
                self.logger.info(
                    f"Task {task_id}: Database connection: {self.db_manager.db is not None}"
                )
                self.logger.info(
                    f"Task {task_id}: Table connection: {self.db_manager.table is not None}"
                )

            # Initialize database if needed
            if not hasattr(self, "db_manager") or self.db_manager.db is None:
                self.logger.info(f"Task {task_id}: Initializing database manager...")
                await self.db_manager.initialize()

            # Check if ColPali model is loaded, if not, load it automatically
            if not self.model_manager.model_loaded:
                self.logger.info(
                    f"Task {task_id}: ColPali model not loaded, initializing automatically..."
                )
                async for progress in self.model_manager.load_model():
                    progress.task_id = task_id
                    progress.step_num = 1
                    progress.total_steps = 5  # Updated total steps
                    progress.progress = (
                        progress.progress * 0.3
                    )  # Take 30% of progress for model loading
                    progress.current_step = (
                        f"Loading ColPali model: {progress.current_step}"
                    )
                    self.latest_progress[task_id] = progress
                    self.logger.info(
                        f"Task {task_id}: Model loading - {progress.current_step} ({progress.progress:.1f}%)"
                    )
                    await asyncio.sleep(0.1)

            # Check if we have any documents indexed
            if self.db_manager.table is None:
                self.logger.warning(f"Task {task_id}: No documents table found")
                error_progress = StreamingProgress(
                    task_id=task_id,
                    progress=0.0,
                    current_step="No documents available",
                    step_num=0,
                    total_steps=1,
                    error="No documents have been indexed yet. Please ingest documents first.",
                )
                self.latest_progress[task_id] = error_progress
                return

            # Additional check: verify table has data
            try:
                row_count = len(self.db_manager.table)
                self.logger.info(
                    f"Task {task_id}: Table contains {row_count} rows before search"
                )
                if row_count == 0:
                    self.logger.warning(f"Task {task_id}: Table exists but is empty")
                    error_progress = StreamingProgress(
                        task_id=task_id,
                        progress=100.0,
                        current_step="Search completed",
                        step_num=5,
                        total_steps=5,
                        details="No documents indexed in database",
                        results=[],
                    )
                    self.latest_progress[task_id] = error_progress
                    return
            except Exception as count_error:
                self.logger.error(
                    f"Task {task_id}: Could not count table rows: {count_error}"
                )

            # Encode query
            self.logger.info(
                f"Task {task_id}: Encoding search query: '{request.query}'"
            )
            query_embedding = None
            async for progress in self.model_manager.encode_query(request.query):
                progress.task_id = task_id
                progress.step_num = 2
                progress.total_steps = 5  # Updated total steps
                progress.progress = 30.0 + (progress.progress / 100.0) * 25.0  # 30-55%
                self.latest_progress[task_id] = progress
                self.logger.info(
                    f"Task {task_id}: Query encoding - {progress.current_step} ({progress.progress:.1f}%)"
                )
                await asyncio.sleep(0.1)

            # Get the actual query embedding
            self.logger.info(f"Task {task_id}: Getting query embedding from model")
            query_embedding = await self.model_manager.encode_query_simple(
                request.query
            )
            self.logger.info(
                f"Task {task_id}: Query embedding generated - type: {type(query_embedding)}, shape: {getattr(query_embedding, 'shape', 'no shape')}"
            )

            # Search database
            self.logger.info(
                f"Task {task_id}: Performing vector similarity search with top_k={request.top_k}"
            )
            search_results = None
            progress_count = 0
            async for progress in self.db_manager.search_embeddings(
                query_embedding, request.top_k
            ):
                progress_count += 1
                self.logger.info(
                    f"Task {task_id}: Search progress update #{progress_count}: {progress.current_step}"
                )

                # Update progress to fit within the overall search workflow
                progress.task_id = task_id
                progress.step_num = 3
                progress.total_steps = 5
                progress.progress = 55.0 + (progress.progress / 100.0) * 40.0  # 55-95%
                self.latest_progress[task_id] = progress
                self.logger.info(
                    f"Task {task_id}: Vector search - {progress.current_step} ({progress.progress:.1f}%)"
                )

                # Capture results from the final progress update
                if progress.results is not None:
                    search_results = progress.results
                    self.logger.info(
                        f"Task {task_id}: Captured search results: {len(search_results)} items"
                    )

                # Log errors if any
                if progress.error:
                    self.logger.error(f"Task {task_id}: Search error: {progress.error}")

                await asyncio.sleep(0.1)

            # Process search results (already converted to dicts in search_embeddings)
            results = search_results if search_results else []
            self.logger.info(
                f"Task {task_id}: Final processed results count: {len(results)}"
            )

            # Log details of results if any
            if results:
                for i, result in enumerate(results[:3]):  # Log first 3 results
                    self.logger.info(f"Task {task_id}: Final result {i}: {result}")
            else:
                self.logger.warning(f"Task {task_id}: No final results to return")

            # Final completion
            final_progress = StreamingProgress(
                task_id=task_id,
                progress=100.0,
                current_step="Search completed",
                step_num=5,
                total_steps=5,
                details=f"Found {len(results)} relevant matches",
                results=results,  # Store actual results
            )
            self.latest_progress[task_id] = final_progress

            self.logger.info(
                f"Task {task_id}: COMPLETED - Found {len(results)} results for query '{request.query}'"
            )

        except Exception as e:
            self.logger.error(f"Task {task_id}: FAILED - {str(e)}", exc_info=True)
            error_progress = StreamingProgress(
                task_id=task_id,
                progress=0.0,
                current_step="Search failed",
                step_num=0,
                total_steps=1,
                error=str(e),
            )
            self.latest_progress[task_id] = error_progress


# Global server instance
colpali_server = ColPaliHTTPServer()


async def main():
    """Main entry point"""
    await colpali_server.start_server()


if __name__ == "__main__":
    asyncio.run(main())
