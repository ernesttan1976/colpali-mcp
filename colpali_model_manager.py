#!/usr/bin/env python3
"""
ColPali Long-Running HTTP Server
Supports Apple Silicon M4 with MPS acceleration
"""

import asyncio
import logging
import time
import uuid
import gc  # For garbage collection
import psutil  # For memory monitoring
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, AsyncGenerator

import torch
from PIL import Image


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
