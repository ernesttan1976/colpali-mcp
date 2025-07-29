#!/usr/bin/env python3
"""
ColPali FastMCP HTTP Server
Standalone server that runs on HTTP port
"""

import asyncio
import json
import logging
import time
import uuid
import sys
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Any

import torch
import lancedb
from PIL import Image
from fastmcp import FastMCP

# Mock ColPali imports (replace with actual imports when available)
# from colpali_engine import ColPaliModel, ColPaliProcessor


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

    def to_dict(self):
        return asdict(self)


@dataclass
class SearchResult:
    page_num: int
    doc_name: str
    score: float
    snippet: str
    image_path: Optional[str] = None


class ColPaliModelManager:
    """Handles ColPali model loading and inference"""

    def __init__(self, device: str = "mps"):
        self.device = device
        self.model = None
        self.processor = None
        self.model_loaded = False

    async def load_model(self) -> AsyncGenerator[StreamingProgress, None]:
        """Load ColPali model with streaming progress"""
        task_id = "model_load"

        yield StreamingProgress(
            task_id=task_id,
            progress=0.0,
            current_step="Initializing model loading",
            step_num=1,
            total_steps=3,
            details="Checking device availability",
        )

        # Simulate model loading delay
        await asyncio.sleep(1)

        yield StreamingProgress(
            task_id=task_id,
            progress=30.0,
            current_step="Loading ColPali processor",
            step_num=2,
            total_steps=3,
            details="Loading tokenizer and image processor",
        )

        # Mock processor loading
        await asyncio.sleep(2)
        # self.processor = ColPaliProcessor.from_pretrained("vidore/colpali")

        yield StreamingProgress(
            task_id=task_id,
            progress=70.0,
            current_step="Loading ColPali model weights",
            step_num=3,
            total_steps=3,
            details=f"Loading to {self.device} device",
        )

        # Mock model loading
        await asyncio.sleep(3)
        # self.model = ColPaliModel.from_pretrained("vidore/colpali")
        # self.model.to(self.device)

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

    async def encode_pages(
        self, images: List[Image.Image]
    ) -> AsyncGenerator[StreamingProgress, None]:
        """Encode PDF pages with streaming progress"""
        task_id = f"encode_{uuid.uuid4().hex[:8]}"
        total_pages = len(images)
        start_time = time.time()

        yield StreamingProgress(
            task_id=task_id,
            progress=0.0,
            current_step="Starting page encoding",
            step_num=1,
            total_steps=total_pages,
            details=f"Processing {total_pages} pages",
        )

        embeddings = []

        for i, image in enumerate(images):
            current_progress = (i / total_pages) * 100
            elapsed = time.time() - start_time
            pages_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total_pages - i - 1) / pages_per_sec if pages_per_sec > 0 else None

            yield StreamingProgress(
                task_id=task_id,
                progress=current_progress,
                current_step=f"Encoding page {i + 1}/{total_pages}",
                step_num=i + 1,
                total_steps=total_pages,
                details=f"Processing image {image.size}",
                eta_seconds=int(eta) if eta else None,
                throughput=f"{pages_per_sec:.1f} pages/sec",
            )

            # Simulate ColPali encoding
            await asyncio.sleep(0.5)  # Mock processing time

            # Mock embedding generation
            # embedding = self.model.encode_image(image)
            embedding = torch.randn(1, 128)  # Mock embedding
            embeddings.append(embedding)

        yield StreamingProgress(
            task_id=task_id,
            progress=100.0,
            current_step="Page encoding complete",
            step_num=total_pages,
            total_steps=total_pages,
            details=f"Generated {len(embeddings)} embeddings",
            throughput=f"Average: {total_pages / elapsed:.1f} pages/sec",
        )

        # Final yield with the actual embeddings
        yield {"embeddings": embeddings}

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

        await asyncio.sleep(0.5)

        yield StreamingProgress(
            task_id=task_id,
            progress=40.0,
            current_step="Tokenizing query",
            step_num=2,
            total_steps=3,
            details="Converting text to tokens",
        )

        await asyncio.sleep(1.0)

        yield StreamingProgress(
            task_id=task_id,
            progress=80.0,
            current_step="Generating embedding",
            step_num=3,
            total_steps=3,
            details="ColPali query encoding",
        )

        await asyncio.sleep(1.5)

        # Mock query embedding
        # query_embedding = self.model.encode_query(query)
        query_embedding = torch.randn(1, 128)

        yield StreamingProgress(
            task_id=task_id,
            progress=100.0,
            current_step="Query encoded successfully",
            step_num=3,
            total_steps=3,
            details="Ready for similarity search",
        )

        # Final yield with the query embedding
        yield {"query_embedding": query_embedding}


class LanceDBManager:
    """Handles LanceDB operations"""

    def __init__(self, db_path: Optional[str] = None):
        # Use environment variable or default to the data directory
        if db_path is None:
            db_path = os.getenv(
                "COLPALI_DB_PATH", "/Volumes/myssd/colpali-mcp/data/embeddings_db"
            )

        self.db_path = db_path
        self.db = None
        self.table = None

        # Ensure the database directory exists and is writable
        try:
            os.makedirs(self.db_path, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(self.db_path, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print(f"Database directory is writable: {self.db_path}", file=sys.stderr)
        except (OSError, PermissionError) as e:
            raise RuntimeError(
                f"Cannot create or write to database directory {db_path}: {e}"
            )

        print(f"LanceDBManager initialized with path: {self.db_path}", file=sys.stderr)

    async def initialize(self):
        """Initialize LanceDB connection"""
        try:
            self.db = lancedb.connect(self.db_path)
            print(
                f"Successfully connected to LanceDB at {self.db_path}", file=sys.stderr
            )
        except Exception as e:
            print(f"Failed to connect to LanceDB: {e}", file=sys.stderr)
            raise

    async def store_embeddings(
        self, embeddings: List[torch.Tensor], metadata: List[Dict]
    ) -> AsyncGenerator[StreamingProgress, None]:
        """Store embeddings with streaming progress"""
        task_id = f"store_{uuid.uuid4().hex[:8]}"
        total_embeddings = len(embeddings)

        yield StreamingProgress(
            task_id=task_id,
            progress=0.0,
            current_step="Preparing data for storage",
            step_num=1,
            total_steps=3,
            details=f"Processing {total_embeddings} embeddings",
        )

        # Prepare data
        data = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            data.append(
                {
                    "id": f"{meta['doc_name']}_page_{meta['page_num']}",
                    "embedding": embedding.numpy().flatten().tolist(),
                    "doc_name": meta["doc_name"],
                    "page_num": meta["page_num"],
                    "text_content": meta.get("text_content", ""),
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

        # Create or update table
        try:
            self.table = self.db.open_table("documents")
        except:
            if data:
                self.table = self.db.create_table("documents", data[:1])

        yield StreamingProgress(
            task_id=task_id,
            progress=60.0,
            current_step="Inserting embeddings",
            step_num=3,
            total_steps=3,
            details=f"Batch inserting {len(data)} records",
        )

        await asyncio.sleep(2)

        # Insert data (mock)
        # self.table.add(data)

        yield StreamingProgress(
            task_id=task_id,
            progress=100.0,
            current_step="Storage complete",
            step_num=3,
            total_steps=3,
            details=f"Stored {total_embeddings} embeddings successfully",
        )

    async def search_embeddings(
        self, query_embedding: torch.Tensor, top_k: int = 5
    ) -> AsyncGenerator[StreamingProgress, None]:
        """Search embeddings with streaming progress"""
        task_id = f"search_{uuid.uuid4().hex[:8]}"

        yield StreamingProgress(
            task_id=task_id,
            progress=0.0,
            current_step="Preparing vector search",
            step_num=1,
            total_steps=4,
            details=f"Searching for top {top_k} results",
        )

        await asyncio.sleep(0.5)

        yield StreamingProgress(
            task_id=task_id,
            progress=25.0,
            current_step="Computing similarities",
            step_num=2,
            total_steps=4,
            details="Running vector similarity search",
        )

        await asyncio.sleep(1.5)

        yield StreamingProgress(
            task_id=task_id,
            progress=60.0,
            current_step="Ranking results",
            step_num=3,
            total_steps=4,
            details="Sorting by similarity score",
        )

        await asyncio.sleep(0.8)

        yield StreamingProgress(
            task_id=task_id,
            progress=90.0,
            current_step="Formatting results",
            step_num=4,
            total_steps=4,
            details="Preparing response data",
        )

        await asyncio.sleep(0.3)

        # Mock search results
        mock_results = [
            SearchResult(
                page_num=i + 1,
                doc_name="sample_document.pdf",
                score=0.95 - i * 0.1,
                snippet=f"Sample content from page {i + 1}...",
                image_path=f"page_{i + 1}.png",
            )
            for i in range(min(top_k, 3))
        ]

        yield StreamingProgress(
            task_id=task_id,
            progress=100.0,
            current_step="Search complete",
            step_num=4,
            total_steps=4,
            details=f"Found {len(mock_results)} relevant results",
        )

        # Final yield with search results
        yield {"results": mock_results}


class PDFProcessor:
    """Handles PDF processing and page extraction"""

    @staticmethod
    async def extract_pages(file_path: str) -> AsyncGenerator[StreamingProgress, None]:
        """Extract pages from PDF with streaming progress (MOCK VERSION)"""
        task_id = f"extract_{uuid.uuid4().hex[:8]}"

        yield StreamingProgress(
            task_id=task_id,
            progress=0.0,
            current_step="Opening PDF file (MOCK)",
            step_num=1,
            total_steps=3,
            details=f"Mock processing {file_path}",
        )

        await asyncio.sleep(0.5)

        # Mock PDF processing - simulate 10 pages
        total_pages = 10

        yield StreamingProgress(
            task_id=task_id,
            progress=20.0,
            current_step=f"Processing {total_pages} pages (MOCK)",
            step_num=2,
            total_steps=3,
            details="Creating mock images and text",
        )

        images = []
        metadata = []

        for page_num in range(total_pages):
            # Create mock image
            image = Image.new("RGB", (800, 600), color=(200, 200, 200))
            images.append(image)

            # Create mock metadata
            metadata.append(
                {
                    "page_num": page_num + 1,
                    "doc_name": Path(file_path).stem,
                    "text_content": f"Mock content for page {page_num + 1} from {Path(file_path).name}",
                }
            )

            # Update progress
            page_progress = 20.0 + (page_num + 1) / total_pages * 70.0
            yield StreamingProgress(
                task_id=task_id,
                progress=page_progress,
                current_step=f"Mock extracted page {page_num + 1}/{total_pages}",
                step_num=2,
                total_steps=3,
                details=f"Mock Image: {image.size}, Mock Text: {len(f'Mock content for page {page_num + 1}')} chars",
            )

            await asyncio.sleep(0.1)  # Simulate processing time

        yield StreamingProgress(
            task_id=task_id,
            progress=100.0,
            current_step="Mock PDF extraction complete",
            step_num=3,
            total_steps=3,
            details=f"Mock extracted {len(images)} pages successfully",
        )

        # Final yield with extracted data
        yield {"images": images, "metadata": metadata}


# Initialize FastMCP
mcp = FastMCP("ColPali Streaming Server")

# Global components
model_manager = ColPaliModelManager()
db_manager = LanceDBManager()
pdf_processor = PDFProcessor()
active_tasks: Dict[str, AsyncGenerator] = {}
latest_progress: Dict[str, StreamingProgress] = {}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@mcp.tool()
async def test_connection() -> dict:
    """Test the ColPali server connection"""
    return {
        "status": "success",
        "message": "ColPali FastMCP Server is running!",
        "server_info": {
            "name": "colpali-streaming",
            "version": "1.0.0",
            "database_path": db_manager.db_path,
            "port": "8000",
        },
    }


@mcp.tool()
async def initialize_colpali() -> dict:
    """Initialize ColPali model with progress tracking"""
    task_id = f"init_{uuid.uuid4().hex[:8]}"
    progress_updates = []

    try:
        async for progress in model_manager.load_model():
            latest_progress[task_id] = progress
            progress_updates.append(progress.to_dict())

        return {
            "task_id": task_id,
            "status": "completed",
            "final_progress": progress_updates[-1] if progress_updates else None,
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
        latest_progress[task_id] = error_progress
        return {"task_id": task_id, "status": "failed", "error": str(e)}


@mcp.tool()
async def ingest_pdf_stream(file_path: str, doc_name: Optional[str] = None) -> dict:
    """Ingest PDF with real-time progress streaming"""
    task_id = f"ingest_{uuid.uuid4().hex[:8]}"
    doc_name = doc_name or Path(file_path).stem

    try:
        # Ensure model is loaded
        if not model_manager.model_loaded:
            async for progress in model_manager.load_model():
                latest_progress[task_id] = progress

        # Extract PDF pages
        extracted_data = None
        async for progress_or_data in pdf_processor.extract_pages(file_path):
            if isinstance(progress_or_data, StreamingProgress):
                progress_or_data.task_id = task_id
                latest_progress[task_id] = progress_or_data
            else:
                extracted_data = progress_or_data

        if not extracted_data:
            # Use mock data for demo
            mock_images = [Image.new("RGB", (800, 600)) for _ in range(3)]
            mock_metadata = [
                {
                    "page_num": i + 1,
                    "doc_name": doc_name,
                    "text_content": f"Page {i + 1} content",
                }
                for i in range(3)
            ]
        else:
            mock_images = extracted_data.get("images", [])
            mock_metadata = extracted_data.get("metadata", [])

        # Encode pages
        embeddings = None
        async for progress_or_data in model_manager.encode_pages(mock_images):
            if isinstance(progress_or_data, StreamingProgress):
                progress_or_data.task_id = task_id
                latest_progress[task_id] = progress_or_data
            else:
                embeddings = progress_or_data.get("embeddings", [])

        # Store in database
        async for progress in db_manager.store_embeddings(
            embeddings or [torch.randn(1, 128) for _ in range(3)], mock_metadata
        ):
            progress.task_id = task_id
            latest_progress[task_id] = progress

        return {
            "task_id": task_id,
            "status": "completed",
            "message": f"Successfully ingested {doc_name}",
            "pages_processed": len(mock_images),
        }

    except Exception as e:
        error_progress = StreamingProgress(
            task_id=task_id,
            progress=0.0,
            current_step="Ingestion failed",
            step_num=0,
            total_steps=1,
            error=str(e),
        )
        latest_progress[task_id] = error_progress
        return {"task_id": task_id, "status": "failed", "error": str(e)}


@mcp.tool()
async def search_documents_stream(query: str, top_k: int = 5) -> dict:
    """Search documents with real-time progress updates"""
    task_id = f"search_{uuid.uuid4().hex[:8]}"

    try:
        # Encode query
        query_embedding = None
        async for progress_or_data in model_manager.encode_query(query):
            if isinstance(progress_or_data, StreamingProgress):
                progress_or_data.task_id = task_id
                latest_progress[task_id] = progress_or_data
            else:
                query_embedding = progress_or_data.get("query_embedding")

        # Search database
        results = None
        async for progress_or_data in db_manager.search_embeddings(
            query_embedding or torch.randn(1, 128), top_k
        ):
            if isinstance(progress_or_data, StreamingProgress):
                progress_or_data.task_id = task_id
                latest_progress[task_id] = progress_or_data
            else:
                results = progress_or_data.get("results", [])

        # Format results for response
        formatted_results = [
            {
                "page_num": result.page_num,
                "doc_name": result.doc_name,
                "score": result.score,
                "snippet": result.snippet,
            }
            for result in (results or [])
        ]

        return {
            "task_id": task_id,
            "status": "completed",
            "results": formatted_results,
            "message": f"Found {len(formatted_results)} relevant results",
        }

    except Exception as e:
        error_progress = StreamingProgress(
            task_id=task_id,
            progress=0.0,
            current_step="Search failed",
            step_num=0,
            total_steps=1,
            error=str(e),
        )
        latest_progress[task_id] = error_progress
        return {"task_id": task_id, "status": "failed", "error": str(e)}


@mcp.tool()
async def get_task_progress(task_id: str) -> dict:
    """Get latest progress for a streaming task"""
    if task_id in latest_progress:
        return {
            "task_id": task_id,
            "progress": latest_progress[task_id].to_dict(),
        }
    else:
        return {"error": f"Task {task_id} not found"}


@mcp.tool()
async def list_active_tasks() -> dict:
    """List all currently active streaming tasks"""
    return {
        "active_tasks": [
            {
                "task_id": task_id,
                "current_step": progress.current_step,
                "progress": progress.progress,
            }
            for task_id, progress in latest_progress.items()
            if progress.progress < 100.0 and not progress.error
        ]
    }


@mcp.tool()
async def health_check() -> dict:
    """Check server health and database status"""
    try:
        # Import and use the database class
        sys.path.append("/Volumes/myssd/colpali-mcp")
        from db import DocumentEmbeddingDatabase

        # Initialize database and run health check
        db = DocumentEmbeddingDatabase()
        health = db.health_check()

        return {
            "status": "success",
            "server_health": "healthy",
            "database_health": health,
            "database_path": db_manager.db_path,
            "model_loaded": model_manager.model_loaded,
        }

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {"error": str(e)}


async def startup():
    """Initialize the server components"""
    try:
        # Initialize database
        await db_manager.initialize()
        logger.info("ColPali FastMCP Server initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


if __name__ == "__main__":
    # Set default port
    port = int(os.getenv("COLPALI_PORT", "8000"))

    print(f"Starting ColPali FastMCP Server on port {port}", file=sys.stderr)
    print(f"Database path: {db_manager.db_path}", file=sys.stderr)
    print(f"MCP endpoint: http://localhost:{port}/mcp", file=sys.stderr)

    # Initialize database and run server
    async def init_and_run():
        try:
            await startup()
            print("Server initialized successfully", file=sys.stderr)
        except Exception as e:
            print(f"Startup failed: {e}", file=sys.stderr)
            raise
    
    # Run initialization first
    asyncio.run(init_and_run())
    
    # Then start the HTTP server
    print("Starting HTTP server...", file=sys.stderr)
    mcp.run(port=port)
