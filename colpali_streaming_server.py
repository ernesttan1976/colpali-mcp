#!/usr/bin/env python3
"""
ColPali MCP Server with Real-time Streaming Progress
Supports Apple Silicon M4 with MPS acceleration
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Any
from contextlib import asynccontextmanager

import torch
import lancedb
from PIL import Image
import fitz  # PyMuPDF
from mcp import Server
from mcp.server.stdio import stdio_server

# Mock ColPali imports (replace with actual imports)
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

        return embeddings

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

        return query_embedding


class LanceDBManager:
    """Handles LanceDB operations"""

    def __init__(self, db_path: str = "./colpali_db"):
        self.db_path = db_path
        self.db = None
        self.table = None

    async def initialize(self):
        """Initialize LanceDB connection"""
        self.db = lancedb.connect(self.db_path)

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

        return mock_results


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
            return [], []

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

        return images, metadata


class ColPaliStreamingServer:
    """Main MCP Server with streaming capabilities"""

    def __init__(self):
        self.server = Server("colpali-streaming")
        self.model_manager = ColPaliModelManager()
        self.db_manager = LanceDBManager()
        self.pdf_processor = PDFProcessor()

        # Progress tracking
        self.active_tasks: Dict[str, AsyncGenerator] = {}
        self.latest_progress: Dict[str, StreamingProgress] = {}

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def setup_tools(self):
        """Setup MCP tools"""

        @self.server.list_tools()
        async def list_tools():
            return [
                {
                    "name": "ingest_pdf_stream",
                    "description": "Ingest PDF with real-time progress streaming. Returns task_id for monitoring.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to PDF file",
                            },
                            "doc_name": {
                                "type": "string",
                                "description": "Optional document name (defaults to filename)",
                            },
                        },
                        "required": ["file_path"],
                    },
                },
                {
                    "name": "search_documents_stream",
                    "description": "Search documents with real-time progress updates",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "get_task_progress",
                    "description": "Get latest progress for a streaming task",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "Task ID from streaming operation",
                            }
                        },
                        "required": ["task_id"],
                    },
                },
                {
                    "name": "list_active_tasks",
                    "description": "List all currently active streaming tasks",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "initialize_colpali",
                    "description": "Initialize ColPali model with progress tracking",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]):
            try:
                if name == "initialize_colpali":
                    return await self.initialize_model_stream()
                elif name == "ingest_pdf_stream":
                    return await self.ingest_pdf_stream(**arguments)
                elif name == "search_documents_stream":
                    return await self.search_documents_stream(**arguments)
                elif name == "get_task_progress":
                    return await self.get_task_progress(arguments["task_id"])
                elif name == "list_active_tasks":
                    return await self.list_active_tasks()
                else:
                    return {"error": f"Unknown tool: {name}"}

            except Exception as e:
                self.logger.error(f"Error in {name}: {str(e)}")
                return {"error": str(e)}

    async def initialize_model_stream(self):
        """Initialize ColPali model with streaming progress"""
        task_id = f"init_{uuid.uuid4().hex[:8]}"
        progress_updates = []

        try:
            async for progress in self.model_manager.load_model():
                self.latest_progress[task_id] = progress
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
            self.latest_progress[task_id] = error_progress
            return {"task_id": task_id, "status": "failed", "error": str(e)}

    async def ingest_pdf_stream(self, file_path: str, doc_name: Optional[str] = None):
        """Stream PDF ingestion progress"""
        task_id = f"ingest_{uuid.uuid4().hex[:8]}"
        doc_name = doc_name or Path(file_path).stem

        try:
            # Ensure model is loaded
            if not self.model_manager.model_loaded:
                async for progress in self.model_manager.load_model():
                    self.latest_progress[task_id] = progress

            # Extract PDF pages
            async for progress in self.pdf_processor.extract_pages(file_path):
                progress.task_id = task_id
                self.latest_progress[task_id] = progress

            # This would normally return the images/metadata
            # For demo, we'll use mock data
            mock_images = [Image.new("RGB", (800, 600)) for _ in range(3)]
            mock_metadata = [
                {
                    "page_num": i + 1,
                    "doc_name": doc_name,
                    "text_content": f"Page {i + 1} content",
                }
                for i in range(3)
            ]

            # Encode pages
            async for progress in self.model_manager.encode_pages(mock_images):
                progress.task_id = task_id
                self.latest_progress[task_id] = progress

            # Store in database
            async for progress in self.db_manager.store_embeddings(
                [torch.randn(1, 128) for _ in range(3)], mock_metadata
            ):
                progress.task_id = task_id
                self.latest_progress[task_id] = progress

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
            self.latest_progress[task_id] = error_progress
            return {"task_id": task_id, "status": "failed", "error": str(e)}

    async def search_documents_stream(self, query: str, top_k: int = 5):
        """Stream document search progress"""
        task_id = f"search_{uuid.uuid4().hex[:8]}"

        try:
            # Encode query
            async for progress in self.model_manager.encode_query(query):
                progress.task_id = task_id
                self.latest_progress[task_id] = progress

            # Search database
            async for progress in self.db_manager.search_embeddings(
                torch.randn(1, 128), top_k
            ):
                progress.task_id = task_id
                self.latest_progress[task_id] = progress

            # Mock results for demo
            results = [
                {
                    "page_num": i + 1,
                    "doc_name": "sample_document.pdf",
                    "score": 0.95 - i * 0.1,
                    "snippet": f"Relevant content from page {i + 1}...",
                }
                for i in range(min(top_k, 3))
            ]

            return {
                "task_id": task_id,
                "status": "completed",
                "results": results,
                "message": f"Found {len(results)} relevant results",
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
            self.latest_progress[task_id] = error_progress
            return {"task_id": task_id, "status": "failed", "error": str(e)}

    async def get_task_progress(self, task_id: str):
        """Get latest progress for a task"""
        if task_id in self.latest_progress:
            return {
                "task_id": task_id,
                "progress": self.latest_progress[task_id].to_dict(),
            }
        else:
            return {"error": f"Task {task_id} not found"}

    async def list_active_tasks(self):
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

    async def run(self):
        """Run the MCP server"""
        await self.setup_tools()
        await self.db_manager.initialize()

        async with stdio_server() as (read_stream, write_stream):
            self.logger.info("ColPali MCP Server starting...")
            await self.server.run(read_stream, write_stream)


async def main():
    """Main entry point"""
    server = ColPaliStreamingServer()
    await server.run()


if __name__ == "__main__":
    import io  # Add missing import

    asyncio.run(main())
