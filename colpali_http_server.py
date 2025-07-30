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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Any

import torch
import lancedb
from PIL import Image
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

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


class LanceDBManager:
    """Handles LanceDB operations"""

    def __init__(self, db_path: str = "./colpali_db"):
        self.db_path = db_path
        self.db = None
        self.table = None

    async def initialize(self):
        """Initialize LanceDB connection"""
        self.db = lancedb.connect(self.db_path)

    async def search_embeddings(self, query_embedding, limit=10, score_threshold=0.7):
        """
        Search for similar embeddings in the database

        Args:
            query_embedding: The query embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold

        Yields:
            StreamingProgress objects
        """
        task_id = f"search_{uuid.uuid4().hex[:8]}"
        
        try:
            if self.table is None:
                yield StreamingProgress(
                    task_id=task_id,
                    progress=0.0,
                    current_step="Database not initialized",
                    step_num=0,
                    total_steps=1,
                    error="Database table not initialized"
                )
                return

            yield StreamingProgress(
                task_id=task_id,
                progress=25.0,
                current_step="Searching embeddings",
                step_num=1,
                total_steps=3,
                details="Performing vector similarity search"
            )

            # Perform vector similarity search
            try:
                # Use LanceDB's native search without pandas conversion
                search_query = self.table.search(query_embedding).limit(limit)
                results = search_query.to_lance().to_table().to_pylist()
            except Exception as search_error:
                yield StreamingProgress(
                    task_id=task_id,
                    progress=0.0,
                    current_step="Search failed",
                    step_num=0,
                    total_steps=1,
                    error=f"Vector search failed: {str(search_error)}"
                )
                return

            yield StreamingProgress(
                task_id=task_id,
                progress=75.0,
                current_step="Processing results",
                step_num=2,
                total_steps=3,
                details=f"Found {len(results)} potential matches"
            )

            # Filter by score threshold if needed
            if score_threshold > 0:
                results = [r for r in results if r.get("_distance", 1.0) <= (1 - score_threshold)]

            # Format results
            search_results = []
            for row in results:
                search_results.append(
                    SearchResult(
                        page_num=row.get("page_num", 0),
                        doc_name=row.get("doc_name", ""),
                        score=1 - row.get("_distance", 1.0),  # Convert distance to similarity score
                        snippet=row.get("text_content", "")[:200]  # First 200 chars as snippet
                    )
                )

            yield StreamingProgress(
                task_id=task_id,
                progress=100.0,
                current_step="Search completed",
                step_num=3,
                total_steps=3,
                details=f"Found {len(search_results)} relevant matches",
                results=[result.__dict__ for result in search_results]  # Convert to dict for JSON serialization
            )

        except Exception as e:
            yield StreamingProgress(
                task_id=task_id,
                progress=0.0,
                current_step="Search failed",
                step_num=0,
                total_steps=1,
                error=f"Search failed: {str(e)}"
            )

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

        # Prepare data with timestamp
        import datetime

        current_time = datetime.datetime.now().isoformat()

        data = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            data.append(
                {
                    "id": f"{meta['doc_name']}_page_{meta['page_num']}",
                    "embedding": embedding.numpy().flatten().tolist(),
                    "doc_name": meta["doc_name"],
                    "page_num": meta["page_num"],
                    "text_content": meta.get("text_content", ""),
                    "created_at": current_time,
                    "file_size": meta.get("file_size", "unknown"),
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

            # Try to open existing table
            try:
                self.table = self.db.open_table("documents")
                # Add new data to existing table
                self.table.add(data)
            except Exception:
                # Create new table if doesn't exist
                self.table = self.db.create_table("documents", data)

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
                "active_tasks": len(self.active_tasks),
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
                    # Use LanceDB's native query without pandas
                    all_docs = self.db_manager.table.to_lance().to_table().to_pylist()
                    
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
                                "created_at": doc.get("created_at", "Unknown")
                            }
                        
                        docs_by_name[doc_name]["embeddings_count"] += 1
                        page_num = doc.get("page_num", 0)
                        if page_num > docs_by_name[doc_name]["max_page"]:
                            docs_by_name[doc_name]["max_page"] = page_num

                    # Format response
                    documents = []
                    for doc_name, stats in docs_by_name.items():
                        documents.append({
                            "id": f"doc_{hash(doc_name) % 10000}",
                            "name": doc_name,
                            "pages": stats["max_page"],
                            "embeddings_count": stats["embeddings_count"],
                            "created_date": stats["created_at"][:10] if stats["created_at"] != "Unknown" else "2024-07-25",
                            "size": "N/A",
                        })

                    return {"documents": documents, "total": len(documents)}
                    
                except Exception as table_error:
                    self.logger.warning(f"Could not read table data: {table_error}")
                    return {"documents": [], "total": 0, "info": "No documents indexed yet"}

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
                self.logger.info(f"Task {task_id}: Model not loaded, initializing...")
                async for progress in self.model_manager.load_model():
                    progress.task_id = task_id
                    progress.step_num = 2
                    progress.total_steps = 6
                    self.latest_progress[task_id] = progress
                    self.logger.info(
                        f"Task {task_id}: Model loading - {progress.current_step} ({progress.progress:.1f}%)"
                    )
                    await asyncio.sleep(0.1)  # Allow other tasks to run

            # Extract PDF pages
            self.logger.info(f"Task {task_id}: Starting PDF page extraction")
            images = []
            metadata = []
            async for progress in self.pdf_processor.extract_pages(temp_file_path):
                progress.task_id = task_id
                progress.step_num = 3
                progress.total_steps = 6
                # Adjust progress to fit within step 3's range (20-40%)
                progress.progress = 20.0 + (progress.progress / 100.0) * 20.0
                self.latest_progress[task_id] = progress
                self.logger.info(
                    f"Task {task_id}: PDF extraction - {progress.current_step} ({progress.progress:.1f}%)"
                )
                await asyncio.sleep(0.1)

            # Mock data for demo - replace with actual extracted data
            images = [Image.new("RGB", (800, 600)) for _ in range(3)]
            metadata = [
                {
                    "page_num": i + 1,
                    "doc_name": actual_doc_name,
                    "text_content": f"Page {i + 1} content",
                }
                for i in range(3)
            ]
            self.logger.info(f"Task {task_id}: Extracted {len(images)} pages")

            # Encode pages
            self.logger.info(
                f"Task {task_id}: Starting ColPali encoding for {len(images)} pages"
            )
            embeddings = []
            async for progress in self.model_manager.encode_pages(images):
                progress.task_id = task_id
                progress.step_num = 4
                progress.total_steps = 6
                # Adjust progress to fit within step 4's range (40-70%)
                progress.progress = 40.0 + (progress.progress / 100.0) * 30.0
                self.latest_progress[task_id] = progress
                self.logger.info(
                    f"Task {task_id}: Encoding - {progress.current_step} ({progress.progress:.1f}%)"
                )
                if progress.throughput:
                    self.logger.info(
                        f"Task {task_id}: Encoding throughput: {progress.throughput}"
                    )
                await asyncio.sleep(0.1)

            embeddings = [torch.randn(1, 128) for _ in range(len(images))]
            self.logger.info(f"Task {task_id}: Generated {len(embeddings)} embeddings")

            # Store in database
            self.logger.info(f"Task {task_id}: Storing embeddings in LanceDB")
            async for progress in self.db_manager.store_embeddings(
                embeddings, metadata
            ):
                progress.task_id = task_id
                progress.step_num = 5
                progress.total_steps = 6
                # Adjust progress to fit within step 5's range (70-95%)
                progress.progress = 70.0 + (progress.progress / 100.0) * 25.0
                self.latest_progress[task_id] = progress
                self.logger.info(
                    f"Task {task_id}: Storage - {progress.current_step} ({progress.progress:.1f}%)"
                )
                await asyncio.sleep(0.1)

            # Final completion
            final_progress = StreamingProgress(
                task_id=task_id,
                progress=100.0,
                current_step="Ingestion completed successfully",
                step_num=6,
                total_steps=6,
                details=f"Document '{actual_doc_name}' ready for search",
                throughput=f"{len(images)} pages processed",
            )
            self.latest_progress[task_id] = final_progress
            self.logger.info(
                f"Task {task_id}: COMPLETED - {len(images)} pages indexed for document '{actual_doc_name}'"
            )

        except Exception as e:
            self.logger.error(
                f"Task {task_id}: FAILED during processing - {str(e)}",
                exc_info=True,
            )
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
            # Clean up temporary file
            try:
                Path(temp_file_path).unlink(missing_ok=True)
                self.logger.info(
                    f"Task {task_id}: Cleaned up temporary file {temp_file_path}"
                )
            except Exception as cleanup_error:
                self.logger.warning(
                    f"Task {task_id}: Failed to cleanup temp file: {cleanup_error}"
                )

    async def process_search_background(self, task_id: str, request: SearchRequest):
        """Background processing of search with real-time progress updates"""
        try:
            # Check if we have any documents indexed
            if not hasattr(self, "db_manager") or self.db_manager.table is None:
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

            # Encode query
            self.logger.info(f"Task {task_id}: Encoding search query")
            query_embedding = None
            async for progress in self.model_manager.encode_query(request.query):
                progress.task_id = task_id
                progress.step_num = 2
                progress.total_steps = 4
                progress.progress = 5.0 + (progress.progress / 100.0) * 45.0
                self.latest_progress[task_id] = progress
                self.logger.info(
                    f"Task {task_id}: Query encoding - {progress.current_step} ({progress.progress:.1f}%)"
                )
                await asyncio.sleep(0.1)

            # Set query_embedding (mock for now, replace with actual)
            query_embedding = torch.randn(1, 128)

            # Search database
            self.logger.info(f"Task {task_id}: Performing vector similarity search")
            search_results = None
            async for progress in self.db_manager.search_embeddings(
                query_embedding, request.top_k
            ):
                # Update progress to fit within the overall search workflow
                progress.task_id = task_id
                progress.step_num = 3
                progress.total_steps = 4
                progress.progress = 50.0 + (progress.progress / 100.0) * 40.0
                self.latest_progress[task_id] = progress
                self.logger.info(
                    f"Task {task_id}: Vector search - {progress.current_step} ({progress.progress:.1f}%)"
                )

                # Capture results from the final progress update
                if progress.progress >= 90.0 and progress.results:
                    search_results = progress.results
                await asyncio.sleep(0.1)

            # Process search results (already converted to dicts in search_embeddings)
            results = search_results if search_results else []

            # Final completion
            final_progress = StreamingProgress(
                task_id=task_id,
                progress=100.0,
                current_step="Search completed",
                step_num=4,
                total_steps=4,
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
