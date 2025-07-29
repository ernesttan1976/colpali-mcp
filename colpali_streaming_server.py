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
from aiohttp import web, web_request
import aiohttp_cors

import torch
import lancedb
from PIL import Image
from mcp import server
from mcp.server.stdio import stdio_server

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
            import os
            db_path = os.getenv('COLPALI_DB_PATH', '/Volumes/myssd/colpali-mcp/data/embeddings_db')
        
        self.db_path = db_path
        self.db = None
        self.table = None
        
        # Ensure the database directory exists and is writable
        try:
            os.makedirs(self.db_path, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(self.db_path, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            import sys
            print(f"Database directory is writable: {self.db_path}", file=sys.stderr)
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Cannot create or write to database directory {db_path}: {e}")
        
        import sys
        print(f"LanceDBManager initialized with path: {self.db_path}", file=sys.stderr)

    async def initialize(self):
        """Initialize LanceDB connection"""
        try:
            self.db = lancedb.connect(self.db_path)
            import sys
            print(f"Successfully connected to LanceDB at {self.db_path}", file=sys.stderr)
        except Exception as e:
            import sys
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


class ColPaliStreamingServer:
    """Main MCP Server with streaming capabilities"""

    def __init__(self):
        self.server = server.Server("colpali-streaming")
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
            extracted_data = None
            async for progress_or_data in self.pdf_processor.extract_pages(file_path):
                if isinstance(progress_or_data, StreamingProgress):
                    progress_or_data.task_id = task_id
                    self.latest_progress[task_id] = progress_or_data
                else:
                    # This is the final data
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
            async for progress_or_data in self.model_manager.encode_pages(mock_images):
                if isinstance(progress_or_data, StreamingProgress):
                    progress_or_data.task_id = task_id
                    self.latest_progress[task_id] = progress_or_data
                else:
                    # This is the final embeddings
                    embeddings = progress_or_data.get("embeddings", [])

            # Store in database
            async for progress in self.db_manager.store_embeddings(
                embeddings or [torch.randn(1, 128) for _ in range(3)], mock_metadata
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
            query_embedding = None
            async for progress_or_data in self.model_manager.encode_query(query):
                if isinstance(progress_or_data, StreamingProgress):
                    progress_or_data.task_id = task_id
                    self.latest_progress[task_id] = progress_or_data
                else:
                    # This is the final query embedding
                    query_embedding = progress_or_data.get("query_embedding")

            # Search database
            results = None
            async for progress_or_data in self.db_manager.search_embeddings(
                query_embedding or torch.randn(1, 128), top_k
            ):
                if isinstance(progress_or_data, StreamingProgress):
                    progress_or_data.task_id = task_id
                    self.latest_progress[task_id] = progress_or_data
                else:
                    # This is the final results
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

    async def run_http_server(self, host: str = "127.0.0.1", port: int = 8080):
        """Run as HTTP server for remote MCP connections"""
        try:
            # Setup tools
            await self.setup_tools()
            print("✅ Tools setup completed", file=sys.stderr)
            
            # Initialize database
            await self.db_manager.initialize()
            print("✅ Database initialized", file=sys.stderr)
            
            print(f"🌐 Starting ColPali HTTP MCP server on {host}:{port}", file=sys.stderr)
            
            app = web.Application()
            
            # Add CORS support
            cors = aiohttp_cors.setup(app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*"
                )
            })
            
            async def handle_mcp(request):
                """Handle MCP requests over HTTP"""
                try:
                    if request.method == 'POST':
                        data = await request.json()
                        
                        # Handle MCP initialize
                        if data.get('method') == 'initialize':
                            return web.json_response({
                                "jsonrpc": "2.0",
                                "id": data.get('id'),
                                "result": {
                                    "protocolVersion": "2025-06-18",
                                    "capabilities": {
                                        "tools": {},
                                        "logging": {}
                                    },
                                    "serverInfo": {
                                        "name": "colpali-streaming",
                                        "version": "1.0.0"
                                    }
                                }
                            })
                        
                        # Handle tools/list
                        elif data.get('method') == 'tools/list':
                            # Get tools from our setup
                            tools_list = [
                                {
                                    "name": "ingest_pdf_stream",
                                    "description": "Ingest PDF with real-time progress streaming",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "file_path": {"type": "string", "description": "Path to PDF file"},
                                            "doc_name": {"type": "string", "description": "Optional document name"}
                                        },
                                        "required": ["file_path"]
                                    }
                                },
                                {
                                    "name": "search_documents_stream", 
                                    "description": "Search documents with real-time progress updates",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {"type": "string", "description": "Search query text"},
                                            "top_k": {"type": "integer", "description": "Number of results", "default": 5}
                                        },
                                        "required": ["query"]
                                    }
                                },
                                {
                                    "name": "health_check",
                                    "description": "Check server health and database status",
                                    "inputSchema": {"type": "object", "properties": {}}
                                }
                            ]
                            
                            return web.json_response({
                                "jsonrpc": "2.0",
                                "id": data.get('id'),
                                "result": {"tools": tools_list}
                            })
                        
                        # Handle tools/call
                        elif data.get('method') == 'tools/call':
                            params = data.get('params', {})
                            tool_name = params.get('name')
                            arguments = params.get('arguments', {})
                            
                            # Call the appropriate method
                            if tool_name == "ingest_pdf_stream":
                                result = await self.ingest_pdf_stream(**arguments)
                            elif tool_name == "search_documents_stream":
                                result = await self.search_documents_stream(**arguments)
                            elif tool_name == "get_task_progress":
                                result = await self.get_task_progress(arguments["task_id"])
                            elif tool_name == "list_active_tasks":
                                result = await self.list_active_tasks()
                            elif tool_name == "initialize_colpali":
                                result = await self.initialize_model_stream()
                            elif tool_name == "health_check":
                                result = {
                                    "status": "healthy",
                                    "database_path": self.db_manager.db_path,
                                    "server_info": "ColPali MCP Server v1.0.0"
                                }
                            else:
                                result = {"error": f"Unknown tool: {tool_name}"}
                            
                            return web.json_response({
                                "jsonrpc": "2.0",
                                "id": data.get('id'),
                                "result": result
                            })
                        
                        else:
                            return web.json_response({
                                "jsonrpc": "2.0",
                                "id": data.get('id'),
                                "error": {"code": -32601, "message": "Method not found"}
                            }, status=404)
                    
                    elif request.method == 'GET':
                        return web.json_response({
                            "status": "healthy",
                            "server": "colpali-streaming",
                            "version": "1.0.0",
                            "database_path": self.db_manager.db_path,
                            "endpoints": ["/mcp", "/health"]
                        })
                        
                except Exception as e:
                    print(f"❌ HTTP request error: {e}", file=sys.stderr)
                    return web.json_response({
                        "jsonrpc": "2.0",
                        "id": data.get('id') if 'data' in locals() else None,
                        "error": {"code": -32603, "message": str(e)}
                    }, status=500)
            
            # Add routes
            app.router.add_post('/mcp', handle_mcp)
            app.router.add_get('/mcp', handle_mcp)
            app.router.add_get('/health', handle_mcp)
            
            # Add CORS to all routes
            for route in list(app.router.routes()):
                cors.add(route)
            
            print(f"📋 Server endpoints:", file=sys.stderr)
            print(f"   • POST http://{host}:{port}/mcp - MCP JSON-RPC calls", file=sys.stderr)
            print(f"   • GET  http://{host}:{port}/health - Health check", file=sys.stderr)
            print(f"🔧 Available tools:", file=sys.stderr)
            print(f"   • ingest_pdf_stream", file=sys.stderr)
            print(f"   • search_documents_stream", file=sys.stderr)
            print(f"   • get_task_progress", file=sys.stderr)
            print(f"   • list_active_tasks", file=sys.stderr)
            print(f"   • initialize_colpali", file=sys.stderr)
            
            # Start server
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            
            print(f"🎉 ColPali HTTP MCP server running on http://{host}:{port}", file=sys.stderr)
            print(f"🛑 Press Ctrl+C to stop", file=sys.stderr)
            
            # Keep server running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print(f"\n🛑 Server stopped by user", file=sys.stderr)
            finally:
                await runner.cleanup()
                
        except Exception as e:
            print(f"❌ HTTP server error: {e}", file=sys.stderr)
            raise

    async def run_standalone(self):
        """Run as standalone server for testing"""
        try:
            # Setup tools
            await self.setup_tools()
            print("✅ Tools setup completed", file=sys.stderr)
            
            # Initialize database
            await self.db_manager.initialize()
            print("✅ Database initialized", file=sys.stderr)
            
            print("🚀 ColPali server running standalone - press Ctrl+C to stop", file=sys.stderr)
            print("📝 Available tools:", file=sys.stderr)
            print("   - ingest_pdf_stream", file=sys.stderr)
            print("   - search_documents_stream", file=sys.stderr)
            print("   - get_task_progress", file=sys.stderr)
            print("   - list_active_tasks", file=sys.stderr)
            print("   - initialize_colpali", file=sys.stderr)
            
            # Keep server running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user", file=sys.stderr)
        except Exception as e:
            print(f"❌ Server error: {e}", file=sys.stderr)
            raise

    async def run(self):
        """Run the MCP server"""
        await self.setup_tools()
        await self.db_manager.initialize()

        async with stdio_server() as (read_stream, write_stream):
            # Use stderr for logging to avoid breaking MCP JSON protocol
            import sys
            print("ColPali MCP Server starting...", file=sys.stderr)
            await self.server.run(read_stream, write_stream, {})


async def main():
    """Main entry point"""
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--standalone":
            print("🔧 Starting ColPali server in standalone mode...", file=sys.stderr)
            server_instance = ColPaliStreamingServer()
            await server_instance.run_standalone()
        elif sys.argv[1] == "--http":
            # Default HTTP server
            host = "127.0.0.1"
            port = 8080
            
            # Parse optional host and port
            if len(sys.argv) > 2:
                try:
                    if ":" in sys.argv[2]:
                        host, port_str = sys.argv[2].split(":")
                        port = int(port_str)
                    else:
                        port = int(sys.argv[2])
                except ValueError:
                    print(f"❌ Invalid port: {sys.argv[2]}", file=sys.stderr)
                    sys.exit(1)
            
            if len(sys.argv) > 3:
                host = sys.argv[3]
            
            print(f"🌐 Starting ColPali HTTP server on {host}:{port}...", file=sys.stderr)
            server_instance = ColPaliStreamingServer()
            await server_instance.run_http_server(host, port)
        else:
            print(f"❌ Unknown argument: {sys.argv[1]}", file=sys.stderr)
            print("Usage:", file=sys.stderr)
            print("  python colpali_streaming_server.py                    # MCP stdio mode", file=sys.stderr)
            print("  python colpali_streaming_server.py --standalone       # Standalone mode", file=sys.stderr)
            print("  python colpali_streaming_server.py --http [port] [host] # HTTP server mode", file=sys.stderr)
            print("  python colpali_streaming_server.py --http 8080 0.0.0.0  # HTTP on all interfaces", file=sys.stderr)
            sys.exit(1)
    else:
        print("🔧 Starting ColPali server in MCP stdio mode...", file=sys.stderr)
        server_instance = ColPaliStreamingServer()
        await server_instance.run()


if __name__ == "__main__":
    import sys  # Add missing import

    asyncio.run(main())
