#!/usr/bin/env python3
"""
ColPali MCP Adapter - Lightweight protocol bridge
Connects Claude Desktop to the long-running ColPali HTTP server
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
import aiohttp
from mcp import Server
from mcp.server.stdio import stdio_server

# Configuration
COLPALI_SERVER_URL = "http://localhost:8000"  # Your HTTP server URL


class ColPaliMCPAdapter:
    """Lightweight MCP adapter that bridges to HTTP server"""

    def __init__(self, server_url: str = COLPALI_SERVER_URL):
        self.server = Server("colpali-adapter")
        self.server_url = server_url
        self.session: Optional[aiohttp.ClientSession] = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def setup_http_session(self):
        """Setup HTTP client session"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

    async def check_server_health(self) -> bool:
        """Check if the HTTP server is running"""
        try:
            await self.setup_http_session()
            async with self.session.get(f"{self.server_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    self.logger.info(f"Server health: {health_data}")
                    return True
                return False
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to initialize ColPali model: {e}",
            }

    async def handle_ingest_pdf(self, arguments: Dict) -> Dict:
        """Handle PDF ingestion"""
        try:
            request_data = {
                "file_path": arguments["file_path"],
                "doc_name": arguments.get("doc_name"),
            }
            response = await self.make_http_request("POST", "/ingest", request_data)
            return {
                "status": "success",
                "data": response,
                "message": f"PDF ingestion started for: {arguments['file_path']}",
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to ingest PDF: {e}"}

    async def handle_search_documents(self, arguments: Dict) -> Dict:
        """Handle document search"""
        try:
            request_data = {
                "query": arguments["query"],
                "top_k": arguments.get("top_k", 5),
            }
            response = await self.make_http_request("POST", "/search", request_data)
            return {
                "status": "success",
                "data": response,
                "message": f"Search completed for query: '{arguments['query']}'",
            }
        except Exception as e:
            return {"status": "error", "message": f"Search failed: {e}"}

    async def handle_get_progress(self, arguments: Dict) -> Dict:
        """Handle progress check"""
        try:
            task_id = arguments["task_id"]
            response = await self.make_http_request("GET", f"/progress/{task_id}")
            return {
                "status": "success",
                "data": response,
                "message": f"Progress retrieved for task: {task_id}",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get progress for task {arguments['task_id']}: {e}",
            }

    async def handle_list_tasks(self) -> Dict:
        """Handle listing active tasks"""
        try:
            response = await self.make_http_request("GET", "/tasks")
            return {
                "status": "success",
                "data": response,
                "message": "Active tasks retrieved successfully",
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to list active tasks: {e}"}

    async def run(self):
        """Run the MCP adapter server"""
        await self.setup_tools()

        # Check server connectivity on startup
        if not await self.check_server_health():
            self.logger.warning(
                "ColPali HTTP server is not accessible. "
                "Make sure it's running on localhost:8000"
            )

        try:
            async with stdio_server() as (read_stream, write_stream):
                self.logger.info("ColPali MCP Adapter starting...")
                await self.server.run(read_stream, write_stream)
        finally:
            await self.cleanup()


async def main():
    """Main entry point"""
    adapter = ColPaliMCPAdapter()
    await adapter.run()


if __name__ == "__main__":
    asyncio.run(main())

    async def make_http_request(
        self, method: str, endpoint: str, data: Dict = None
    ) -> Dict:
        """Make HTTP request to the ColPali server"""
        await self.setup_http_session()

        url = f"{self.server_url}{endpoint}"

        try:
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    response.raise_for_status()
                    return await response.json()
            elif method.upper() == "POST":
                async with self.session.post(url, json=data) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP request failed: {e}")
            raise Exception(f"Failed to communicate with ColPali server: {e}")

    async def setup_tools(self):
        """Setup MCP tools that proxy to HTTP server"""

        @self.server.list_tools()
        async def list_tools():
            return [
                {
                    "name": "initialize_colpali",
                    "description": "Initialize the ColPali model on the server",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
                {
                    "name": "ingest_pdf",
                    "description": "Ingest a PDF document for searching",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the PDF file to ingest",
                            },
                            "doc_name": {
                                "type": "string",
                                "description": "Optional custom name for the document",
                            },
                        },
                        "required": ["file_path"],
                    },
                },
                {
                    "name": "search_documents",
                    "description": "Search through ingested documents using ColPali",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "get_task_progress",
                    "description": "Get progress status for a running task",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "Task ID to check progress for",
                            },
                        },
                        "required": ["task_id"],
                    },
                },
                {
                    "name": "list_active_tasks",
                    "description": "List all currently active tasks on the server",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
                {
                    "name": "server_health",
                    "description": "Check the health status of the ColPali server",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]):
            try:
                # Route tools to appropriate HTTP endpoints
                if name == "server_health":
                    return await self.handle_server_health()
                elif name == "initialize_colpali":
                    return await self.handle_initialize()
                elif name == "ingest_pdf":
                    return await self.handle_ingest_pdf(arguments)
                elif name == "search_documents":
                    return await self.handle_search_documents(arguments)
                elif name == "get_task_progress":
                    return await self.handle_get_progress(arguments)
                elif name == "list_active_tasks":
                    return await self.handle_list_tasks()
                else:
                    return {"error": f"Unknown tool: {name}"}

            except Exception as e:
                self.logger.error(f"Error in {name}: {str(e)}")
                return {"error": str(e)}

    async def handle_server_health(self) -> Dict:
        """Handle server health check"""
        try:
            response = await self.make_http_request("GET", "/health")
            return {
                "status": "success",
                "server_health": response,
                "message": "ColPali server is running and healthy",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"ColPali server is not accessible: {e}",
                "suggestion": "Make sure the ColPali HTTP server is running on localhost:8000",
            }

    async def handle_initialize(self) -> Dict:
        """Handle model initialization"""
        try:
            response = await self.make_http_request("POST", "/initialize")
            return {
                "status": "success",
                "data": response,
                "message": "ColPali model initialization started",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"ColPali server is not innitialised: {e}",
                "suggestion": "Make sure the ColPali HTTP server is running on localhost:8000",
            }
