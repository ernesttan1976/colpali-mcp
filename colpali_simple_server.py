#!/usr/bin/env python3
"""
ColPali MCP Server - Fixed Version
Simplified to resolve timeout issues
"""

import asyncio
import json
import logging
import sys
from typing import Dict, List, Optional, Any

from mcp import server
from mcp.server.stdio import stdio_server


class ColPaliStreamingServer:
    """Main MCP Server with streaming capabilities"""

    def __init__(self):
        self.server = server.Server("colpali-streaming")
        # Setup logging to stderr
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stderr,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_tools(self):
        """Setup MCP tools - synchronous version"""

        @self.server.list_tools()
        async def list_tools():
            """List available tools"""
            return [
                {
                    "name": "test_connection",
                    "description": "Test the ColPali server connection",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
                {
                    "name": "ingest_pdf",
                    "description": "Ingest a PDF document for search",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to PDF file",
                            },
                            "doc_name": {
                                "type": "string",
                                "description": "Optional document name",
                            },
                        },
                        "required": ["file_path"],
                    },
                },
                {
                    "name": "search_documents",
                    "description": "Search ingested documents",
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
                    "name": "list_documents",
                    "description": "List all ingested documents",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
                {
                    "name": "health_check",
                    "description": "Check server health and database status",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]):
            """Handle tool calls"""
            try:
                self.logger.info(f"Tool called: {name} with args: {arguments}")
                
                if name == "test_connection":
                    return await self.test_connection()
                elif name == "ingest_pdf":
                    return await self.ingest_pdf(**arguments)
                elif name == "search_documents":
                    return await self.search_documents(**arguments)
                elif name == "list_documents":
                    return await self.list_documents()
                elif name == "health_check":
                    return await self.health_check()
                else:
                    return {"error": f"Unknown tool: {name}"}

            except Exception as e:
                self.logger.error(f"Error in {name}: {str(e)}")
                return {"error": str(e)}

    async def test_connection(self):
        """Test connection tool"""
        return {
            "status": "success",
            "message": "ColPali MCP Server is running!",
            "server_info": {
                "name": "colpali-streaming",
                "version": "1.0.0",
                "database_path": "/Volumes/myssd/colpali-mcp/data/embeddings_db"
            }
        }

    async def ingest_pdf(self, file_path: str, doc_name: Optional[str] = None):
        """Ingest a PDF document"""
        try:
            # Import and use the database class
            import sys
            import os
            sys.path.append('/Volumes/myssd/colpali-mcp')
            
            from db import DocumentEmbeddingDatabase
            
            # Initialize database
            db = DocumentEmbeddingDatabase()
            
            # Check if already exists
            if db.embeddings_exist(file_path):
                return {
                    "status": "already_exists",
                    "message": f"Embeddings for {os.path.basename(file_path)} already exist",
                    "file_path": file_path
                }
            
            # For now, return a mock response
            return {
                "status": "success", 
                "message": f"PDF ingestion started for {file_path}",
                "note": "This is a mock implementation - actual ColPali processing not yet implemented"
            }
            
        except Exception as e:
            self.logger.error(f"Error ingesting PDF: {e}")
            return {"error": str(e)}

    async def search_documents(self, query: str, top_k: int = 5):
        """Search documents"""
        try:
            # Import and use the database class
            import sys
            sys.path.append('/Volumes/myssd/colpali-mcp')
            
            from db import DocumentEmbeddingDatabase
            
            # Initialize database
            db = DocumentEmbeddingDatabase()
            
            # List available documents
            documents = db.list_documents()
            
            return {
                "status": "success",
                "query": query,
                "results": [
                    {
                        "doc_name": f"sample_doc_{i+1}.pdf",
                        "page_num": i + 1,
                        "score": 0.9 - i * 0.1,
                        "snippet": f"Mock search result for '{query}' - page {i+1}"
                    }
                    for i in range(min(top_k, 3))
                ],
                "available_documents": len(documents),
                "note": "This is a mock search - actual ColPali search not yet implemented"
            }
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return {"error": str(e)}

    async def list_documents(self):
        """List all documents"""
        try:
            # Import and use the database class
            import sys
            sys.path.append('/Volumes/myssd/colpali-mcp')
            
            from db import DocumentEmbeddingDatabase
            
            # Initialize database
            db = DocumentEmbeddingDatabase()
            documents = db.list_documents()
            
            return {
                "status": "success",
                "documents": documents,
                "total_count": len(documents)
            }
            
        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            return {"error": str(e)}

    async def health_check(self):
        """Check server health"""
        try:
            # Import and use the database class
            import sys
            import os
            sys.path.append('/Volumes/myssd/colpali-mcp')
            
            from db import DocumentEmbeddingDatabase
            
            # Initialize database and run health check
            db = DocumentEmbeddingDatabase()
            health = db.health_check()
            
            return {
                "status": "success",
                "server_health": "healthy",
                "database_health": health,
                "python_version": sys.version,
                "working_directory": os.getcwd()
            }
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {"error": str(e)}

    async def run(self):
        """Run the MCP server"""
        try:
            # Setup tools synchronously
            self.setup_tools()
            self.logger.info("Tools setup completed")

            async with stdio_server() as (read_stream, write_stream):
                self.logger.info("ColPali MCP Server starting...")
                await self.server.run(read_stream, write_stream, {})
                
        except Exception as e:
            self.logger.error(f"Error running server: {e}")
            raise


async def main():
    """Main entry point"""
    try:
        server_instance = ColPaliStreamingServer()
        await server_instance.run()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    asyncio.run(main())
