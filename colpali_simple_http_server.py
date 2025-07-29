#!/usr/bin/env python3
"""
ColPali FastMCP HTTP Server 
Working minimal version based on FastMCP docs
"""

import asyncio
import os
import sys
import torch
from pathlib import Path
from typing import Optional
from fastmcp import FastMCP

# Add project path for db imports
sys.path.append('/Volumes/myssd/colpali-mcp')

# Create FastMCP server
mcp = FastMCP(name="ColPali Server")


@mcp.tool()
async def test_connection() -> dict:
    """Test the ColPali server connection"""
    return {
        "status": "success",
        "message": "ColPali FastMCP Server is running!",
        "server_info": {
            "name": "colpali-streaming",
            "version": "1.0.0",
            "database_path": "/Volumes/myssd/colpali-mcp/data/embeddings_db"
        }
    }


@mcp.tool()
async def health_check() -> dict:
    """Check server health and database status"""
    try:
        from db import DocumentEmbeddingDatabase
        
        # Initialize database and run health check
        db = DocumentEmbeddingDatabase()
        health = db.health_check()
        
        return {
            "status": "success",
            "server_health": "healthy",
            "database_health": health,
            "database_path": db.db_path
        }
        
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def list_documents() -> dict:
    """List all ingested documents"""
    try:
        from db import DocumentEmbeddingDatabase
        
        db = DocumentEmbeddingDatabase()
        documents = db.list_documents()
        
        return {
            "status": "success",
            "documents": documents,
            "total_count": len(documents)
        }
        
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def ingest_pdf(file_path: str, doc_name: Optional[str] = None) -> dict:
    """Ingest a PDF document for search"""
    try:
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
        
        # Mock PDF processing for now
        mock_embeddings = [torch.randn(128) for _ in range(5)]
        page_count = 5
        
        # Save to database
        success = db.save_embeddings(file_path, mock_embeddings, page_count)
        
        if success:
            return {
                "status": "success",
                "message": f"Successfully ingested {os.path.basename(file_path)}",
                "file_path": file_path,
                "pages_processed": page_count,
                "note": "Using mock embeddings - real ColPali processing not yet implemented"
            }
        else:
            return {"error": "Failed to save embeddings"}
            
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def search_documents(query: str, top_k: int = 5) -> dict:
    """Search ingested documents"""
    try:
        from db import DocumentEmbeddingDatabase
        
        # Initialize database
        db = DocumentEmbeddingDatabase()
        documents = db.list_documents()
        
        # Mock search results
        results = []
        for i, doc in enumerate(documents[:top_k]):
            results.append({
                "doc_name": doc.get("filename", f"doc_{i}"),
                "page_num": i + 1,
                "score": 0.9 - i * 0.1,
                "snippet": f"Mock search result for '{query}' in {doc.get('filename', 'unknown')}"
            })
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "available_documents": len(documents),
            "note": "Using mock search - real ColPali search not yet implemented"
        }
        
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def delete_document(file_path: str) -> dict:
    """Delete a document and its embeddings"""
    try:
        from db import DocumentEmbeddingDatabase
        
        db = DocumentEmbeddingDatabase()
        success = db.delete_embeddings(file_path)
        
        if success:
            return {
                "status": "success", 
                "message": f"Deleted embeddings for {os.path.basename(file_path)}"
            }
        else:
            return {"error": "Document not found or deletion failed"}
            
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Set port from environment or default
    port = int(os.getenv('COLPALI_PORT', '8000'))
    
    print(f"🚀 Starting ColPali FastMCP Server on port {port}", file=sys.stderr)
    print(f"📁 Database: /Volumes/myssd/colpali-mcp/data/embeddings_db", file=sys.stderr)
    print(f"🌐 MCP endpoint: http://localhost:{port}/mcp", file=sys.stderr)
    print(f"❤️  Health check: http://localhost:{port}/health", file=sys.stderr)
    print("", file=sys.stderr)
    print("Available tools:", file=sys.stderr)
    print("  - test_connection", file=sys.stderr)
    print("  - health_check", file=sys.stderr)
    print("  - list_documents", file=sys.stderr)
    print("  - ingest_pdf", file=sys.stderr)
    print("  - search_documents", file=sys.stderr)
    print("  - delete_document", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Run the server using FastMCP's HTTP transport
    mcp.run(transport="http", host="0.0.0.0", port=port)
