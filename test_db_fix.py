#!/usr/bin/env python3
"""
Quick test to verify database path fix
"""

import sys
import os
sys.path.append('/Volumes/myssd/colpali-mcp')

# Test the fixed db.py
try:
    from db import DocumentEmbeddingDatabase
    print("Testing DocumentEmbeddingDatabase...")
    db = DocumentEmbeddingDatabase()
    print(f"✅ DocumentEmbeddingDatabase initialized with path: {db.db_path}")
    
    health = db.health_check()
    print(f"✅ Health check: {health['status']}")
    if health['issues']:
        print(f"⚠️  Issues: {health['issues']}")
    
except Exception as e:
    print(f"❌ Error with DocumentEmbeddingDatabase: {e}")

# Test the streaming server's LanceDBManager
try:
    # Import the streaming server module
    import asyncio
    from colpali_streaming_server import LanceDBManager
    
    print("\nTesting LanceDBManager from streaming server...")
    manager = LanceDBManager()
    print(f"✅ LanceDBManager initialized with path: {manager.db_path}")
    
    # Test async initialization
    async def test_init():
        try:
            await manager.initialize()
            print("✅ LanceDBManager.initialize() succeeded")
        except Exception as e:
            print(f"❌ LanceDBManager.initialize() failed: {e}")
    
    asyncio.run(test_init())
    
except Exception as e:
    print(f"❌ Error with LanceDBManager: {e}")

print("\n🔍 Environment check:")
print(f"COLPALI_DB_PATH env var: {os.getenv('COLPALI_DB_PATH', 'Not set')}")
print(f"Data directory exists: {os.path.exists('/Volumes/myssd/colpali-mcp/data')}")
print(f"Data directory writable: {os.access('/Volumes/myssd/colpali-mcp/data', os.W_OK)}")
