#!/usr/bin/env python3
"""
Quick test to verify the MCP server fixes
"""

import asyncio
import sys
import os

# Add the project directory to Python path
sys.path.append('/Volumes/myssd/colpali-mcp')

async def test_server_components():
    """Test individual server components without running full MCP server"""
    
    print("🧪 Testing server components...", file=sys.stderr)
    
    try:
        # Test LanceDBManager initialization
        from colpali_streaming_server import LanceDBManager
        
        print("✅ LanceDBManager imported successfully", file=sys.stderr)
        
        # Test initialization
        db_manager = LanceDBManager()
        print(f"✅ LanceDBManager created with path: {db_manager.db_path}", file=sys.stderr)
        
        # Test async initialization
        await db_manager.initialize()
        print("✅ LanceDBManager.initialize() completed", file=sys.stderr)
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing components: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

async def test_mcp_server_creation():
    """Test MCP server creation without running it"""
    
    try:
        from colpali_streaming_server import ColPaliStreamingServer
        
        print("✅ ColPaliStreamingServer imported successfully", file=sys.stderr)
        
        # Create server instance
        server = ColPaliStreamingServer()
        print("✅ ColPaliStreamingServer created successfully", file=sys.stderr)
        
        # Test setup tools (without running)
        await server.setup_tools()
        print("✅ Server tools setup completed", file=sys.stderr)
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating MCP server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🚀 Starting ColPali MCP Server Tests", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    
    # Test components
    component_test = await test_server_components()
    server_test = await test_mcp_server_creation()
    
    print("=" * 50, file=sys.stderr)
    
    if component_test and server_test:
        print("🎉 ALL TESTS PASSED! Server should work now.", file=sys.stderr)
        print("   Try running your MCP server again.", file=sys.stderr)
    else:
        print("❌ Some tests failed. Check the errors above.", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
