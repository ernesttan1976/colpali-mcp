#!/usr/bin/env python3
"""
Test the FastMCP HTTP server directly
"""

import asyncio
import sys
import aiohttp
import json

async def test_fastmcp_server():
    """Test the FastMCP server endpoints"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing FastMCP HTTP Server...")
    print(f"📍 Server URL: {base_url}")
    print()
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Health endpoint
        try:
            print("1️⃣ Testing health endpoint...")
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    print("   ✅ Health endpoint responding")
                else:
                    print(f"   ❌ Health endpoint failed: {response.status}")
        except Exception as e:
            print(f"   ❌ Health endpoint error: {e}")
        
        # Test 2: MCP endpoint
        try:
            print("2️⃣ Testing MCP endpoint...")
            async with session.get(f"{base_url}/mcp") as response:
                if response.status == 200:
                    print("   ✅ MCP endpoint responding")
                    text = await response.text()
                    print(f"   📄 Response preview: {text[:100]}...")
                else:
                    print(f"   ❌ MCP endpoint failed: {response.status}")
        except Exception as e:
            print(f"   ❌ MCP endpoint error: {e}")
        
        # Test 3: Try to call a tool directly (if supported)
        try:
            print("3️⃣ Testing tool call...")
            tool_data = {
                "method": "tools/call",
                "params": {
                    "name": "test_connection",
                    "arguments": {}
                }
            }
            
            async with session.post(
                f"{base_url}/mcp", 
                json=tool_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print("   ✅ Tool call successful")
                    print(f"   📊 Result: {json.dumps(result, indent=2)}")
                else:
                    print(f"   ❌ Tool call failed: {response.status}")
                    error_text = await response.text()
                    print(f"   📄 Error: {error_text}")
        except Exception as e:
            print(f"   ❌ Tool call error: {e}")
        
        print()
        print("🎯 Test Summary:")
        print("   If all tests pass, the server is working correctly")
        print("   If MCP endpoint fails, check FastMCP configuration")
        print("   If tool calls fail, the proxy might be needed")

if __name__ == "__main__":
    asyncio.run(test_fastmcp_server())
