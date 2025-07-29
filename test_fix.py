#!/usr/bin/env python3
"""
Quick test to verify the tensor dimension fix
"""

print("🔧 Testing tensor dimension fix...")

# Test the fixed version
import sys
import os
sys.path.insert(0, '/Volumes/myssd/colpali-mcp')

try:
    result = os.system("cd /Volumes/myssd/colpali-mcp && python test_apple_silicon.py")
    if result == 0:
        print("✅ Test passed!")
    else:
        print("❌ Test failed")
except Exception as e:
    print(f"Error: {e}")
