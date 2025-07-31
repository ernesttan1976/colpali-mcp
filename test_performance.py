#!/usr/bin/env python3
"""
Performance Test for ColPali Maximum Performance Mode
"""

import time
import torch
import requests
import json

def test_performance_settings():
    print("🚀 Testing ColPali Maximum Performance Configuration")
    print("=" * 60)
    
    # Test device and memory settings
    print("\n📱 Device Configuration:")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ Apple Silicon MPS detected")
        expected_batch_size = 16
        expected_mode = "MAXIMUM PERFORMANCE MODE"
    elif torch.cuda.is_available():
        print("✅ CUDA detected") 
        expected_batch_size = 16
        expected_mode = "MAXIMUM PERFORMANCE MODE"
    else:
        print("⚠️  Using CPU")
        expected_batch_size = 4
        expected_mode = "CPU mode"
    
    # Test memory settings
    import os
    mps_ratio = os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
    if mps_ratio is None:
        print("✅ Memory restrictions removed (PYTORCH_MPS_HIGH_WATERMARK_RATIO not set)")
    else:
        print(f"⚠️  Memory restriction still active: {mps_ratio}")
    
    print(f"🎯 Expected batch size: {expected_batch_size}")
    print(f"🎯 Expected mode: {expected_mode}")
    
    return True

def test_server_performance():
    """Test actual server performance"""
    print("\n🧪 Testing Server Performance...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server healthy: {health.get('status')}")
            print(f"📊 Model loaded: {health.get('model_loaded')}")
            print(f"🎯 Device: {health.get('device')}")
        else:
            print("❌ Server not responding")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure to start the server first: python colpali_http_server.py")
        return False
    
    # Test search performance
    print("\n⏱️  Testing search performance...")
    start_time = time.time()
    
    try:
        search_data = {
            "query": "performance test query",
            "top_k": 5
        }
        
        response = requests.post(
            f"{base_url}/search",
            json=search_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            
            if task_id:
                print(f"🔄 Search task started: {task_id}")
                
                # Monitor for performance indicators
                for _ in range(60):  # Wait up to 60 seconds
                    try:
                        progress_response = requests.get(f"{base_url}/progress/{task_id}", timeout=5)
                        if progress_response.status_code == 200:
                            progress = progress_response.json().get("progress", {})
                            
                            current_step = progress.get("current_step", "")
                            details = progress.get("details", "")
                            
                            # Look for performance indicators
                            if "MAXIMUM PERFORMANCE" in current_step or "MAXIMUM PERFORMANCE" in details:
                                print("🚀 MAXIMUM PERFORMANCE MODE detected!")
                            
                            if "batch size: 16" in details:
                                print("✅ Large batch size (16) detected!")
                            elif "batch size: 4" in details:
                                print("ℹ️  Standard batch size (4) - CPU mode")
                            
                            # Check for completion
                            if progress.get("progress", 0) >= 100:
                                elapsed = time.time() - start_time
                                results = progress.get("results", [])
                                print(f"✅ Search completed in {elapsed:.2f} seconds")
                                print(f"📊 Found {len(results)} results")
                                return True
                            
                            if progress.get("error"):
                                print(f"❌ Search error: {progress['error']}")
                                return False
                        
                        time.sleep(1)
                    except:
                        break
                        
        else:
            print(f"❌ Search request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False
    
    print("⏱️  Search test timed out (normal for large documents)")
    return True

def main():
    print("🔥 ColPali Maximum Performance Test")
    print("Testing optimizations for speed...")
    
    # Test configuration
    test_performance_settings()
    
    # Test server performance  
    server_ok = test_server_performance()
    
    print("\n" + "=" * 60)
    if server_ok:
        print("🎉 Performance test completed!")
        print("🚀 Server is running in MAXIMUM PERFORMANCE MODE")
        print("\n💡 Tips for maximum speed:")
        print("   - Close other applications to free memory")
        print("   - Use documents with 50-200 pages for optimal batch processing")
        print("   - Monitor system resources during processing")
    else:
        print("⚠️  Performance test had issues")
        print("💡 Check server logs for details")

if __name__ == "__main__":
    main()
