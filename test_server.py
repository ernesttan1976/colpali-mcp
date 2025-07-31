#!/usr/bin/env python3
"""
Test script for ColPali Production Server
Run this after starting the server to verify it's working correctly
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_server():
    print("🧪 Testing ColPali Production Server")
    print("=" * 50)
    
    # 1. Health check
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        health = response.json()
        print(f"✅ Server healthy: {health}")
        
        if not health.get("model_loaded", False):
            print("⚠️  Model not loaded yet - searches will auto-load the model")
        else:
            print("✅ Model already loaded and ready!")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # 2. Model status
    print("\n2️⃣ Checking model status...")
    try:
        response = requests.get(f"{BASE_URL}/model/status")
        model_status = response.json()
        print(f"📊 Model status: {json.dumps(model_status, indent=2)}")
    except Exception as e:
        print(f"⚠️  Model status check failed: {e}")
    
    # 3. Manual model initialization (optional)
    if not health.get("model_loaded", False):
        print("\n3️⃣ Manually initializing model...")
        try:
            response = requests.post(f"{BASE_URL}/initialize")
            init_result = response.json()
            print(f"📝 Initialization started: {init_result}")
            
            if "task_id" in init_result:
                # Monitor progress
                task_id = init_result["task_id"]
                print(f"📈 Monitoring initialization progress for task: {task_id}")
                
                for _ in range(30):  # Wait up to 30 seconds
                    try:
                        progress_response = requests.get(f"{BASE_URL}/progress/{task_id}")
                        progress = progress_response.json()
                        
                        if progress and "progress" in progress:
                            prog_data = progress["progress"]
                            print(f"   📊 {prog_data.get('progress', 0):.1f}% - {prog_data.get('current_step', 'Working...')}")
                            
                            if prog_data.get('progress', 0) >= 100:
                                print("✅ Model initialization complete!")
                                break
                                
                            if prog_data.get('error'):
                                print(f"❌ Initialization error: {prog_data['error']}")
                                break
                        
                        time.sleep(2)
                    except:
                        break
        except Exception as e:
            print(f"⚠️  Manual initialization failed: {e}")
    
    # 4. Test search (this will auto-load model if needed)
    print("\n4️⃣ Testing search functionality...")
    try:
        search_data = {
            "query": "test search query",
            "top_k": 3
        }
        
        response = requests.post(
            f"{BASE_URL}/search",
            json=search_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            search_result = response.json()
            print(f"🔍 Search initiated: {search_result}")
            
            if "task_id" in search_result:
                task_id = search_result["task_id"]
                print(f"📈 Monitoring search progress for task: {task_id}")
                
                for _ in range(60):  # Wait up to 60 seconds
                    try:
                        progress_response = requests.get(f"{BASE_URL}/progress/{task_id}")
                        progress = progress_response.json()
                        
                        if progress and "progress" in progress:
                            prog_data = progress["progress"]
                            print(f"   📊 {prog_data.get('progress', 0):.1f}% - {prog_data.get('current_step', 'Working...')}")
                            
                            if prog_data.get('progress', 0) >= 100:
                                results = prog_data.get('results', [])
                                print(f"✅ Search complete! Found {len(results)} results")
                                if results:
                                    print("📋 Sample results:")
                                    for i, result in enumerate(results[:2]):
                                        print(f"   {i+1}. {result}")
                                else:
                                    print("ℹ️  No results (expected if no documents are indexed)")
                                break
                                
                            if prog_data.get('error'):
                                print(f"❌ Search error: {prog_data['error']}")
                                break
                        
                        time.sleep(2)
                    except:
                        break
        else:
            print(f"❌ Search request failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")
    
    # 5. List documents
    print("\n5️⃣ Checking indexed documents...")
    try:
        response = requests.get(f"{BASE_URL}/documents")
        docs = response.json()
        print(f"📚 Documents in database: {docs.get('total', 0)}")
        
        if docs.get('documents'):
            print("📋 Available documents:")
            for doc in docs['documents'][:3]:
                print(f"   - {doc.get('name', 'Unknown')} ({doc.get('pages', 0)} pages)")
        else:
            print("ℹ️  No documents indexed yet. Upload a PDF to test full functionality!")
            
    except Exception as e:
        print(f"⚠️  Document listing failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Server test complete!")
    print("💡 To upload a document: curl -X POST -F \"file=@document.pdf\" http://localhost:8000/ingest")
    print("🔍 To search: curl -X POST -H \"Content-Type: application/json\" -d '{\"query\": \"your query\", \"top_k\": 5}' http://localhost:8000/search")

if __name__ == "__main__":
    test_server()
