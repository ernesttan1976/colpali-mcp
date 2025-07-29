# 🍎 Apple Silicon ColPali - 3.5GB Memory Limit

## 🎯 **Ultra-Low Memory Optimizations Applied**

### **Changes Made:**
- **Memory Limit:** Reduced from 8GB → **3.5GB strict limit**
- **Model Loading:** Max 2GB GPU memory allocation
- **Batch Size:** Forced to 1 (single image processing)
- **Chunk Size:** Reduced from 50 → 25 embeddings per chunk
- **Garbage Collection:** After every batch/chunk (aggressive cleanup)
- **Buffer Sizes:** Reduced from 2-3GB → 0.5-1.5GB buffers

### **Memory Breakdown (3.5GB total):**
- **Model:** ~2.0GB (float16, strict limit)
- **Processing:** ~1.0GB (single image batches)
- **Embeddings:** ~0.3GB (smaller chunks)
- **Buffer:** ~0.2GB (minimal safety margin)

### **Performance Impact:**
- **Slower Processing:** Single image batches (safer)
- **More Cleanup:** Garbage collection after every operation
- **Smaller Chunks:** 25 embeddings vs 50 (more I/O)
- **Stricter Limits:** Hard memory enforcement

## 🧪 **Ready to Test with 4.4GB Available**

Your system with 4.4GB available should work fine now:

```bash
# Test the ultra-low memory version
python validate_apple_silicon.py  # Should pass memory check now
python test_apple_silicon.py      # Quick test (3 pages)
python app_apple_silicon.py       # Full A330 manual test
```

### **Expected Behavior:**
- ✅ **Memory Check:** Will pass with 4.4GB available
- ⚡ **Model Loading:** ~2-3 minutes, peaks at ~3GB
- 🐌 **Processing:** Slower due to single image batches
- 🧹 **Cleanup:** Frequent "Cleaned cache" messages
- 📊 **Peak Usage:** Should stay under 3.5GB

### **Warning Signs to Watch:**
- 🚨 **"Memory limit reached"** → Reduce batch size further
- 🚨 **Model loading fails** → Close more applications
- 🚨 **MPS errors** → May fall back to CPU

## 💡 **If Still Issues:**

1. **Close more apps:** Browser tabs, Slack, etc.
2. **Restart Python:** Fresh memory state
3. **CPU fallback:** Set `device = "cpu"` if MPS issues
4. **Reduce further:** Can go to 3GB or 2.5GB limit if needed

---

**Try it now!** The system should work comfortably within your 4.4GB available memory. 🚀

```bash
python test_apple_silicon.py
```
