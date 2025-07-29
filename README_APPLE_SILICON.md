# ColPali Apple Silicon - Optimized for 16GB RAM

🍎 **Apple Silicon optimized version of ColPali for PDF document retrieval**

Designed to run efficiently on Apple Silicon with 16GB RAM, using maximum 8GB memory.

## 🚀 Quick Start

### Prerequisites
- Apple Silicon Mac (M1, M2, M3 series)
- macOS with Python 3.9+
- Homebrew (for poppler installation)
- 16GB RAM (will use max 8GB)

### Installation

1. **Setup environment:**
```bash
# Make setup script executable
chmod +x setup_apple_silicon.sh

# Run setup (installs dependencies and creates virtual environment)
./setup_apple_silicon.sh
```

2. **Activate environment:**
```bash
source venv_apple_silicon/bin/activate
```

3. **Test installation:**
```bash
python test_apple_silicon.py
```

## 🧪 Testing with Flight Manual PDF

The system is configured to test with your A330 Flight Training Manual:
- **Test PDF:** `/Users/ernest/Documents/Scribd/282739699-Flight-Training-Manual-A330-pdf.pdf`
- **Memory Limit:** 8GB maximum usage
- **Batch Processing:** Optimized for large PDFs (1000+ pages)

### Run Full Test
```bash
python app_apple_silicon.py
```

This will:
1. ✅ Load the ColQwen2 model optimized for Apple Silicon
2. 📄 Process the A330 Flight Manual PDF 
3. 💾 Generate and cache embeddings
4. 🔍 Test search queries like "flight controls", "emergency procedures"
5. 📊 Monitor memory usage throughout

## 🎯 Key Optimizations for Apple Silicon

### Memory Management
- **Memory Limit:** Hard 8GB limit with monitoring
- **Batch Processing:** Small batches to prevent memory spikes
- **Garbage Collection:** Aggressive cleanup after operations
- **MPS Cache:** Automatic Apple Silicon GPU cache management

### Performance Features
- **MPS Device:** Uses Apple Silicon GPU when available
- **Float16:** Memory-efficient data types
- **Caching:** Intelligent PDF image and embedding caching
- **Chunked Processing:** Large PDFs processed in manageable chunks

### Storage Optimization
- **Direct File Storage:** Bypasses memory-heavy LanceDB for large documents
- **Compressed Embeddings:** Efficient numpy storage format
- **Metadata Tracking:** Fast document lookup without full embedding load

## 📁 File Structure (Apple Silicon Specific)

```
├── requirements-apple-silicon.txt  # Apple Silicon optimized dependencies
├── setup_apple_silicon.sh         # One-command setup script
├── test_apple_silicon.py          # Quick functionality test
├── app_apple_silicon.py           # Main application (memory optimized)
├── db_apple_silicon.py            # Memory-efficient database
└── data/
    ├── embeddings_db/              # Embedding storage
    └── image_cache/                # PDF image cache
```

## 🔍 Usage Examples

### Basic Search
```python
from app_apple_silicon import AppleSiliconColPali

# Initialize with 8GB memory limit
colpali = AppleSiliconColPali(max_memory_gb=8.0)

# Load model
colpali.load_model()

# Process PDF
embeddings, images = colpali.process_pdf_batch(
    "/path/to/your/pdf", 
    batch_size=2
)

# Search
results = colpali.search(
    query="emergency procedures", 
    embeddings=embeddings, 
    images=images, 
    k=5
)
```

### Memory Monitoring
```python
from db_apple_silicon import AppleSiliconEmbeddingDB

db = AppleSiliconEmbeddingDB(max_memory_gb=8.0)
stats = db.get_stats()
print(f"Memory usage: {stats['memory_usage_gb']:.2f}GB")
```

## 🛠️ Troubleshooting

### Memory Issues
```bash
# Check memory usage
python -c "import psutil; print(f'{psutil.Process().memory_info().rss / (1024**3):.2f}GB')"

# Force cleanup
python -c "import gc; import torch; gc.collect(); torch.mps.empty_cache() if torch.backends.mps.is_available() else None"
```

### MPS Issues
```bash
# Check MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Test MPS tensor
python -c "import torch; print(torch.tensor([1.0]).to('mps') if torch.backends.mps.is_available() else 'MPS not available')"
```

### PDF Processing Issues
```bash
# Test poppler
pdftoppm -h

# Test PDF conversion
python -c "from pdf2image import convert_from_path; print('PDF processing works')"
```

## 📊 Performance Expectations

### A330 Flight Manual (~20MB, ~300 pages)
- **Model Loading:** ~2-3 minutes (first time)
- **PDF Processing:** ~5-10 minutes (with caching)
- **Search Queries:** ~2-5 seconds per query
- **Memory Usage:** ~6-7GB peak

### Memory Usage Breakdown
- **Model:** ~3-4GB
- **Processing:** ~2-3GB
- **Embeddings:** ~1-2GB
- **Buffer:** ~1GB

## 🔄 Next Steps: MCP Server

Once Apple Silicon testing is complete, we'll add MCP server functionality:

1. **MCP Protocol Integration** - Add server/client communication
2. **Tool Definitions** - Expose ingestion and query as MCP tools
3. **Async Processing** - Handle concurrent requests efficiently
4. **API Interface** - Clean interface for external applications

## 🐛 Known Limitations

- **First Run:** Model download requires internet (3-4GB)
- **Large PDFs:** >1000 pages may exceed memory limits
- **Cold Start:** First query after model load is slower
- **MPS Fallback:** Some operations may fall back to CPU

## 💡 Tips for Best Performance

1. **Restart Python** between different large PDFs
2. **Use SSD storage** for cache and embeddings
3. **Close other applications** to maximize available memory
4. **Monitor Activity Monitor** during processing
5. **Let initial model download complete** before testing

---

**Ready to test?** Run `python test_apple_silicon.py` to verify everything works! 🚀
