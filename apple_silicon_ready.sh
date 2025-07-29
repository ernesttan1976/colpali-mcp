#!/bin/bash

# Make setup script executable
chmod +x setup_apple_silicon.sh

# Make validation script executable  
chmod +x validate_apple_silicon.py

echo "🍎 Apple Silicon ColPali Setup Complete!"
echo ""
echo "📁 Created Files:"
echo "   ✅ requirements-apple-silicon.txt  - Apple Silicon optimized dependencies"
echo "   ✅ setup_apple_silicon.sh         - One-command setup script"
echo "   ✅ validate_apple_silicon.py      - Pre-flight validation"
echo "   ✅ test_apple_silicon.py          - Quick functionality test"
echo "   ✅ app_apple_silicon.py           - Main application (memory optimized)"
echo "   ✅ db_apple_silicon.py            - Memory-efficient database"
echo "   ✅ README_APPLE_SILICON.md        - Apple Silicon documentation"
echo ""
echo "🚀 Quick Start:"
echo "   1. ./setup_apple_silicon.sh        # Install dependencies"
echo "   2. source venv_apple_silicon/bin/activate"
echo "   3. python validate_apple_silicon.py  # Check system"
echo "   4. python test_apple_silicon.py      # Quick test"
echo "   5. python app_apple_silicon.py       # Full test with A330 manual"
echo ""
echo "📊 Memory Optimization:"
echo "   • Maximum 8GB RAM usage"
echo "   • MPS (Apple Silicon GPU) optimized"
echo "   • Batch processing for large PDFs"
echo "   • Intelligent caching and cleanup"
echo ""
echo "📄 Test PDF: /Users/ernest/Documents/Scribd/282739699-Flight-Training-Manual-A330-pdf.pdf"
echo ""
echo "Ready for Apple Silicon testing! 🎯"
