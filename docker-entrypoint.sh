#!/bin/bash
set -e

echo "=== Initializing Colpali Model Environment ==="

# Create necessary directories with explicit structure
mkdir -p /app/data/embeddings_db
mkdir -p /app/models/colqwen2/model
mkdir -p /app/models/colqwen2/processor

# Check if directories exist and are properly mounted
if [ ! -w "/app/models/colqwen2/model" ]; then
    echo "WARNING: Cannot write to model directory. Volume may not be properly mounted!"
else
    echo "✅ Model directory is properly mounted and writable"
    
    # Create a test file to verify write permissions
    echo "Testing write permissions..." > /app/models/colqwen2/model/test_write.tmp
    rm /app/models/colqwen2/model/test_write.tmp
fi

# Set proper permissions
chmod -R 777 /app/data
chmod -R 777 /app/models

# List existing model files
echo "=== Model Directory Contents ==="
ls -la /app/models/colqwen2/model || echo "No model files yet"
ls -la /app/models/colqwen2/processor || echo "No processor files yet"

# Execute the provided command (usually python app.py)
echo "=== Starting Application ==="
exec "$@"