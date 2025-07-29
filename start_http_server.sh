#!/bin/bash
# Start ColPali HTTP Server
set -e

cd /Volumes/myssd/colpali-mcp

# Activate virtual environment
source venv_apple_silicon/bin/activate

# Set environment variables
export COLPALI_DB_PATH="/Volumes/myssd/colpali-mcp/data/embeddings_db"
export COLPALI_PORT="8000"

echo "Starting ColPali HTTP Server..."
echo "Database: $COLPALI_DB_PATH"
echo "Port: $COLPALI_PORT"
echo "URL: http://localhost:$COLPALI_PORT"

# Start the server
python colpali_simple_http_server.py
