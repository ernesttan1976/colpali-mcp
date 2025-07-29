#!/bin/bash
# Script to set up model directory structure for ColPali
# Run this before starting the Docker container to ensure model persistence

echo "=== Setting up model directories for ColPali ==="

# Create full directory structure
mkdir -p ./models/colqwen2/model
mkdir -p ./models/colqwen2/processor
mkdir -p ./data/embeddings_db

# Set permissions for Docker container access
chmod -R 777 ./models
chmod -R 777 ./data

echo "✅ Model directories created successfully"
echo "Directory structure:"
find ./models -type d | sort

echo ""
echo "You can now start the Docker container with:"
echo "docker-compose up -d"