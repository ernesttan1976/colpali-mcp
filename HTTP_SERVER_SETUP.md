## ColPali MCP Server Setup Instructions

### 1. Start the HTTP Server

```bash
cd /Volumes/myssd/colpali-mcp
chmod +x start_http_server.sh
./start_http_server.sh
```

You should see:
```
🚀 Starting ColPali FastMCP Server on port 8000
📁 Database: /Volumes/myssd/colpali-mcp/data/embeddings_db
🌐 MCP endpoint: http://localhost:8000/mcp
❤️  Health check: http://localhost:8000/health

Available tools:
  - test_connection
  - health_check  
  - list_documents
  - ingest_pdf
  - search_documents
  - delete_document
```

### 2. Install Proxy Dependencies

```bash
cd /Volumes/myssd/colpali-mcp
npm install
```

### 3. Test the Server

```bash
# Test basic connectivity
curl http://localhost:8000/health

# Test MCP endpoint
curl http://localhost:8000/mcp
```

### 4. Configure Claude Desktop

**File Location:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Configuration:**
```json
{
  "mcpServers": {
    "colpali": {
      "command": "node",
      "args": [
        "/Volumes/myssd/colpali-mcp/mcp_proxy.js"
      ],
      "env": {
        "COLPALI_SERVER_URL": "http://localhost:8000"
      },
      "disabled": false
    }
  }
}
```

### 5. Restart Claude Desktop

After updating the configuration:
1. Quit Claude Desktop completely
2. Restart Claude Desktop
3. The ColPali tools should appear in the tools list

### 6. Available Tools

- **test_connection** - Verify server is running
- **health_check** - Check database and server status
- **list_documents** - Show all ingested documents  
- **ingest_pdf** - Process a PDF file (mock implementation)
- **search_documents** - Search through documents (mock implementation)
- **delete_document** - Remove a document and its embeddings

### Environment Variables

```bash
export COLPALI_PORT=8000                                           # Server port
export COLPALI_DB_PATH="/Volumes/myssd/colpali-mcp/data/embeddings_db"  # Database path
```

### Troubleshooting

1. **Server won't start:** Check if port 8000 is available
2. **Tools don't appear:** Verify Claude Desktop config and restart
3. **Database issues:** Check permissions on data directory
4. **Connection refused:** Ensure server is running and accessible

### Testing Tools

Once connected in Claude Desktop, try:
- "Test the ColPali connection"
- "Check the health of the ColPali server"
- "List all documents in the database"
