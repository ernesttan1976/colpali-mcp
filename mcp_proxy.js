#!/usr/bin/env node
/**
 * Simple MCP HTTP proxy for ColPali FastMCP server
 * This script acts as a bridge between Claude Desktop (STDIO) and your HTTP server
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const COLPALI_SERVER_URL = process.env.COLPALI_SERVER_URL || "http://localhost:8000";

// Create MCP server
const server = new McpServer(
  {
    name: "colpali-proxy",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Helper function to make HTTP requests to FastMCP server
async function callFastMCP(toolName, args = {}) {
  try {
    const response = await fetch(`${COLPALI_SERVER_URL}/mcp/tools/${toolName}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(args),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`Error calling FastMCP tool ${toolName}:`, error);
    throw error;
  }
}

// Register tools that proxy to FastMCP server
server.setRequestHandler("tools/list", async () => {
  return {
    tools: [
      {
        name: "test_connection",
        description: "Test the ColPali server connection",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "health_check", 
        description: "Check server health and database status",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "list_documents",
        description: "List all ingested documents",
        inputSchema: {
          type: "object", 
          properties: {},
        },
      },
      {
        name: "ingest_pdf",
        description: "Ingest a PDF document for search",
        inputSchema: {
          type: "object",
          properties: {
            file_path: {
              type: "string",
              description: "Path to PDF file",
            },
            doc_name: {
              type: "string", 
              description: "Optional document name",
            },
          },
          required: ["file_path"],
        },
      },
      {
        name: "search_documents",
        description: "Search ingested documents",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Search query text",
            },
            top_k: {
              type: "integer",
              description: "Number of results to return",
              default: 5,
            },
          },
          required: ["query"],
        },
      },
      {
        name: "delete_document",
        description: "Delete a document and its embeddings",
        inputSchema: {
          type: "object",
          properties: {
            file_path: {
              type: "string",
              description: "Path to PDF file to delete",
            },
          },
          required: ["file_path"],
        },
      },
    ],
  };
});

server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;
  
  try {
    const result = await callFastMCP(name, args);
    
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  } catch (error) {
    return {
      content: [
        {
          type: "text", 
          text: `Error: ${error.message}`,
        },
      ],
      isError: true,
    };
  }
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  
  console.error(`ColPali MCP Proxy connected to ${COLPALI_SERVER_URL}`);
}

main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
