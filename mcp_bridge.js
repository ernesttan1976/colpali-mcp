#!/usr/bin/env node
/**
 * MCP Bridge for ColPali HTTP Server
 * Bridges stdio MCP protocol to HTTP
 */

const { spawn } = require('child_process');
const fetch = require('node-fetch');

const COLPALI_SERVER_URL = process.env.COLPALI_SERVER_URL || 'http://127.0.0.1:8080/mcp';

class MCPHttpBridge {
    constructor() {
        this.setupStdio();
    }

    setupStdio() {
        process.stdin.setEncoding('utf8');
        
        let buffer = '';
        process.stdin.on('data', (chunk) => {
            buffer += chunk;
            
            // Process complete JSON messages
            let newlineIndex;
            while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
                const line = buffer.slice(0, newlineIndex);
                buffer = buffer.slice(newlineIndex + 1);
                
                if (line.trim()) {
                    this.handleMessage(line.trim());
                }
            }
        });

        process.stdin.on('end', () => {
            if (buffer.trim()) {
                this.handleMessage(buffer.trim());
            }
        });
    }

    async handleMessage(message) {
        try {
            const data = JSON.parse(message);
            
            // Forward to ColPali HTTP server
            const response = await fetch(COLPALI_SERVER_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            
            // Send response back via stdout
            process.stdout.write(JSON.stringify(result) + '\n');
            
        } catch (error) {
            // Send error response
            const errorResponse = {
                jsonrpc: "2.0",
                id: null,
                error: {
                    code: -32603,
                    message: error.message || 'Internal error'
                }
            };
            
            process.stdout.write(JSON.stringify(errorResponse) + '\n');
        }
    }
}

// Start the bridge
new MCPHttpBridge();
