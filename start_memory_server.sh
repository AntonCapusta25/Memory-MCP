# start_memory_server.sh - Startup script for memory server
#!/bin/bash

echo "🧠 Starting Memory MCP Server on Render..."

# Create memory files directory
mkdir -p memory_files

# Set proper permissions
chmod 755 memory_files

# Start the server
echo "🚀 Starting server on port $PORT"
python memory_server.py
