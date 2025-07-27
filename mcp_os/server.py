# server.py - Refactored with cleaner separation of concerns
"""
Main MCP server with modular tool registration
"""

from mcp.server.fastmcp import FastMCP
import logging

# Create the main MCP server
mcp = FastMCP("local-os")

# Import and register modular tool sets
import fs_tools
from app_tools import AppToolsManager, register_app_tools

def main():
    """Initialize and run the MCP server"""
    
    # Register file system tools
    fs_tools.register_fs_tools(mcp)
    
    # Initialize and register application tools
    app_manager = AppToolsManager("apps_config.json")
    register_app_tools(mcp, app_manager)
    
    # Run the server
    logging.info("Starting MCP server with all tools registered")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()