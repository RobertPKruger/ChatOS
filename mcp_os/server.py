# server.py - Updated with web search capabilities
"""
Main MCP server with modular tool registration including web search
"""

from mcp.server.fastmcp import FastMCP
import logging

# Create the main MCP server
mcp = FastMCP("local-os")

# Import and register modular tool sets
import fs_tools
from app_tools import AppToolsManager, register_app_tools
from web_search_tools import WebSearchManager, register_web_search_tools

def main():
    """Initialize and run the MCP server"""
    
    # Register file system tools
    fs_tools.register_fs_tools(mcp)
    
    # Initialize and register application tools
    app_manager = AppToolsManager("apps_config.json")
    register_app_tools(mcp, app_manager)
    
    # Initialize and register web search tools
    search_manager = WebSearchManager()
    register_web_search_tools(mcp, search_manager)
    
    # Run the server
    logging.info("Starting MCP server with all tools registered (including web search)")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()