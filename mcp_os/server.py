# server.py - Updated with enhanced web search capabilities
"""
Main MCP server with enhanced web search tools
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
    
    # Initialize and register enhanced web search tools
    try:
        from web_search_tools import WebSearchManager, register_web_search_tools
        search_manager = WebSearchManager()
        register_web_search_tools(mcp, search_manager)
        logging.info("Enhanced web search tools registered successfully")
    except ImportError as e:
        logging.error(f"Could not import web search tools: {e}")
        logging.warning("Web search functionality will not be available")
    
    # Run the server
    logging.info("Starting MCP server with all tools registered")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()