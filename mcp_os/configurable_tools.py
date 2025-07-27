# configurable_tools.py - Extension for config-driven tool behavior
"""
Extension system that allows defining tool behaviors in configuration
"""

import json
import logging
from typing import Dict, Any, Callable, List
from dataclasses import dataclass

@dataclass
class ToolDefinition:
    """Represents a tool that can be configured via JSON"""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: str  # Function name to call
    category: str = "general"
    enabled: bool = True
    platform_specific: bool = False
    supported_platforms: List[str] = None

class ConfigurableToolSystem:
    """System for registering tools based on configuration"""
    
    def __init__(self, config_path: str = "tool_config.json"):
        self.config_path = config_path
        self.tool_handlers: Dict[str, Callable] = {}
        self.tool_definitions: Dict[str, ToolDefinition] = {}
        self.load_config()
    
    def register_handler(self, name: str, handler: Callable):
        """Register a handler function for a tool"""
        self.tool_handlers[name] = handler
        logging.info(f"Registered tool handler: {name}")
    
    def load_config(self):
        """Load tool definitions from configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            for tool_name, tool_config in config.get("tools", {}).items():
                definition = ToolDefinition(
                    name=tool_name,
                    description=tool_config.get("description", ""),
                    parameters=tool_config.get("parameters", {}),
                    handler=tool_config.get("handler", ""),
                    category=tool_config.get("category", "general"),
                    enabled=tool_config.get("enabled", True),
                    platform_specific=tool_config.get("platform_specific", False),
                    supported_platforms=tool_config.get("supported_platforms", [])
                )
                self.tool_definitions[tool_name] = definition
                
            logging.info(f"Loaded {len(self.tool_definitions)} tool definitions")
            
        except FileNotFoundError:
            logging.warning(f"Tool configuration file not found: {self.config_path}")
        except Exception as e:
            logging.error(f"Error loading tool configuration: {e}")
    
    def register_tools_with_mcp(self, mcp, platform: str = None):
        """Register all configured tools with the MCP server"""
        for tool_name, definition in self.tool_definitions.items():
            if not definition.enabled:
                continue
                
            # Check platform compatibility
            if definition.platform_specific and platform:
                if definition.supported_platforms and platform not in definition.supported_platforms:
                    logging.info(f"Skipping {tool_name} - not supported on {platform}")
                    continue
            
            # Get the handler function
            handler_func = self.tool_handlers.get(definition.handler)
            if not handler_func:
                logging.warning(f"No handler found for tool {tool_name} (handler: {definition.handler})")
                continue
            
            # Create the MCP tool
            self._create_mcp_tool(mcp, definition, handler_func)
    
    def _create_mcp_tool(self, mcp, definition: ToolDefinition, handler_func: Callable):
        """Create and register an MCP tool from a definition"""
        
        # Create a wrapper function that matches the tool signature
        def tool_wrapper(**kwargs):
            try:
                return handler_func(**kwargs)
            except Exception as e:
                logging.error(f"Error in tool {definition.name}: {e}")
                return f"Error executing {definition.name}: {str(e)}"
        
        # Set the function metadata
        tool_wrapper.__name__ = definition.name
        tool_wrapper.__doc__ = definition.description
        
        # Register with MCP
        mcp.tool()(tool_wrapper)
        logging.info(f"Registered configurable tool: {definition.name}")

# Example tool_config.json structure:
EXAMPLE_TOOL_CONFIG = {
    "metadata": {
        "version": "1.0",
        "description": "Configurable tool definitions"
    },
    "tool_categories": {
        "file_operations": "File and directory operations",
        "app_management": "Application launching and management", 
        "system_info": "System information and diagnostics",
        "network": "Network and web operations"
    },
    "tools": {
        "create_project_folder": {
            "description": "Create a new project folder with standard structure",
            "handler": "create_project_folder_handler",
            "category": "file_operations",
            "enabled": True,
            "parameters": {
                "project_name": {
                    "type": "string",
                    "description": "Name of the project"
                },
                "template": {
                    "type": "string", 
                    "description": "Project template to use",
                    "default": "basic"
                }
            }
        },
        "batch_launch_apps": {
            "description": "Launch multiple applications at once",
            "handler": "batch_launch_handler",
            "category": "app_management", 
            "enabled": True,
            "parameters": {
                "app_list": {
                    "type": "array",
                    "description": "List of applications to launch"
                },
                "delay": {
                    "type": "number",
                    "description": "Delay between launches in seconds",
                    "default": 1.0
                }
            }
        },
        "system_health_check": {
            "description": "Perform a comprehensive system health check",
            "handler": "system_health_handler",
            "category": "system_info",
            "enabled": True,
            "platform_specific": True,
            "supported_platforms": ["windows", "linux"],
            "parameters": {
                "detailed": {
                    "type": "boolean",
                    "description": "Include detailed diagnostics",
                    "default": False
                }
            }
        },
        "quick_web_search": {
            "description": "Perform a quick web search and return summarized results",
            "handler": "web_search_handler", 
            "category": "network",
            "enabled": False,  # Disabled by default
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "number",
                    "description": "Maximum number of results",
                    "default": 5
                }
            }
        }
    },
    "tool_groups": {
        "development_setup": {
            "description": "Tools for setting up development environment",
            "tools": ["create_project_folder", "batch_launch_apps"],
            "auto_enable": True
        },
        "system_maintenance": {
            "description": "System maintenance and diagnostics",
            "tools": ["system_health_check"],
            "auto_enable": False
        }
    }
}

def save_example_config(filename: str = "tool_config.json"):
    """Save the example configuration to a file"""
    with open(filename, 'w') as f:
        json.dump(EXAMPLE_TOOL_CONFIG, f, indent=2)
    print(f"Example tool configuration saved to {filename}")

# Example handler implementations
def create_project_folder_handler(project_name: str, template: str = "basic") -> str:
    """Handler for creating project folders"""
    import os
    import pathlib
    
    try:
        base_path = pathlib.Path.home() / "Projects" / project_name
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Create standard folders based on template
        if template == "basic":
            folders = ["src", "docs", "tests"]
        elif template == "web":
            folders = ["src", "public", "assets", "docs", "tests"]
        elif template == "python":
            folders = ["src", "tests", "docs", "scripts", "data"]
        else:
            folders = ["src", "docs"]
            
        for folder in folders:
            (base_path / folder).mkdir(exist_ok=True)
            
        # Create basic files
        (base_path / "README.md").write_text(f"# {project_name}\n\nProject description here.\n")
        (base_path / ".gitignore").write_text("__pycache__/\n*.pyc\n.env\n")
        
        return f"Created project '{project_name}' at {base_path} with template '{template}'"
        
    except Exception as e:
        return f"Failed to create project folder: {e}"

def batch_launch_handler(app_list: List[str], delay: float = 1.0) -> str:
    """Handler for launching multiple apps"""
    import time
    
    # This would use the existing AppToolsManager
    launched = []
    failed = []
    
    for app in app_list:
        try:
            # Simulate app launch (would use real AppToolsManager)
            print(f"Launching {app}...")
            launched.append(app)
            time.sleep(delay)
        except Exception as e:
            failed.append(f"{app}: {str(e)}")
    
    result = f"Successfully launched: {', '.join(launched)}"
    if failed:
        result += f"\nFailed: {', '.join(failed)}"
    
    return result

def system_health_handler(detailed: bool = False) -> str:
    """Handler for system health checks"""
    import platform
    import psutil
    import shutil
    
    try:
        info = {
            "OS": platform.system(),
            "CPU Usage": f"{psutil.cpu_percent(interval=1)}%",
            "Memory Usage": f"{psutil.virtual_memory().percent}%",
            "Disk Usage": f"{psutil.disk_usage('/').percent}%"
        }
        
        if detailed:
            info.update({
                "Boot Time": psutil.boot_time(),
                "Active Processes": len(psutil.pids()),
                "Network Connections": len(psutil.net_connections())
            })
        
        result = "System Health Check:\n"
        for key, value in info.items():
            result += f"  {key}: {value}\n"
            
        return result
        
    except Exception as e:
        return f"System health check failed: {e}"

# Integration example
def setup_configurable_tools(mcp, platform: str):
    """Setup the configurable tool system"""
    tool_system = ConfigurableToolSystem()
    
    # Register handlers
    tool_system.register_handler("create_project_folder_handler", create_project_folder_handler)
    tool_system.register_handler("batch_launch_handler", batch_launch_handler) 
    tool_system.register_handler("system_health_handler", system_health_handler)
    
    # Register tools with MCP
    tool_system.register_tools_with_mcp(mcp, platform)
    
    return tool_system