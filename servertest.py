#!/usr/bin/env python3
"""
Simple test script to check what your MCP server is actually returning
"""

import json
import subprocess
import sys

def test_mcp_server_direct():
    """Test MCP server by sending JSON-RPC requests directly"""
    
    # Start your server
    server_process = subprocess.Popen(
        [sys.executable, "mcp_os/server.py"],  # Replace with your server file
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Send initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        server_process.stdin.write(json.dumps(init_request) + "\n")
        server_process.stdin.flush()
        
        # Read response
        response = server_process.stdout.readline()
        print("Init response:", response.strip())
        
        # Send initialized notification
        initialized = {
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        }
        server_process.stdin.write(json.dumps(initialized) + "\n")
        server_process.stdin.flush()
        
        # List tools with correct format
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        server_process.stdin.write(json.dumps(list_tools_request) + "\n")
        server_process.stdin.flush()
        
        # Read tools response
        tools_response = server_process.stdout.readline()
        print("Tools response:", tools_response.strip())
        
        # Parse tools response to get correct tool names
        try:
            tools_data = json.loads(tools_response)
            if "result" in tools_data:
                print("Available tools:")
                for tool in tools_data["result"].get("tools", []):
                    print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
        except Exception as e:
            print(f"Error parsing tools: {e}")
        
        # Send tool call request with correct format
        tool_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "launch_app",
                "arguments": {
                    "app": "notepad"
                }
            }
        }
        
        server_process.stdin.write(json.dumps(tool_request) + "\n")
        server_process.stdin.flush()
        
        # Read response
        response = server_process.stdout.readline()
        print("Tool response:", response.strip())
        
        # Parse and extract text
        try:
            response_data = json.loads(response)
            if "result" in response_data and "content" in response_data["result"]:
                for content in response_data["result"]["content"]:
                    if content.get("type") == "text":
                        print(f"SUCCESS - Extracted text: {content['text']}")
            elif "error" in response_data:
                print(f"ERROR: {response_data['error']}")
        except Exception as e:
            print(f"Could not parse response: {e}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        server_process.terminate()

if __name__ == "__main__":
    test_mcp_server_direct()