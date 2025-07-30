# web_search_tools.py - Add web search capabilities to your MCP server
"""
Web search tools for real-time information retrieval
"""

import requests
import json
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

class WebSearchManager:
    """Manages web search operations"""
    
    def __init__(self):
        # You can use different search APIs - this example uses a simple approach
        self.search_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search_web(self, query: str, num_results: int = 5) -> str:
        """Search the web for current information"""
        try:
            # Simple DuckDuckGo instant answers (no API key required)
            encoded_query = quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
            
            response = requests.get(url, headers=self.search_headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                results = []
                
                # Check for instant answer
                if data.get('Answer'):
                    results.append(f"Answer: {data['Answer']}")
                
                if data.get('AbstractText'):
                    results.append(f"Summary: {data['AbstractText']}")
                
                # Add related topics
                if data.get('RelatedTopics'):
                    for topic in data['RelatedTopics'][:3]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            results.append(f"Related: {topic['Text']}")
                
                if results:
                    return f"Web search results for '{query}':\n\n" + "\n\n".join(results)
                else:
                    return f"No immediate answers found for '{query}'. You may want to visit a financial website for current stock prices."
            else:
                return f"Search request failed. Status: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search failed: {str(e)}"
    
    def get_stock_price(self, symbol: str) -> str:
        """Get current stock price information"""
        try:
            # Use a free API for stock data (example with Alpha Vantage free tier)
            # Note: You might want to add an API key for better results
            
            # Simple approach using Yahoo Finance (unofficial)
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}"
            
            response = requests.get(url, headers=self.search_headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'chart' in data and data['chart']['result']:
                    result = data['chart']['result'][0]
                    meta = result.get('meta', {})
                    
                    current_price = meta.get('regularMarketPrice')
                    previous_close = meta.get('previousClose')
                    
                    if current_price:
                        change = current_price - previous_close if previous_close else 0
                        change_percent = (change / previous_close * 100) if previous_close else 0
                        
                        return f"""Stock Price for {symbol.upper()}:
Current Price: ${current_price:.2f}
Previous Close: ${previous_close:.2f}
Change: ${change:+.2f} ({change_percent:+.2f}%)

Note: This is real-time market data. Prices may vary by source."""
                    else:
                        return f"Could not retrieve current price for {symbol.upper()}"
                else:
                    return f"No data found for symbol {symbol.upper()}"
            else:
                return f"Failed to fetch stock data. Status: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Stock price error: {e}")
            return f"Failed to get stock price: {str(e)}"

def register_web_search_tools(mcp, search_manager: WebSearchManager):
    """Register web search tools with the MCP server"""
    
    @mcp.tool()
    def web_search(query: str, num_results: int = 5) -> str:
        """Search the web for current information.
        
        Args:
            query: The search query
            num_results: Maximum number of results to return (default: 5)
        """
        return search_manager.search_web(query, num_results)
    
    @mcp.tool()
    def get_current_stock_price(symbol: str) -> str:
        """Get the current stock price for a given symbol.
        
        Args:
            symbol: The stock symbol (e.g., 'NVDA', 'AAPL', 'MSFT')
        """
        return search_manager.get_stock_price(symbol)
    
    @mcp.tool()
    def search_news(query: str) -> str:
        """Search for current news about a topic.
        
        Args:
            query: The news topic to search for
        """
        return search_manager.search_web(f"news {query}", 5)
    
    @mcp.tool()
    def get_weather(location: str) -> str:
        """Get current weather information for a location.
        
        Args:
            location: The city/location to get weather for
        """
        return search_manager.search_web(f"weather {location}", 3)
    
    @mcp.tool()
    def search_definition(term: str) -> str:
        """Get the definition of a term or concept.
        
        Args:
            term: The term to define
        """
        return search_manager.search_web(f"define {term}", 3)