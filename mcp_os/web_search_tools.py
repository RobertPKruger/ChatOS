# web_search_tools.py - ENHANCED with better weather and response cleaning
"""
Enhanced web search tools with improved weather data and response formatting
"""

import requests
import json
import logging
from typing import Dict, List, Any, Optional
from urllib.parse import quote
import re

logger = logging.getLogger(__name__)

class WebSearchManager:
    """Enhanced web search operations with better weather support"""
    
    def __init__(self):
        self.search_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def _clean_response_text(self, response_text: str) -> str:
        """Clean response text from any formatting artifacts"""
        if not response_text:
            return "No information available."
        
        # Handle TextContent objects that got stringified
        if "[TextContent(" in response_text:
            # Extract the actual text content
            text_match = re.search(r"text=['\"]([^'\"]+)['\"]", response_text)
            if text_match:
                response_text = text_match.group(1)
            else:
                # Fallback: try to extract any readable text
                clean_text = re.sub(r'\[TextContent\([^)]+\)\]', '', response_text)
                response_text = clean_text.strip()
        
        # Remove other artifacts
        response_text = re.sub(r'annotations=None\)', '', response_text)
        response_text = re.sub(r'type=[\'"]text[\'"]', '', response_text)
        
        return response_text.strip()
    
    def get_weather_openweather(self, location: str) -> str:
        """Get weather using multiple fast APIs with quick timeout"""
        try:
            # Method 1: Try wttr.in with very short timeout (3 seconds)
            location_encoded = quote(location.replace(" ", "+"))
            url = f"http://wttr.in/{location_encoded}?format=3"  # Simpler format
            
            response = requests.get(url, headers=self.search_headers, timeout=3)
            
            if response.status_code == 200:
                weather_data = response.text.strip()
                
                if weather_data and "Unknown location" not in weather_data and len(weather_data) > 5:
                    return f"Weather for {location}: {weather_data}"
            
            # Method 2: Quick fallback using DuckDuckGo
            return self.get_weather_duckduckgo_fast(location)
            
        except Exception as e:
            logger.error(f"Fast weather lookup failed: {e}")
            return self.get_weather_duckduckgo_fast(location)

    def get_weather_duckduckgo_fast(self, location: str) -> str:
        """Fast weather lookup using DuckDuckGo with minimal timeout"""
        try:
            query = f"weather {location} temperature"
            encoded_query = quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1"
            
            response = requests.get(url, headers=self.search_headers, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for weather in instant answer
                if data.get('Answer'):
                    answer = data['Answer']
                    if any(word in answer.lower() for word in ['°', 'temp', 'weather', 'degrees', 'celsius', 'fahrenheit']):
                        return f"Weather for {location}: {answer}"
                
                # Look in abstract
                if data.get('AbstractText'):
                    abstract = data['AbstractText']
                    if any(word in abstract.lower() for word in ['°', 'temp', 'weather', 'degrees']):
                        return f"Weather for {location}: {abstract}"
            
            # Final fallback
            return f"I couldn't find current weather data for {location}. For the most accurate weather, please check weather.com or weather.gov."
            
        except Exception as e:
            logger.error(f"DuckDuckGo weather failed: {e}")
            return f"Weather service temporarily unavailable for {location}. Please try weather.com for current conditions."
    
    def get_weather_alternative(self, location: str) -> str:
        """Alternative weather lookup using web search"""
        try:
            # Search for weather information
            query = f"weather forecast {location} today temperature"
            encoded_query = quote(query)
            
            # Try a simple weather search
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
            
            response = requests.get(url, headers=self.search_headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for weather-related instant answers
                if data.get('Answer'):
                    answer = data['Answer']
                    if any(weather_word in answer.lower() for weather_word in ['°f', '°c', 'temperature', 'weather', 'degrees']):
                        return f"Weather for {location}: {answer}"
                
                if data.get('AbstractText'):
                    abstract = data['AbstractText']
                    if any(weather_word in abstract.lower() for weather_word in ['°f', '°c', 'temperature', 'weather', 'degrees']):
                        return f"Weather information for {location}: {abstract}"
                
                # Check related topics for weather info
                if data.get('RelatedTopics'):
                    for topic in data['RelatedTopics'][:3]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            text = topic['Text']
                            if any(weather_word in text.lower() for weather_word in ['°f', '°c', 'temperature', 'weather', 'degrees']):
                                return f"Weather for {location}: {text}"
            
            # Final fallback
            return f"I couldn't find current weather data for {location}. You might want to check a weather website like weather.com or weather.gov for the most current conditions."
            
        except Exception as e:
            logger.error(f"Alternative weather search error: {e}")
            return f"I'm having trouble getting weather information for {location} right now. Please try again later or check a weather website."
    
    def search_web(self, query: str, num_results: int = 5) -> str:
        """Enhanced web search with better response formatting"""
        try:
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
                    return f"No immediate answers found for '{query}'. You may want to search on a specific website for more detailed information."
            else:
                return f"Search request failed. Status: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search failed: {str(e)}"
    
    def get_stock_price(self, symbol: str) -> str:
        """Get current stock price information - enhanced error handling"""
        try:
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
    """Register enhanced web search tools with the MCP server"""
    
    @mcp.tool()
    def web_search(query: str, num_results: int = 5) -> str:
        """Search the web for current information.
        
        Args:
            query: The search query
            num_results: Maximum number of results to return (default: 5)
        """
        result = search_manager.search_web(query, num_results)
        return search_manager._clean_response_text(result)
    
    @mcp.tool()
    def get_current_stock_price(symbol: str) -> str:
        """Get the current stock price for a given symbol.
        
        Args:
            symbol: The stock symbol (e.g., 'NVDA', 'AAPL', 'MSFT')
        """
        result = search_manager.get_stock_price(symbol)
        return search_manager._clean_response_text(result)
    
    @mcp.tool()
    def search_news(query: str) -> str:
        """Search for current news about a topic.
        
        Args:
            query: The news topic to search for
        """
        result = search_manager.search_web(f"news {query}", 5)
        return search_manager._clean_response_text(result)
    
    @mcp.tool()
    def get_weather(location: str) -> str:
        """Get current weather information for a location.
        
        Args:
            location: The city/location to get weather for
        """
        result = search_manager.get_weather_openweather(location)
        return search_manager._clean_response_text(result)
    
    @mcp.tool()
    def search_definition(term: str) -> str:
        """Get the definition of a term or concept.
        
        Args:
            term: The term to define
        """
        result = search_manager.search_web(f"define {term}", 3)
        return search_manager._clean_response_text(result)
    
    @mcp.tool()
    def get_weather_forecast(location: str) -> str:
        """Get detailed weather forecast for a location.
        
        Args:
            location: The city/location to get forecast for
        """
        result = search_manager.get_weather_openweather(location)
        return search_manager._clean_response_text(result)

