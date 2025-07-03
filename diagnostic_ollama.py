#!/usr/bin/env python3
"""
Diagnostic script to test Ollama connectivity and model availability
Run this before starting your chat OS to verify local setup
"""

import json
import requests
import sys
from typing import List, Dict

def test_ollama_connection(host: str = "http://localhost:11434") -> bool:
    """Test basic connectivity to Ollama"""
    print(f"🔍 Testing connection to Ollama at {host}...")
    
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        response.raise_for_status()
        print("✅ Ollama is running and accessible")
        return True
    except requests.ConnectionError:
        print("❌ Cannot connect to Ollama. Is it running?")
        print("   Try: ollama serve")
        return False
    except requests.Timeout:
        print("❌ Connection to Ollama timed out")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False

def list_available_models(host: str = "http://localhost:11434") -> List[str]:
    """List all available models in Ollama"""
    print(f"\n🔍 Checking available models...")
    
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        models = [model['name'] for model in data.get('models', [])]
        
        if models:
            print("✅ Available models:")
            for model in models:
                print(f"   - {model}")
        else:
            print("❌ No models found in Ollama")
            print("   Try: ollama pull mistral")
        
        return models
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []

def test_model_chat(host: str, model: str) -> bool:
    """Test chat completion with specific model"""
    print(f"\n🔍 Testing chat with model '{model}'...")
    
    test_messages = [
        {"role": "user", "content": "Hello! Please respond with just 'Hi there!' to confirm you're working."}
    ]
    
    payload = {
        "model": model,
        "messages": test_messages,
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{host}/api/chat",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        reply = data.get("message", {}).get("content", "")
        
        print(f"✅ Model responded: {reply}")
        return True
        
    except requests.Timeout:
        print(f"❌ Model '{model}' timed out (>30s)")
        return False
    except requests.HTTPError as e:
        print(f"❌ HTTP error with model '{model}': {e}")
        if hasattr(e, 'response') and e.response:
            try:
                error_detail = e.response.json()
                print(f"   Error detail: {error_detail}")
            except:
                print(f"   Response text: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Error testing model '{model}': {e}")
        return False

def recommend_models():
    """Recommend good models for chat OS"""
    print(f"\n💡 Recommended models for ChatOS:")
    print("   Small & Fast:")
    print("   - ollama pull gemma2:2b")
    print("   - ollama pull phi3:3.8b")
    print("")
    print("   Medium (good balance):")
    print("   - ollama pull mistral:7b")
    print("   - ollama pull llama3.1:8b")
    print()
    print("   Large (best quality):")
    print("   - ollama pull mistral-small:22b")  # Your current config
    print("   - ollama pull llama3.1:70b")

def main():
    """Run full diagnostic"""
    print("🚀 ChatOS Ollama Diagnostic Tool")
    print("=" * 40)
    
    host = "http://localhost:11434"
    target_model = "mistral-small:22b-instruct-2409-q4_0"  # From your config
    
    # Test 1: Basic connectivity
    if not test_ollama_connection(host):
        print(f"\n❌ Cannot proceed without Ollama connection")
        print(f"   Make sure Ollama is installed and running:")
        print(f"   1. Install: https://ollama.ai/download")
        print(f"   2. Run: ollama serve")
        sys.exit(1)
    
    # Test 2: List models
    available_models = list_available_models(host)
    
    # Test 3: Check target model
    if target_model in available_models:
        print(f"\n✅ Target model '{target_model}' is available")
        test_model_chat(host, target_model)
    else:
        print(f"\n❌ Target model '{target_model}' not found")
        print(f"   Install it with: ollama pull {target_model}")
        
        # Try with a simpler model name
        simple_name = target_model.split(':')[0]  # Just "mistral-small"
        if any(simple_name in model for model in available_models):
            print(f"   Found similar model with '{simple_name}' - testing that instead...")
            similar_model = next(model for model in available_models if simple_name in model)
            test_model_chat(host, similar_model)
    
    # Test 4: Recommendations
    if not available_models:
        recommend_models()
    
    print(f"\n🎯 Diagnostic complete!")
    print(f"   If all tests passed, your ChatOS should work with local models.")
    print(f"   If tests failed, check the error messages above.")

if __name__ == "__main__":
    main()