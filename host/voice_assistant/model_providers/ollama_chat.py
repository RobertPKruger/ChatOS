# voice_assistant/model_providers/ollama_chat.py
import json
import logging
import requests   # <─ sync HTTP client

logger = logging.getLogger(__name__)

class OllamaChatProvider:
    """
    Synchronous wrapper for Ollama’s REST API.
    Returns exactly the same object shape as the OpenAI client:
    an object with .choices[0].message.content so the rest of the
    pipeline stays unchanged.
    """

    class _Choice:
        def __init__(self, content: str):
            self.message = type("Msg", (), {"content": content})

    class _Completion:
        def __init__(self, content: str):
            self.choices = [OllamaChatProvider._Choice(content)]

    def __init__(self, model: str = "mistral-small:22b-instruct-2409-q4_0",
                 host: str = "http://localhost:11434",
                 stream: bool = False):
        self.model  = model
        self.url    = f"{host}/api/chat"
        self.stream = stream

    def complete(self, messages, **kwargs):
        """
        `messages` should be a list of dicts like
        { "role": "user", "content": "Hello" }
        Returns _Completion object mimicking OpenAI client.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
            **kwargs,
        }
        try:
            r = requests.post(self.url,
                              data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"},
                              timeout=60)
            r.raise_for_status()
        except requests.RequestException as err:
            raise RuntimeError(f"Ollama request failed: {err}") from err

        data = r.json()
        reply_text = data["message"]["content"]
        return OllamaChatProvider._Completion(reply_text)

    # Optional: synchronous generator for streaming tokens
    def generate_stream(self, messages, **kwargs):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        with requests.post(self.url,
                           data=json.dumps(payload).encode(),
                           headers={"Content-Type": "application/json"},
                           timeout=60,
                           stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                yield chunk["message"]["content"]
