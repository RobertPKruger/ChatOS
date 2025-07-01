# voice_assistant/model_providers/openai_chat.py
from openai import OpenAI

class OpenAIChatProvider:
    """
    Synchronous, OpenAI-style provider with .complete() so it can be used
    interchangeably with OllamaChatProvider inside FailoverChatProvider.
    """
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model  = model

    def complete(self, messages, **kwargs):
        """
        Mirrors the signature expected by FailoverChatProvider:
        returns an object with .choices[0].message.content.
        """
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )

    # Optional: streaming wrapper if you need it
    def generate_stream(self, messages, **kwargs):
        for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        ):
            yield chunk
