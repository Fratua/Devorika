"""
LLM Provider - Multi-model support with intelligent routing
Supports Claude (Anthropic), GPT (OpenAI), and local models
"""

import os
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import json


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def stream_generate(self, messages: List[Dict[str, str]], **kwargs):
        """Stream a response from the LLM."""
        pass


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider - Our primary and most advanced model."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-5-20250929"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        return self._client

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 8192, **kwargs) -> str:
        """Generate response using Claude."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs
        )
        return response.content[0].text

    def stream_generate(self, messages: List[Dict[str, str]], max_tokens: int = 8192, **kwargs):
        """Stream response using Claude."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs
        ) as stream:
            for text in stream.text_stream:
                yield text


class GPTProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 8192, **kwargs) -> str:
        """Generate response using GPT."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    def stream_generate(self, messages: List[Dict[str, str]], max_tokens: int = 8192, **kwargs):
        """Stream response using GPT."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class LocalLLMProvider(LLMProvider):
    """Local LLM provider using Ollama or similar."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "codellama"):
        self.base_url = base_url
        self.model = model

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using local LLM."""
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False}
            )
            return response.json()["message"]["content"]
        except ImportError:
            raise ImportError("requests package not installed. Run: pip install requests")

    def stream_generate(self, messages: List[Dict[str, str]], **kwargs):
        """Stream response using local LLM."""
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True},
                stream=True
            )
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
        except ImportError:
            raise ImportError("requests package not installed. Run: pip install requests")


class LLMRouter:
    """Intelligent router that selects the best LLM for each task."""

    def __init__(self, primary: str = "claude", fallback: Optional[str] = "gpt"):
        self.providers = {}
        self.primary = primary
        self.fallback = fallback
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available providers."""
        try:
            self.providers["claude"] = ClaudeProvider()
        except Exception as e:
            print(f"Claude provider not available: {e}")

        try:
            self.providers["gpt"] = GPTProvider()
        except Exception as e:
            print(f"GPT provider not available: {e}")

        try:
            self.providers["local"] = LocalLLMProvider()
        except Exception as e:
            print(f"Local provider not available: {e}")

    def get_provider(self, task_type: str = "general") -> LLMProvider:
        """
        Select the best provider for the task.

        Task-specific routing:
        - code_generation: Claude (best for complex code)
        - debugging: Claude (superior reasoning)
        - quick_tasks: GPT or Local (faster)
        - analysis: Claude (deep understanding)
        """
        # Task-specific routing logic
        if task_type in ["code_generation", "debugging", "analysis", "planning"]:
            preferred = "claude"
        elif task_type == "quick_tasks":
            preferred = self.fallback or "gpt"
        else:
            preferred = self.primary

        # Return preferred provider or fallback
        if preferred in self.providers:
            return self.providers[preferred]
        elif self.fallback in self.providers:
            return self.providers[self.fallback]
        elif self.providers:
            return list(self.providers.values())[0]
        else:
            raise RuntimeError("No LLM providers available")

    def generate(self, messages: List[Dict[str, str]], task_type: str = "general", **kwargs) -> str:
        """Generate response using the best provider for the task."""
        provider = self.get_provider(task_type)
        return provider.generate(messages, **kwargs)

    def stream_generate(self, messages: List[Dict[str, str]], task_type: str = "general", **kwargs):
        """Stream response using the best provider for the task."""
        provider = self.get_provider(task_type)
        return provider.stream_generate(messages, **kwargs)
