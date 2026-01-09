import requests
import json
from src.core.interfaces.ports import ILLMProvider

class OllamaGemmaProvider(ILLMProvider):
    def __init__(self, model_name: str = "gemma3:12b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as e:
            # Fallback or error logging could go here
            raise RuntimeError(f"Ollama connection failed: {e}")
