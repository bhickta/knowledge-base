import os
import google.generativeai as genai
from src.core.interfaces.ports import ILLMProvider

class GeminiFlashProvider(ILLMProvider):
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API Key is required. Set GEMINI_API_KEY env var.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API failed: {e}")
