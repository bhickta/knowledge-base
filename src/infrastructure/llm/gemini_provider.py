import os
import google.generativeai as genai
from src.core.interfaces.ports import ILLMProvider

class GeminiFlashProvider(ILLMProvider):
    def __init__(self, api_key: str = None, model_name: str = None):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemma-3-27b")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API Key is required. Set GEMINI_API_KEY env var.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Check for 404 (Model not found)
            if "404" in str(e):
                print(f"\\n[ERROR] Model '{self.model_name}' not found.")
                print("Available models:")
                try:
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            print(f" - {m.name}")
                except Exception as list_err:
                    print(f" (Could not list models: {list_err})")
                
            raise RuntimeError(f"Gemini API failed: {e}")
