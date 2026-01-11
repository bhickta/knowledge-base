import os
import time
import google.generativeai as genai
from src.core.interfaces.ports import ILLMProvider

class GeminiFlashProvider(ILLMProvider):
    def __init__(self, api_key: str = None, model_name: str = None, rate_limit_rpm: int = 25):
        """
        Args:
            rate_limit_rpm: Requests per minute limit (default: 25 for free tier)
        """
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemma-3-27b")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.rate_limit_rpm = rate_limit_rpm
        self.min_delay = 60.0 / rate_limit_rpm  # seconds between requests
        self.last_request_time = 0
        
        if not self.api_key:
            raise ValueError("Gemini API Key is required. Set GEMINI_API_KEY env var.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str) -> str:
        # Rate limiting: ensure minimum delay between requests
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed
            time.sleep(sleep_time)
        
        try:
            response = self.model.generate_content(prompt)
            self.last_request_time = time.time()
            return response.text
        except Exception as e:
            self.last_request_time = time.time()
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

