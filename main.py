import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure knowledge_agent is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.infrastructure.llm.ollama_provider import OllamaGemmaProvider
from src.infrastructure.llm.gemini_provider import GeminiFlashProvider
from src.infrastructure.embedding.local_embedder import LocalEmbeddingProvider
from src.infrastructure.storage.fs_repo import MarkdownFileRepository
from src.core.services.dedup_service import DeduplicationService

def main():
    print("Initializing Knowledge Agent...")
    
    # 1. Setup Infrastructure
    # Choose Provider based on Config
    provider_type = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider_type == "gemini":
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        print(f"Using Cloud Provider: {model_name}")
        llm = GeminiFlashProvider(model_name=model_name) # Requires GEMINI_API_KEY env var
    else:
        print("Using Local Provider: Ollama (Gemma 3)")
        llm = OllamaGemmaProvider(model_name="gemma3:12b")
    
    # Check if BGE-M3 is ready
    print("Loading Embedding Model (BAAI/bge-m3)...")
    embedder = LocalEmbeddingProvider(model_name="BAAI/bge-m3")
    
    repo = MarkdownFileRepository()
    
    # 2. Setup Service
    service = DeduplicationService(
        llm=llm,
        embedder=embedder,
        repo=repo,
        threshold=0.65
    )
    
    # 3. Execution
    # Configurable via Environment Variables
    default_zettel = "/home/bhickta/development/upsc/Zettelkasten/DD Basu Polity"
    default_inbox = "/home/bhickta/development/upsc/Inbox"
    
    ZETTELKASTEN_DIR = os.getenv("ZETTELKASTEN_DIR", default_zettel)
    PROCESSED_DIR = os.getenv("PROCESSED_DIR", default_inbox)
    
    # Step A: Build Index
    service.build_index(ZETTELKASTEN_DIR)
    
    # Step B: Process
    service.process_directory(PROCESSED_DIR)
    
    print("Deduplication complete.")

if __name__ == "__main__":
    main()
