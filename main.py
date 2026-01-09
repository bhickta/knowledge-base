import sys
import os

# Ensure knowledge_agent is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.infrastructure.llm.ollama_provider import OllamaGemmaProvider
from src.infrastructure.embedding.local_embedder import LocalEmbeddingProvider
from src.infrastructure.storage.fs_repo import MarkdownFileRepository
from src.core.services.dedup_service import DeduplicationService

def main():
    print("Initializing Knowledge Agent...")
    
    # 1. Setup Infrastructure
    # Check if Ollama is running? We assume yes or Provider handles error.
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
        threshold=0.85
    )
    
    # 3. Execution
    # TODO: Make these configurable via CLI args
    ZETTELKASTEN_DIR = "/home/bhickta/development/upsc/Zettelkasten"
    PROCESSED_DIR = "/home/bhickta/development/upsc/Processed"
    
    # Step A: Build Index
    service.build_index(ZETTELKASTEN_DIR)
    
    # Step B: Process
    service.process_directory(PROCESSED_DIR)
    
    print("Deduplication complete.")

if __name__ == "__main__":
    main()
