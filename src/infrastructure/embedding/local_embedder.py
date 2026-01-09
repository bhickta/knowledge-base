from typing import List
from sentence_transformers import SentenceTransformer
from src.core.interfaces.ports import IEmbeddingProvider

class LocalEmbeddingProvider(IEmbeddingProvider):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text: str) -> List[float]:
        # BGE-M3 returns numpy array, convert to list
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
