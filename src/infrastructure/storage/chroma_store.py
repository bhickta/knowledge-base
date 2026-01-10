import chromadb
from typing import List, Tuple, Optional
import os

class ChromaStore:
    def __init__(self, persist_path: str = "chroma_db", collection_name: str = "zettelkasten"):
        self.persist_path = persist_path
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # Create or Get collection
        # We assume cosine similarity is good
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )

    def get_last_modified(self, path: str) -> Optional[float]:
        try:
            # We use the file path as the ID
            result = self.collection.get(ids=[path], include=["metadatas"])
            if result and result['metadatas']:
                meta = result['metadatas'][0]
                return meta.get('last_modified')
            return None
        except Exception:
            return None

    def upsert_note(self, path: str, content: str, embedding: List[float], last_modified: float):
        self.collection.upsert(
            documents=[content],
            metadatas=[{"last_modified": last_modified, "source": path}],
            ids=[path],
            embeddings=[embedding]
        )

    def query_similar(self, query_embedding: List[float], top_k: int = 1) -> List[Tuple[str, float, str]]:
        """
        Returns list of (path, score, content)
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        
        candidates = []
        if results['ids'] and results['ids'][0]:
            count = len(results['ids'][0])
            for i in range(count):
                path = results['ids'][0][i]
                # Chroma returns DISsimilarity (distance) for cosine? 
                # Wait, for "cosine" space: distance = 1 - similarity.
                # So similarity = 1 - distance.
                distance = results['distances'][0][i]
                score = 1.0 - distance
                
                content = results['documents'][0][i]
                
                candidates.append((path, score, content))
        
        return candidates

    def count(self) -> int:
        return self.collection.count()
