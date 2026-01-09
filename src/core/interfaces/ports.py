from abc import ABC, abstractmethod
from typing import List, Optional
from src.core.domain.note import Note

class ILLMProvider(ABC):
    """Interface for Large Language Model interactions."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates text based on the provided prompt."""
        pass

class IEmbeddingProvider(ABC):
    """Interface for generating vector embeddings."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generates a vector embedding for the given text."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Returns the dimension of the embeddings."""
        pass

class INoteRepository(ABC):
    """Interface for note storage and retrieval."""
    
    @abstractmethod
    def read_note(self, path: str) -> Note:
        """Reads a note from the given path."""
        pass
    
    @abstractmethod
    def write_note(self, note: Note) -> None:
        """Writes a note to storage."""
        pass
    
    @abstractmethod
    def list_notes(self, directory: str) -> List[str]:
        """Lists all note paths in a directory (recursive)."""
        pass
    
    @abstractmethod
    def archive_note(self, path: str) -> None:
        """Moves a note to an archive location."""
        pass
