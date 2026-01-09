from dataclasses import dataclass, field
from typing import Dict, Optional, List

@dataclass
class Note:
    """
    Represents a single atomic note in the system.
    """
    path: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @property
    def key(self) -> str:
        """Unique identifier for the note, typically its file path."""
        return self.path
