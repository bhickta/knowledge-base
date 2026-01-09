import os
import shutil
from typing import List
from src.core.interfaces.ports import INoteRepository
from src.core.domain.note import Note

class MarkdownFileRepository(INoteRepository):
    def read_note(self, path: str) -> Note:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Note not found at {path}")
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # TODO: Implement basic Frontmatter parsing if needed
        # For now, we store raw content.
        return Note(path=path, content=content)

    def write_note(self, note: Note) -> None:
        os.makedirs(os.path.dirname(note.path), exist_ok=True)
        with open(note.path, 'w', encoding='utf-8') as f:
            f.write(note.content)

    def list_notes(self, directory: str) -> List[str]:
        notes = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".md"):
                    notes.append(os.path.join(root, file))
        return notes

    def archive_note(self, path: str) -> None:
        if not os.path.exists(path):
            return
            
        archive_dir = os.path.join(os.path.dirname(os.path.dirname(path)), "archived")
        filename = os.path.basename(path)
        dest = os.path.join(archive_dir, filename)
        
        os.makedirs(archive_dir, exist_ok=True)
        shutil.move(path, dest)
