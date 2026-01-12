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
            
        # Robustly move out of Inbox
        # We assume the path contains "Inbox"
        if "Inbox" in path:
            try:
                # Find the 'Inbox' segment and get everything after it
                parts = path.split(os.sep)
                if "Inbox" in parts:
                    inbox_index = parts.index("Inbox")
                    relative_path = os.sep.join(parts[inbox_index + 1:])
                    base_root = os.sep.join(parts[:inbox_index])
                    archive_root = os.path.join(base_root, "Processed_Archive")
                    dest = os.path.join(archive_root, relative_path)
                else:
                    # Fallback if "Inbox" is part of a string but not a folder segment
                    base_root = path.split("Inbox")[0]
                    archive_root = os.path.join(base_root, "Processed_Archive")
                    dest = os.path.join(archive_root, os.path.basename(path))
            except ValueError:
                base_root = os.path.dirname(os.path.dirname(path))
                archive_root = os.path.join(base_root, "Processed_Archive")
                dest = os.path.join(archive_root, os.path.basename(path))
        else:
            # Fallback for weird paths
            archive_root = os.path.join(os.path.dirname(os.path.dirname(path)), "Processed_Archive")
            dest = os.path.join(archive_root, os.path.basename(path))

        print(f"Archiving to {dest}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(path, dest)
