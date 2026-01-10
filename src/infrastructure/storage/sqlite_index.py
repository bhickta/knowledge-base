import sqlite3
import os
import json
import numpy as np
from typing import List, Tuple, Optional
from src.core.domain.note import Note

class SQLiteIndexRepository:
    def __init__(self, db_path: str = "knowledge_index.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                path TEXT PRIMARY KEY,
                content_hash TEXT,
                last_modified REAL,
                embedding BLOB
            )
        """)
        conn.commit()
        conn.close()

    def get_last_modified(self, path: str) -> Optional[float]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT last_modified FROM notes WHERE path = ?", (path,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def upsert_note(self, note: Note, embedding: List[float], last_modified: float):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Convert embedding list to bytes (float32)
        emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
        
        cursor.execute("""
            INSERT INTO notes (path, content_hash, last_modified, embedding)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                content_hash=excluded.content_hash,
                last_modified=excluded.last_modified,
                embedding=excluded.embedding
        """, (note.path, str(hash(note.content)), last_modified, emb_bytes))
        conn.commit()
        conn.close()

    def get_all_embeddings(self) -> List[Tuple[str, List[float]]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT path, embedding FROM notes")
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for path, emb_blob in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32).tolist()
            results.append((path, emb))
        return results

    def remove_note(self, path: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM notes WHERE path = ?", (path,))
        conn.commit()
        conn.close()
