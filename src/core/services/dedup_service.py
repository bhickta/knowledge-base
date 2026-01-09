from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from src.core.interfaces.ports import ILLMProvider, IEmbeddingProvider, INoteRepository
from src.core.domain.note import Note

class DeduplicationService:
    def __init__(
        self,
        llm: ILLMProvider,
        embedder: IEmbeddingProvider,
        repo: INoteRepository,
        threshold: float = 0.85
    ):
        self.llm = llm
        self.embedder = embedder
        self.repo = repo
        self.threshold = threshold
        self.index: List[Tuple[Note, List[float]]] = []

    def build_index(self, target_directory: str):
        """Scans the target directory and executes embeddings for all notes."""
        print(f"Building index from {target_directory}...")
        paths = self.repo.list_notes(target_directory)
        for path in tqdm(paths, desc="Indexing Zettelkasten"):
            try:
                note = self.repo.read_note(path)
                embedding = self.embedder.embed(note.content)
                self.index.append((note, embedding))
            except Exception as e:
                print(f"Failed to index {path}: {e}")
        print(f"Index built with {len(self.index)} notes.")

    def find_best_match(self, source_embedding: List[float]) -> Tuple[Note, float]:
        """Finds the closest note in the index using Cosine Similarity."""
        if not self.index:
            return None, 0.0

        target_embeddings = np.array([item[1] for item in self.index])
        source_vec = np.array(source_embedding)
        
        # Cosine Similarity: (A . B) / (||A|| * ||B||)
        # BGE-M3 embeddings are normalized, so we just dot product if source is also normalized.
        # But let's be safe and do full calculation or assume embedder normalizes.
        # LocalEmbedder does normalize=True.
        
        scores = np.dot(target_embeddings, source_vec)
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        return self.index[best_idx][0], float(best_score)

    def process_directory(self, source_directory: str):
        """Main workflow: Iterate source, find match, merge/copy."""
        source_paths = self.repo.list_notes(source_directory)
        
        for path in tqdm(source_paths, desc="Processing New Notes"):
            # print(f"Processing candidate: {path}") # tqdm handles logging better
            # try:
            source_note = self.repo.read_note(path)
            source_emb = self.embedder.embed(source_note.content)
            
            match_note, score = self.find_best_match(source_emb)
            
            if match_note and score >= self.threshold:
                self._merge_notes(source_note, match_note, score)
            else:
                self._copy_note(source_note)
            # except Exception as e:
            #     print(f"Error processing {path}: {e}")

    def _merge_notes(self, source: Note, target: Note, score: float):
        print(f"  [MERGE] Found match ({score:.2f}) -> {target.path}")
        prompt = f"""
You are an expert knowledge curator. Merge the following two notes into a single, comprehensive Master Note.
Rules:
1. Preserve ALL unique facts, dates, names, and specific details from both.
2. If conflicts exist, note them explicitly.
3. Use a clear, structured Markdown format.
4. Do NOT output any conversational text, just the Markdown content.

--- NOTE 1 (Existing) ---
{target.content}

--- NOTE 2 (New) ---
{source.content}

--- MERGED NOTE ---
"""
        merged_content = self.llm.generate(prompt)
        
        # Update Target
        target.content = merged_content
        self.repo.write_note(target)
        self.repo.archive_note(source.path)
        
        # Live Index Update: Re-embed the updated target note
        new_embedding = self.embedder.embed(target.content)
        
        # Find and replace the old entry in the index
        # We search by path, assuming path is unique key
        for i, (n, _) in enumerate(self.index):
            if n.path == target.path:
                self.index[i] = (target, new_embedding)
                break

    def _copy_note(self, source: Note):
        print(f"  [NEW] No match found. Importing.")
        # Logic to determine where to save it in Zettelkasten
        # Simplified: Just write to Zettelkasten/Imported/{subfolders}/{filename}
        new_path = source.path.replace("Processed", "Zettelkasten/Imported")
        new_note = Note(path=new_path, content=source.content)
        self.repo.write_note(new_note)
        self.repo.archive_note(source.path)
        
        # Live Index Update: Add new note to index
        new_embedding = self.embedder.embed(new_note.content)
        self.index.append((new_note, new_embedding))
