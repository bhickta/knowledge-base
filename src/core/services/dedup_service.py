from typing import List, Tuple
import numpy as np
import pickle
import os
from tqdm import tqdm
from src.core.interfaces.ports import ILLMProvider, IEmbeddingProvider, INoteRepository
from src.core.domain.note import Note

class DeduplicationService:
    def __init__(
        self,
        llm: ILLMProvider,
        embedder: IEmbeddingProvider,
        repo: INoteRepository,
        threshold: float = 0.60,
        cache_path: str = "index.pkl"
    ):
        self.llm = llm
        self.embedder = embedder
        self.repo = repo
        self.threshold = threshold
        self.cache_path = cache_path
        self.index: List[Tuple[Note, List[float]]] = []

    def _save_index(self):
        """Persist the current index to disk."""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.index, f)
            print(f"Index cache saved to {os.path.abspath(self.cache_path)}")
        except Exception as e:
            print(f"Failed to save index cache: {e}")

    def _load_index(self) -> bool:
        """Try to load index from disk. Returns True if successful."""
        if os.path.exists(self.cache_path):
            try:
                print(f"Loading cached index from {self.cache_path}...")
                with open(self.cache_path, 'rb') as f:
                    self.index = pickle.load(f)
                print(f"Loaded {len(self.index)} notes from cache.")
                return True
            except Exception as e:
                print(f"Failed to load cache: {e}")
        else:
            print(f"No cache found at {os.path.abspath(self.cache_path)}")
        return False

    def build_index(self, target_directory: str):
        """Scans the target directory and executes embeddings for all notes."""
        if self._load_index():
            return

        print(f"Building index from {target_directory}...")
        paths = self.repo.list_notes(target_directory)
        for path in tqdm(paths, desc="Indexing Zettelkasten"):
            try:
                note = self.repo.read_note(path)
                embedding = self.embedder.embed(note.content)
                self.index.append((note, embedding))
            except Exception as e:
                print(f"Error indexing {path}: {e}")
        
        self._save_index()
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
            
            if match_note:
                print(f"  Best candidate: {os.path.basename(match_note.path)} (Score: {score:.4f})")
            
            if match_note and score >= self.threshold:
                self._merge_notes(source_note, match_note, score)
            else:
                self._copy_note(source_note)
            # except Exception as e:
            #     print(f"Error processing {path}: {e}")

    def _merge_notes(self, source: Note, target: Note, score: float):
        print(f"  [MERGE] Found match ({score:.2f}) -> {target.path}")
        prompt = f"""
You are an expert knowledge curator updating an existing note.
Your goal is to incorporate new information from "NOTE 2" into "NOTE 1" with MINIMAL changes to the existing text of NOTE 1.

Rules:
1. **LOSSLESS**: Do not delete any information from NOTE 1.
2. **MINIMAL DIFF**: Keep the structure, headings, and wording of NOTE 1 exactly as is, unless you are correcting a factual error.
3. **INSERTION**: Insert the new facts from NOTE 2 into the appropriate sections of NOTE 1. If no section fits, append a new section.
4. **FORMAT**: Output the final unified Markdown content. Do not output conversational text.

--- NOTE 1 (Base Note - Keep Structure) ---
{target.content}

--- NOTE 2 (New Info - Extract & Insert) ---
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
        
        # Persist changes
        self._save_index()

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
        
        # Persist changes
        self._save_index()
