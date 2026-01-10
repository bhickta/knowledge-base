from typing import List, Tuple
import os
import time
from tqdm import tqdm
from src.core.interfaces.ports import ILLMProvider, IEmbeddingProvider, INoteRepository
from src.core.domain.note import Note
from src.infrastructure.storage.chroma_store import ChromaStore

class DeduplicationService:
    def __init__(
        self,
        llm: ILLMProvider,
        embedder: IEmbeddingProvider,
        repo: INoteRepository,
        threshold: float = 0.60,
        db_path: str = "chroma_db"
    ):
        self.llm = llm
        self.embedder = embedder
        self.repo = repo
        self.threshold = threshold
        self.index_store = ChromaStore(persist_path=db_path)

    def build_index(self, target_directory: str):
        """Incrementally scans the target directory and updates embeddings."""
        print(f"Scanning {target_directory} for changes...")
        paths = self.repo.list_notes(target_directory)
        
        updates = 0
        for path in tqdm(paths, desc="Indexing Zettelkasten"):
            try:
                # Check timestamps
                mtime = os.path.getmtime(path)
                cached_mtime = self.index_store.get_last_modified(path)
                
                if cached_mtime is None or mtime > cached_mtime:
                    # New or Modified
                    note = self.repo.read_note(path)
                    embedding = self.embedder.embed(note.content)
                    self.index_store.upsert_note(path, note.content, embedding, mtime)
                    updates += 1
            except Exception as e:
                print(f"Error indexing {path}: {e}")
        
        if updates > 0:
            print(f"Updated index with {updates} changes.")
        else:
            print("Index is up to date.")
        
        print(f"Total notes in index: {self.index_store.count()}")

    def find_best_match(self, source_embedding: List[float]) -> Tuple[Note, float]:
        """Finds the closest note in the index using ChromaDB."""
        candidates = self.index_store.query_similar(source_embedding, top_k=1)
        
        if not candidates:
            return None, 0.0

        path, score, content = candidates[0]
        # We can construct a Note from the result
        return Note(path=path, content=content), score

    def process_directory(self, source_directory: str):
        """Main workflow: Iterate source, find match, merge/copy."""
        source_paths = self.repo.list_notes(source_directory)
        
        for path in tqdm(source_paths, desc="Processing New Notes"):
            try:
                source_note = self.repo.read_note(path)
                source_emb = self.embedder.embed(source_note.content)
                
                match_note, score = self.find_best_match(source_emb)
                
                if match_note:
                    print(f"  Best candidate: {os.path.basename(match_note.path)} (Score: {score:.4f})")
                
                if match_note and score >= self.threshold:
                    self._merge_notes(source_note, match_note, score)
                else:
                    self._copy_note(source_note)
            except Exception as e:
                  print(f"Error processing {path}: {e}")

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
        try:
            merged_content = self.llm.generate(prompt)
            
            # Append source link for traceability
            source_name = os.path.splitext(os.path.basename(source.path))[0]
            merged_content += f"\n\n---\n**Merged Source:** [[{source_name}]]"
            
            # Update Target
            target.content = merged_content
            self.repo.write_note(target)
            
            # Archive Source
            self.repo.archive_note(source.path)
            
            # Live Index Update
            new_embedding = self.embedder.embed(target.content)
            mtime = time.time()
            self.index_store.upsert_note(target.path, target.content, new_embedding, mtime)
            
        except Exception as e:
            print(f"Merge failed: {e}")

    def _copy_note(self, source: Note):
        print(f"  [NEW] No match found. Importing.")
        # Logic to determine where to save it in Zettelkasten
        new_path = source.path.replace("Inbox", "Zettelkasten/Imported")
        new_note = Note(path=new_path, content=source.content)
        self.repo.write_note(new_note)
        self.repo.archive_note(source.path)
        
        # Live Index Update
        new_embedding = self.embedder.embed(new_note.content)
        self.index_store.upsert_note(new_note.path, new_note.content, new_embedding, time.time())
