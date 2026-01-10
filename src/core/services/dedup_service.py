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
        """
        Main workflow: 
        1. Scan all candidates.
        2. Group by target (Cluster).
        3. Batch merge.
        """
        source_paths = self.repo.list_notes(source_directory)
        
        # Plan: Target Path -> List[(SourceNote, Score)]
        merge_plan = {}
        # List of SourceNotes to import directly
        to_import = []

        print(f"Scanning {len(source_paths)} notes to plan merges...")
        
        # Pass 1: Planning (Compute Embeddings & Groupping) - No API Calls (except embedding)
        for path in tqdm(source_paths, desc="Planning Merges"):
            try:
                source_note = self.repo.read_note(path)
                source_emb = self.embedder.embed(source_note.content)
                
                match_note, score = self.find_best_match(source_emb)
                
                # If match > threshold, add to plan
                if match_note and score >= self.threshold:
                     if match_note.path not in merge_plan:
                         merge_plan[match_note.path] = {"target": match_note, "sources": []}
                     merge_plan[match_note.path]["sources"].append((source_note, score))
                else:
                    to_import.append(source_note)
            except Exception as e:
                print(f"Error planning for {path}: {e}")

        # Pass 2: Execution
        
        # A. Handle NEW files (Imports) - Fast, no LLM
        if to_import:
            print(f"Importing {len(to_import)} new notes...")
            for note in tqdm(to_import, desc="Importing"):
                self._copy_note(note)

        # B. Handle Merges (Batched)
        total_batches = len(merge_plan)
        if total_batches > 0:
            print(f"Executing {total_batches} merge batches (saving API calls)...")
            
            for target_path, data in tqdm(merge_plan.items(), desc="Merging Batches"):
                target = data["target"]
                sources = [s[0] for s in data["sources"]] # Extract note objects
                
                self._batch_merge_notes(target, sources)
                
                # Rate Limiting: Sleep to respect 15 RPM (approx 4s per call)
                # We can be slightly faster if batches are large, but safety first.
                time.sleep(2) 

    def _batch_merge_notes(self, target: Note, sources: List[Note]):
        """Merges multiple source notes into one target note in a single LLM call."""
        
        source_names = [os.path.splitext(os.path.basename(s.path))[0] for s in sources]
        print(f"  [MERGE] {len(sources)} notes -> {os.path.basename(target.path)} ({', '.join(source_names)})")

        # Construct Prompt for Multiple Sources
        sources_text = ""
        for i, src in enumerate(sources, 1):
             sources_text += f"\n--- NEW INFO FRAGMENT {i} ({source_names[i-1]}) ---\n{src.content}\n"

        prompt = f"""
You are an expert knowledge curator updating an existing note.
Your goal is to incorporate information from multiple "NEW INFO FRAGMENTS" into "BASE NOTE" with MINIMAL changes to the existing text.

Rules:
1. **LOSSLESS**: Do not delete any information from BASE NOTE.
2. **MINIMAL DIFF**: Keep the structure, headings, and wording of BASE NOTE exactly as is, unless you are correcting a factual error.
3. **INSERTION**: Insert the new facts from the FRAGMENTS into the appropriate sections of BASE NOTE. Group related info.
4. **FORMAT**: Output the final unified Markdown content.

--- BASE NOTE (Keep Structure) ---
{target.content}

{sources_text}

--- MERGED NOTE ---
"""
        try:
            merged_content = self.llm.generate(prompt)
            
            # Append source links
            links = ", ".join([f"[[{name}]]" for name in source_names])
            merged_content += f"\n\n---\n**Merged Sources:** {links}"
            
            # Update Target
            target.content = merged_content
            self.repo.write_note(target)
            
            # Archive All Sources
            for src in sources:
                self.repo.archive_note(src.path)
            
            # Live Index Update
            new_embedding = self.embedder.embed(target.content)
            mtime = time.time()
            self.index_store.upsert_note(target.path, target.content, new_embedding, mtime)
            
        except Exception as e:
            print(f"Batch merge failed for {target.path}: {e}")

    def _copy_note(self, source: Note):
        # print(f"  [NEW] No match found. Importing.") # Reduce spam
        # Logic to determine where to save it in Zettelkasten
        new_path = source.path.replace("Inbox", "Zettelkasten/Imported")
        new_note = Note(path=new_path, content=source.content)
        self.repo.write_note(new_note)
        self.repo.archive_note(source.path)
        
        # Live Index Update
        new_embedding = self.embedder.embed(new_note.content)
        self.index_store.upsert_note(new_note.path, new_note.content, new_embedding, time.time())
