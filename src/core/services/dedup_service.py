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

    def process_directory(self, source_directory: str, dry_run: bool = False):
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
                self._import_note(note, dry_run=dry_run)

        # B. Handle Merges and Links (Batched by Target)
        total_batches = len(merge_plan)
        if total_batches > 0:
            print(f"Analyzing {total_batches} potential merge/link batches...")
            
            for target_path, data in tqdm(merge_plan.items(), desc="Processing Batches"):
                target = data["target"]
                
                # Split sources into 'Must Merge' vs 'Classify'
                must_merge = []
                to_verify = []
                
                for src, score in data["sources"]:
                    if score >= 0.88: # High Confidence -> Auto Merge
                        must_merge.append(src)
                    elif score >= 0.70: # Medium Confidence -> Ask LLM
                        to_verify.append(src)
                    else:
                        self._import_note(src)
                
                # Handle Must Merges (Auto)
                if must_merge:
                    self._batch_merge_notes(target, must_merge, dry_run=dry_run)
                
                # Handle Verification (Merge vs Link)
                for src in to_verify:
                    decision = self._classify_match(src, target)
                    if decision == "MERGE":
                        self._batch_merge_notes(target, [src], dry_run=dry_run)
                    else:
                        print(f"  [LINK] '{os.path.basename(src.path)}' -> '{os.path.basename(target.path)}'")
                        self._link_note(src, target, dry_run=dry_run)

    def _classify_match(self, source: Note, target: Note) -> str:
        """Asks LLM if two notes are the SAME concept (Merge) or just RELATED (Link)."""
        prompt = f"""
Compare these two study notes:

NOTE A:
{target.content[:500]}...

NOTE B:
{source.content[:500]}...

Are these two notes describing the EXACT SAME atomic concept, or are they different sub-topics that are just related?
Examples:
- "Veto Power" and "Types of Veto" -> MERGE (Same topic)
- "Veto Power" and "Presidential Appointments" -> LINK (Related under President, but distinct sub-topics)

Respond with exactly one word: MERGE or LINK.
Decision:"""
        try:
            response = self.llm.generate(prompt).strip().upper()
            return "MERGE" if "MERGE" in response else "LINK"
        except Exception as e:
            print(f"Classification failed: {e}")
            return "LINK"

    def _link_note(self, source: Note, target: Note, dry_run: bool = False):
        """Creates the source note as new but adds bidirectional links."""
        base_name = os.path.basename(source.path)
        new_path = os.path.join(os.path.dirname(target.path), base_name)
        
        if os.path.exists(new_path):
             new_path = new_path.replace(".md", "_1.md")

        # Names for links
        source_name = os.path.splitext(os.path.basename(new_path))[0]
        target_name = os.path.splitext(os.path.basename(target.path))[0]

        # 1. Update Source with link to Target
        source.content = self._append_related_link(source.content, target_name)
        new_note = Note(path=new_path, content=source.content)

        if dry_run:
            print(f"  [DRY RUN] Would write new note: {new_note.path}")
        else:
            self.repo.write_note(new_note)
            self.repo.archive_note(source.path)
        
        # 2. Update Target with link to Source
        target.content = self._append_related_link(target.content, source_name)
        
        if dry_run:
            print(f"  [DRY RUN] Would update target note: {target.path}")
        else:
            self.repo.write_note(target)
            
            # Live Index Updates
            for n in [new_note, target]:
                emb = self.embedder.embed(n.content)
                self.index_store.upsert_note(n.path, n.content, emb, time.time())

    def _append_related_link(self, content: str, link_name: str) -> str:
        """Helper to append a WikiLink to a consolidated 'Related' section."""
        import re
        
        # Find existing Related section
        pattern = re.compile(r"\n\n---\n\*\*Related:\*\*\s*(.*)$", re.MULTILINE)
        match = pattern.search(content)
        
        existing_links = set()
        clean_content = content
        
        if match:
            raw_links = match.group(1)
            links = re.findall(r"\[\[(.*?)\]\]", raw_links)
            existing_links.update(links)
            clean_content = pattern.sub("", content)
        
        existing_links.add(link_name)
        sorted_links = sorted(list(existing_links))
        links_str = ", ".join([f"[[{name}]]" for name in sorted_links])
        
        return clean_content.rstrip() + f"\n\n---\n**Related:** {links_str}"

    def _import_note(self, source: Note, dry_run: bool = False):
        """Imports a completely new note."""
        new_path = source.path.replace("Inbox", "Zettelkasten/Imported")
        new_note = Note(path=new_path, content=source.content)
        
        if dry_run:
            print(f"  [DRY RUN] Would import note to: {new_path}")
        else:
            self.repo.write_note(new_note)
            self.repo.archive_note(source.path)
            
            new_embedding = self.embedder.embed(new_note.content)
            self.index_store.upsert_note(new_note.path, new_note.content, new_embedding, time.time())

    def _batch_merge_notes(self, target: Note, sources: List[Note], dry_run: bool = False):
        """Merges multiple source notes into one target note in a single LLM call."""
        
        source_names = [os.path.splitext(os.path.basename(s.path))[0] for s in sources]
        print(f"  [MERGE] {len(sources)} notes -> {os.path.basename(target.path)} ({', '.join(source_names)})")

        if dry_run:
            print(f"  [DRY RUN] Would merge {len(sources)} sources into {target.path}")
            return

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
            
            # --- Metadata & Footer Management ---
            import re
            
            # 1. Update Source Metadata (YAML)
            yaml_pattern = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
            yaml_match = yaml_pattern.search(merged_content)
            
            if yaml_match:
                yaml_content = yaml_match.group(1)
                source_field_pattern = re.compile(r"^Source:\s*(.*)$", re.MULTILINE)
                source_match = source_field_pattern.search(yaml_content)
                
                # Get current sources from fragment YAML
                new_sources = set()
                for src in sources:
                    src_yaml_match = yaml_pattern.search(src.content)
                    if src_yaml_match:
                        src_s_match = source_field_pattern.search(src_yaml_match.group(1))
                        if src_s_match:
                            new_sources.add(src_s_match.group(1).strip())

                if source_match:
                    current_source_val = source_match.group(1).strip()
                    sources_list = [s.strip() for s in re.split(r",|;", current_source_val)]
                    existing_sources = set(sources_list)
                    existing_sources.update(new_sources)
                    
                    new_source_str = ", ".join(sorted(list(existing_sources)))
                    
                    # Safe replacement using span to avoid re.sub escape issues
                    s_start, s_end = source_match.span()
                    new_yaml_content = yaml_content[:s_start] + f"Source: {new_source_str}" + yaml_content[s_end:]
                    
                    # Update merged_content with high-confidence slice replacement
                    y_start, y_end = yaml_match.span()
                    merged_content = merged_content[:y_start] + f"---\n{new_yaml_content}\n---" + merged_content[y_end:]

            # 2. Footers: Merged Sources [[WikiLinks]]
            existing_links = set()
            footer_pattern = re.compile(r"\n\n---\n\*\*Merged Sources:\*\*\s*(.*)$", re.MULTILINE)
            footer_match = footer_pattern.search(merged_content)
            
            clean_content = merged_content
            if footer_match:
                raw_links = footer_match.group(1)
                links = re.findall(r"\[\[(.*?)\]\]", raw_links)
                existing_links.update(links)
                # Empty replacement is safe from escape errors
                clean_content = footer_pattern.sub("", merged_content)

            existing_links.update(source_names)
            sorted_links = sorted(list(existing_links))
            links_str = ", ".join([f"[[{name}]]" for name in sorted_links])
            final_footer = f"\n\n---\n**Merged Sources:** {links_str}"
            
            final_content = clean_content.rstrip() + final_footer
            
            target.content = final_content
            self.repo.write_note(target)
            
            for src in sources:
                self.repo.archive_note(src.path)
            
            new_embedding = self.embedder.embed(target.content)
            self.index_store.upsert_note(target.path, target.content, new_embedding, time.time())
            
        except Exception as e:
            print(f"Batch merge failed for {target.path}: {e}")


def create_service(use_local_llm: bool = False) -> DeduplicationService:
    """Factory function to create DeduplicationService with configured providers."""
    from src.infrastructure.embedding.local_embedder import LocalEmbeddingProvider
    from src.infrastructure.storage.fs_repo import MarkdownFileRepository
    
    provider_type = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if use_local_llm or provider_type == "ollama":
        from src.infrastructure.llm.ollama_provider import OllamaGemmaProvider
        llm = OllamaGemmaProvider(model_name="gemma3:12b")
        print("Using Local LLM: Ollama (Gemma 3)")
    else:
        from src.infrastructure.llm.gemini_provider import GeminiFlashProvider
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        llm = GeminiFlashProvider(model_name=model_name)
        print(f"Using Cloud LLM: {model_name}")
    
    print("Loading Embedding Model (BAAI/bge-m3)...")
    embedder = LocalEmbeddingProvider(model_name="BAAI/bge-m3")
    repo = MarkdownFileRepository()
    
    return DeduplicationService(llm=llm, embedder=embedder, repo=repo)
