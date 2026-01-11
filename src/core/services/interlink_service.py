"""
InterlinkService - High-quality semantic interlinking for atomic notes.

Reuses knowledge_agent infrastructure (BGE-M3 + ChromaDB) to discover
and classify relationships between notes in the Zettelkasten.
"""

import os
import re
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict

from tqdm import tqdm

from src.core.domain.note import Note
from src.core.interfaces.ports import ILLMProvider, IEmbeddingProvider
from src.infrastructure.storage.chroma_store import ChromaStore
from src.infrastructure.storage.fs_repo import MarkdownFileRepository

from src.core.services.interlink_prompts import (
    LinkType, 
    CLASSIFY_RELATIONSHIP_PROMPT,
    BATCH_CLASSIFY_PROMPT,
    format_targets_for_batch,
    parse_batch_response
)



@dataclass
class ProposedLink:
    """A proposed interlink between two notes."""
    source_path: str
    target_path: str
    target_name: str
    link_type: LinkType
    similarity_score: float


class InterlinkService:
    """
    Service for discovering and adding high-quality interlinks between atomic notes.
    
    Uses semantic similarity (BGE-M3 embeddings) to find related notes,
    then LLM to classify relationship types for UPSC-optimized categorization.
    """
    
    def __init__(
        self,
        embedder: IEmbeddingProvider,
        llm: ILLMProvider,
        db_path: str = None,
        similarity_threshold: float = 0.45,
        top_k: int = 5
    ):
        self.embedder = embedder
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.repo = MarkdownFileRepository()
        
        # Use knowledge_agent's ChromaDB by default
        if db_path is None:
            # Resolve path relative to this module: src/core/services -> knowledge_agent
            module_dir = os.path.dirname(os.path.abspath(__file__))
            knowledge_agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(module_dir)))
            db_path = os.path.join(knowledge_agent_dir, "chroma_db")
        self.index_store = ChromaStore(persist_path=db_path)
        
    def build_index(self, target_directory: str) -> int:
        """
        Incrementally build/update the embedding index for a directory.
        Returns number of notes indexed.
        """
        print(f"Scanning {target_directory} for notes...")
        paths = self.repo.list_notes(target_directory)
        
        updates = 0
        for path in tqdm(paths, desc="Indexing notes"):
            try:
                mtime = os.path.getmtime(path)
                cached_mtime = self.index_store.get_last_modified(path)
                
                if cached_mtime is None or mtime > cached_mtime:
                    note = self.repo.read_note(path)
                    embedding = self.embedder.embed(note.content)
                    self.index_store.upsert_note(path, note.content, embedding, mtime)
                    updates += 1
            except Exception as e:
                print(f"Error indexing {path}: {e}")
        
        total = self.index_store.count()
        print(f"Index updated: {updates} new/modified, {total} total notes")
        return total

    def find_related_notes(
        self, 
        note: Note, 
        top_k: int = None
    ) -> List[Tuple[Note, float]]:
        """
        Find semantically similar notes using embedding similarity.
        Returns list of (Note, similarity_score) tuples.
        """
        if top_k is None:
            top_k = self.top_k
            
        embedding = self.embedder.embed(note.content)
        # Get more candidates than needed, we'll filter
        candidates = self.index_store.query_similar(embedding, top_k=top_k + 5)
        
        results = []
        for path, score, content in candidates:
            # Skip self-matches
            if path == note.path:
                continue
            # Skip low-similarity matches
            if score < self.similarity_threshold:
                continue
            results.append((Note(path=path, content=content), score))
            if len(results) >= top_k:
                break
                
        return results

    def classify_relationship(self, source: Note, target: Note) -> LinkType:
        """
        Use LLM to classify the relationship between two notes.
        """
        # Truncate content for prompt efficiency
        source_preview = source.content[:800]
        target_preview = target.content[:800]
        
        prompt = CLASSIFY_RELATIONSHIP_PROMPT.format(
            source_content=source_preview,
            target_content=target_preview
        )
        
        try:
            response = self.llm.generate(prompt).strip().upper()
            # Extract the relationship type
            for link_type in LinkType:
                if link_type.value.upper() in response:
                    return link_type
            return LinkType.RELATED  # Default fallback
        except Exception as e:
            print(f"Classification error: {e}")
            return LinkType.RELATED

    def classify_relationships_batch(
        self, 
        source: Note, 
        targets: List[Tuple[Note, float]]
    ) -> List[LinkType]:
        """
        Classify multiple relationships in a single LLM call (more efficient).
        """
        if not targets:
            return []
            
        source_preview = source.content[:600]
        
        # Format targets for batch prompt
        targets_data = []
        for target_note, score in targets:
            name = os.path.splitext(os.path.basename(target_note.path))[0]
            preview = target_note.content[:300]
            targets_data.append((name, preview))
        
        targets_content = format_targets_for_batch(targets_data)
        
        prompt = BATCH_CLASSIFY_PROMPT.format(
            source_content=source_preview,
            targets_content=targets_content
        )
        
        try:
            response = self.llm.generate(prompt)
            return parse_batch_response(response, len(targets))
        except Exception as e:
            print(f"Batch classification error: {e}")
            return [LinkType.RELATED] * len(targets)

    def analyze_note(
        self, 
        note_path: str, 
        top_k: int = None
    ) -> List[ProposedLink]:
        """
        Analyze a single note and return proposed interlinks.
        """
        note = self.repo.read_note(note_path)
        related = self.find_related_notes(note, top_k)
        
        if not related:
            return []
        
        # Batch classify for efficiency
        link_types = self.classify_relationships_batch(note, related)
        
        proposals = []
        for (target_note, score), link_type in zip(related, link_types):
            if link_type == LinkType.UNRELATED:
                continue
                
            target_name = os.path.splitext(os.path.basename(target_note.path))[0]
            proposals.append(ProposedLink(
                source_path=note_path,
                target_path=target_note.path,
                target_name=target_name,
                link_type=link_type,
                similarity_score=score
            ))
        
        return proposals

    def format_categorized_links(self, links: List[ProposedLink]) -> str:
        """
        Format links into UPSC-optimized categorized markdown section.
        """
        if not links:
            return ""
        
        # Group by link type
        by_type: Dict[LinkType, List[ProposedLink]] = defaultdict(list)
        for link in links:
            by_type[link.link_type].append(link)
        
        # Order of sections (most important first for UPSC)
        section_order = [
            LinkType.PREREQUISITE,
            LinkType.RELATED,
            LinkType.COMPARISON,
            LinkType.CAUSE_EFFECT,
            LinkType.EXAMPLE,
            LinkType.PART_OF,
        ]
        
        lines = ["\n\n## Related Notes\n"]
        
        for link_type in section_order:
            if link_type not in by_type:
                continue
            
            display_name = link_type.display_name
            if not display_name:
                continue
                
            link_strs = [f"[[{l.target_name}]]" for l in by_type[link_type]]
            lines.append(f"**{display_name}:** {', '.join(link_strs)}")
        
        return "\n".join(lines)

    def get_existing_links(self, content: str) -> set:
        """Extract existing WikiLinks from note content."""
        pattern = r'\[\[([^\]]+)\]\]'
        return set(re.findall(pattern, content))

    def has_related_section(self, content: str) -> bool:
        """Check if note already has a Related Notes section."""
        return '## Related Notes' in content or '**Related:**' in content

    def apply_links_to_note(
        self, 
        note_path: str, 
        links: List[ProposedLink],
        dry_run: bool = False
    ) -> str:
        """
        Apply proposed links to a note, avoiding duplicates.
        PRESERVES existing links from old-style Related sections.
        Returns the updated content.
        """
        note = self.repo.read_note(note_path)
        content = note.content
        
        # Get existing links to avoid duplicates
        existing = self.get_existing_links(content)
        new_links = [l for l in links if l.target_name not in existing]
        
        if not new_links:
            return content
        
        # Extract links from old-style "---\n**Related:**" format to preserve them
        old_style_links = []
        # Use finditer to locate ALL occurrences of old-style sections
        for match in re.finditer(r'\n\n---\n\*\*Related:\*\*\s*(.*?)(?=\n\n---\n|$)', content, flags=re.DOTALL):
            old_links_text = match.group(1)
            found_links = re.findall(r'\[\[([^\]]+)\]\]', old_links_text)
            old_style_links.extend(found_links)
        
        # Fallback for simpler pattern if the above doesn't catch the last one
        if not old_style_links:
             # Try simpler pattern that grabs everything after the header until end of string 
             # if it's the last thing
             simple_matches = re.finditer(r'\n\n---\n\*\*Related:\*\*\s*(.*)$', content, flags=re.MULTILINE)
             for m in simple_matches:
                 old_style_links.extend(re.findall(r'\[\[([^\]]+)\]\]', m.group(1)))

        # Remove any existing "## Related Notes" section to rebuild it
        content = re.sub(r'\n\n## Related Notes\n.*$', '', content, flags=re.DOTALL)
        
        # Remove ALL old-style "---\n**Related:**" sections
        # The regex matches the separator and the line, plus content until the next separator or end (?)
        # To be safe, we'll just remove the semantic block we know of. 
        # Actually, simpler to replace the specific header line and the following line 
        # but capturing the Variable content is hard in sub.
        # Let's use a robust pattern to strip them out.
        content = re.sub(r'\n\n---\n\*\*Related:\*\*\s*.*?(?=\n\n---|(?:\Z))', '', content, flags=re.DOTALL)
        # Check for any stragglers at the very end of file
        content = re.sub(r'\n\n---\n\*\*Related:\*\*\s*.*$', '', content, flags=re.DOTALL)
        
        # Create ProposedLinks for preserved old-style links (as RELATED type)
        for old_link in old_style_links:
            if old_link not in [l.target_name for l in new_links]:
                new_links.append(ProposedLink(
                    source_path=note_path,
                    target_path="",  # Unknown path for old links
                    target_name=old_link,
                    link_type=LinkType.RELATED,
                    similarity_score=0.0
                ))
        
        # Add categorized links (includes both new and preserved old links)
        categorized_section = self.format_categorized_links(new_links)
        updated_content = content.rstrip() + categorized_section
        
        if not dry_run:
            note.content = updated_content
            self.repo.write_note(note)
        
        return updated_content

    def interlink_directory(
        self,
        directory: str,
        top_k: int = None,
        dry_run: bool = False,
        skip_linked: bool = False,
        progress_callback=None
    ) -> Dict[str, List[ProposedLink]]:
        """
        Interlink all notes in a directory.
        
        Args:
            skip_linked: If True, skip notes that already have a Related Notes section.
                         This saves LLM calls on re-runs.
        
        Returns dict of {note_path: [proposed_links]}
        """
        paths = self.repo.list_notes(directory)
        all_proposals = {}
        skipped = 0
        
        desc = "Analyzing notes (dry-run)" if dry_run else "Interlinking notes"
        for path in tqdm(paths, desc=desc):
            try:
                # Skip already-linked notes if requested
                if skip_linked:
                    note = self.repo.read_note(path)
                    if self.has_related_section(note.content):
                        skipped += 1
                        continue
                
                proposals = self.analyze_note(path, top_k)
                all_proposals[path] = proposals
                
                if proposals and not dry_run:
                    self.apply_links_to_note(path, proposals)
                    
                if progress_callback:
                    progress_callback(path, proposals)
                    
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        if skip_linked and skipped > 0:
            print(f"  ⏭️  Skipped {skipped} already-linked notes")
        
        return all_proposals


def create_service(use_local_llm: bool = False) -> InterlinkService:
    """
    Factory function to create InterlinkService with configured providers.
    """
    from src.infrastructure.embedding.local_embedder import LocalEmbeddingProvider
    
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
    
    return InterlinkService(embedder=embedder, llm=llm)
