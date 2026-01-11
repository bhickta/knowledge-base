#!/usr/bin/env python3
"""
Interlink CLI - Add high-quality semantic interlinks to atomic notes.

Usage:
    # Interlink all notes in Zettelkasten (dry-run first)
    python interlink.py --directory /path/to/Zettelkasten --dry-run
    
    # Interlink for real
    python interlink.py --directory /path/to/Zettelkasten
    
    # Interlink single note
    python interlink.py --note /path/to/note.md --top-k 5
    
    # Rebuild embedding index
    python interlink.py --rebuild-index --directory /path/to/Zettelkasten
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure imports work from knowledge_agent directory
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
load_dotenv(SCRIPT_DIR / ".env")

from src.core.services.interlink_service import InterlinkService, create_service, ProposedLink
from src.core.services.interlink_prompts import LinkType


def print_proposals(path: str, proposals: list):
    """Pretty print proposed links for a note."""
    if not proposals:
        return
        
    note_name = os.path.basename(path)
    print(f"\n📄 {note_name}")
    
    for p in proposals:
        emoji = {
            LinkType.PREREQUISITE: "📚",
            LinkType.RELATED: "🔗",
            LinkType.COMPARISON: "⚖️",
            LinkType.CAUSE_EFFECT: "➡️",
            LinkType.EXAMPLE: "💡",
            LinkType.PART_OF: "🧩",
        }.get(p.link_type, "🔗")
        
        print(f"  {emoji} [{p.link_type.value}] → [[{p.target_name}]] ({p.similarity_score:.2f})")


def main():
    parser = argparse.ArgumentParser(
        description="Add high-quality semantic interlinks to atomic notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--directory", "-d",
        type=str,
        help="Directory containing notes to interlink"
    )
    
    parser.add_argument(
        "--note", "-n",
        type=str,
        help="Single note to analyze/interlink"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of related notes to link per note (default: 5)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show proposed links without modifying files"
    )
    
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild embedding index before interlinking"
    )
    
    parser.add_argument(
        "--skip-linked",
        action="store_true",
        help="Skip notes that already have Related Notes section (efficient for re-runs)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Minimum similarity threshold (default: 0.45)"
    )
    
    parser.add_argument(
        "--local-llm",
        action="store_true",
        help="Use local Ollama instead of Gemini"
    )
    
    args = parser.parse_args()
    
    # Validate args
    if not args.directory and not args.note:
        parser.error("Either --directory or --note is required")
    
    # Resolve default directory
    if args.directory:
        directory = os.path.abspath(args.directory)
    else:
        # Use parent of note for index if single note
        directory = os.path.dirname(os.path.abspath(args.note))
    
    print("=" * 60)
    print("🔗 Interlink Agent - UPSC Atomic Notes")
    print("=" * 60)
    
    # Initialize service
    service = create_service(use_local_llm=args.local_llm)
    service.similarity_threshold = args.threshold
    service.top_k = args.top_k
    
    # Build/update index
    if args.rebuild_index or args.directory:
        print(f"\n📊 Building index for: {directory}")
        service.build_index(directory)
    
    # Process
    if args.note:
        # Single note mode
        note_path = os.path.abspath(args.note)
        print(f"\n🔍 Analyzing: {os.path.basename(note_path)}")
        
        proposals = service.analyze_note(note_path, args.top_k)
        print_proposals(note_path, proposals)
        
        if proposals and not args.dry_run:
            service.apply_links_to_note(note_path, proposals)
            print(f"\n✅ Added {len(proposals)} links to note")
        elif args.dry_run:
            print(f"\n📋 [DRY RUN] Would add {len(proposals)} links")
    else:
        # Directory mode
        print(f"\n{'📋 [DRY RUN] ' if args.dry_run else ''}Processing: {directory}")
        
        all_proposals = service.interlink_directory(
            directory,
            top_k=args.top_k,
            dry_run=args.dry_run,
            skip_linked=args.skip_linked,
            progress_callback=lambda p, props: print_proposals(p, props) if props else None
        )
        
        # Summary
        total_links = sum(len(p) for p in all_proposals.values())
        notes_linked = sum(1 for p in all_proposals.values() if p)
        
        print("\n" + "=" * 60)
        print(f"📊 Summary:")
        print(f"   Notes processed: {len(all_proposals)}")
        print(f"   Notes with new links: {notes_linked}")
        print(f"   Total links {'proposed' if args.dry_run else 'added'}: {total_links}")
        
        if args.dry_run:
            print("\n💡 Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
