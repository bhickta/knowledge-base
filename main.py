#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure imports work from knowledge_agent directory
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
load_dotenv(SCRIPT_DIR / ".env")

from src.core.services.dedup_service import create_service

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Agent - Deduplicate and Merge Notes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--directory", "-d",
        type=str,
        help="Inbox directory to process (PROCESSED_DIR)"
    )
    
    parser.add_argument(
        "--zettelkasten", "-z",
        type=str,
        help="Zettelkasten directory (for index/duplicates)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.65,
        help="Similarity threshold (default: 0.65)"
    )
    
    parser.add_argument(
        "--local-llm",
        action="store_true",
        help="Use local Ollama instead of Gemini"
    )
    
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild embedding index before processing"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned merges without modifying files"
    )
    
    args = parser.parse_args()
    
    # Defaults
    default_zettel = os.getenv("ZETTELKASTEN_DIR", "/home/bhickta/development/upsc/Zettelkasten/DD Basu Polity")
    default_inbox = os.getenv("PROCESSED_DIR", "/home/bhickta/development/upsc/Inbox")
    
    zettel_dir = os.path.abspath(args.zettelkasten or default_zettel)
    inbox_dir = os.path.abspath(args.directory or default_inbox)
    
    print("=" * 60)
    print("🧠 Knowledge Agent - Deduplication & Merge")
    print("=" * 60)
    print(f"Inbox:        {inbox_dir}")
    print(f"Zettelkasten: {zettel_dir}")
    print(f"Provider:     {'Local (Ollama)' if args.local_llm else 'Cloud (Gemini)'}")
    print("=" * 60)
    
    # Initialize Service
    service = create_service(use_local_llm=args.local_llm)
    service.threshold = args.threshold
    
    # Step A: Build/Update Index
    print(f"\n📊 Checking index for: {zettel_dir}")
    service.build_index(zettel_dir)
    
    # Step B: Process Inbox
    print(f"\n📥 Processing Inbox: {inbox_dir}")
    if args.dry_run:
        print("   [DRY RUN MODE ENABLED]")
        
    service.process_directory(inbox_dir, dry_run=args.dry_run)
    
    print("\n✅ Deduplication complete.")

if __name__ == "__main__":
    main()
