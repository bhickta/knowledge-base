import shutil
import os
import sys

def clear_cache():
    db_path = "chroma_db"
    
    if os.path.exists(db_path):
        print(f"Removing ChromaDB cache at {os.path.abspath(db_path)}...")
        try:
            shutil.rmtree(db_path)
            print("Cache cleared successfully.")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    else:
        print("No cache found to clear.")

if __name__ == "__main__":
    confirm = input("This will delete the entire Knowledge Index (ChromaDB). Are you sure? (y/n): ")
    if confirm.lower() == 'y':
        clear_cache()
    else:
        print("Operation cancelled.")
