import unittest
from unittest.mock import MagicMock
from src.core.services.dedup_service import DeduplicationService
from src.core.domain.note import Note

class TestDeduplicationService(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_embedder = MagicMock()
        self.mock_repo = MagicMock()
        
        self.service = DeduplicationService(
            llm=self.mock_llm,
            embedder=self.mock_embedder,
            repo=self.mock_repo,
            threshold=0.8
        )

    def test_merge_flow(self):
        # Setup
        target_note = Note(path="zettel/target.md", content="Existing content")
        source_note = Note(path="Processed/source.md", content="New content")
        
        # Mock Repo behavior
        self.mock_repo.list_notes.side_effect = [
            ["zettel/target.md"], # For build_index
            ["Processed/source.md"] # For process_directory
        ]
        self.mock_repo.read_note.side_effect = [target_note, source_note]
        
        # Mock Embeddings (High similarity)
        self.mock_embedder.embed.side_effect = [
            [1.0, 0.0], # Target (Build Index)
            [0.9, 0.1], # Source (Search)
            [1.0, 0.0]  # Re-embed Merged Result (Live Update)
        ]
        
        # Execution
        self.service.build_index("zettel")
        self.service.process_directory("processed")
        
        # Verify
        # Should call LLM to merge
        self.mock_llm.generate.assert_called_once()
        # Should write the updated target note
        self.mock_repo.write_note.assert_called_with(target_note)
        # Should archive the source note
        self.mock_repo.archive_note.assert_called_with("Processed/source.md")

    def test_no_match_flow(self):
        # Setup
        target_note = Note(path="zettel/target.md", content="Apples")
        source_note = Note(path="Processed/source.md", content="Oranges")
        
        self.mock_repo.list_notes.side_effect = [["zettel/target.md"], ["Processed/source.md"]]
        self.mock_repo.read_note.side_effect = [target_note, source_note]
        
        # Mock Embeddings (Low similarity)
        self.mock_embedder.embed.side_effect = [
            [1.0, 0.0], # Target (Build Index)
            [0.0, 1.0], # Source (Search)
            [0.0, 1.0]  # Re-embed New Note (Live Update)
        ]
        
        # Execution
        self.service.build_index("zettel")
        # Index starts with 1 item
        self.assertEqual(len(self.service.index), 1)
        self.service.process_directory("processed")
        
        # Verify
        self.mock_llm.generate.assert_not_called()
        # Should Create New Note (copied)
        self.mock_repo.write_note.assert_called() 
        args, _ = self.mock_repo.write_note.call_args
        # Should have "Imported" in the path now that "Processed" was correctly capitalized in source
        self.assertIn("Imported", args[0].path)
        
        # Live Indexing Verification: Index should now have 2 items (Target + New Source)
        self.assertEqual(len(self.service.index), 2)

if __name__ == '__main__':
    unittest.main()
