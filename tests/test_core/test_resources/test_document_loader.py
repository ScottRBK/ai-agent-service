# tests/test_core/test_resources/test_document_loader.py
"""
Unit tests for DocumentLoader class.
"""

import pytest
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from app.core.resources.document_loader import DocumentLoader
from app.models.resources.knowledge_base import DocumentType


class TestDocumentLoader:
    """Test cases for DocumentLoader."""

    def test_load_file_text_document(self):
        """Test loading a basic text file."""
        test_content = "This is a test document with some content."
        mock_path = Mock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.suffix = ".txt"
        
        with patch("builtins.open", mock_open(read_data=test_content)):
            result = DocumentLoader.load_file(mock_path)
            
        assert result == test_content

    def test_load_file_markdown_document(self):
        """Test loading a markdown file."""
        test_content = "# Test Document\n\nThis is markdown content."
        mock_path = Mock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.suffix = ".md"
        
        with patch("builtins.open", mock_open(read_data=test_content)):
            result = DocumentLoader.load_file(mock_path)
            
        assert result == test_content

    def test_load_file_json_document(self):
        """Test loading a JSON file."""
        test_data = {"name": "test", "value": 123}
        test_json = json.dumps(test_data)
        mock_path = Mock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.suffix = ".json"
        
        with patch("builtins.open", mock_open(read_data=test_json)):
            result = DocumentLoader.load_file(mock_path)
            
        # Should return formatted JSON
        expected = json.dumps(test_data, indent=2)
        assert result == expected

    def test_load_file_yaml_document(self):
        """Test loading a YAML file."""
        test_data = {"name": "test", "value": 123}
        test_yaml = yaml.dump(test_data)
        mock_path = Mock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.suffix = ".yaml"
        
        with patch("builtins.open", mock_open(read_data=test_yaml)):
            result = DocumentLoader.load_file(mock_path)
            
        # Should return formatted YAML
        expected = yaml.dump(test_data, default_flow_style=False)
        assert result == expected

    def test_load_file_not_found(self):
        """Test error handling for non-existent files."""
        mock_path = Mock(spec=Path)
        mock_path.exists.return_value = False
        mock_path.__str__ = Mock(return_value="/path/to/nonexistent.txt")
        
        with pytest.raises(FileNotFoundError, match="Document not found"):
            DocumentLoader.load_file(mock_path)

    def test_load_file_unsupported_binary_type(self):
        """Test error handling for unsupported binary files."""
        mock_path = Mock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.suffix = ".unknown"
        
        # Simulate UnicodeDecodeError for binary content
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.return_value.read.side_effect = UnicodeDecodeError(
                'utf-8', b'\x00\x01', 0, 1, 'invalid start byte'
            )
            
            with pytest.raises(ValueError, match="Unsupported file type"):
                DocumentLoader.load_file(mock_path)

    def test_detect_document_type_markdown(self):
        """Test DocumentType detection for markdown files."""
        mock_path = Mock(spec=Path)
        mock_path.suffix = ".md"
        
        result = DocumentLoader.detect_document_type(mock_path)
        assert result == DocumentType.MARKDOWN

    def test_detect_document_type_text(self):
        """Test DocumentType detection for text files."""
        mock_path = Mock(spec=Path)
        mock_path.suffix = ".txt"
        
        result = DocumentLoader.detect_document_type(mock_path)
        assert result == DocumentType.TEXT

    def test_detect_document_type_code(self):
        """Test DocumentType detection for code files."""
        mock_path = Mock(spec=Path)
        mock_path.suffix = ".py"
        
        result = DocumentLoader.detect_document_type(mock_path)
        assert result == DocumentType.CODE

    def test_detect_document_type_json(self):
        """Test DocumentType detection for JSON files."""
        mock_path = Mock(spec=Path)
        mock_path.suffix = ".json"
        
        result = DocumentLoader.detect_document_type(mock_path)
        assert result == DocumentType.JSON

    def test_detect_document_type_unknown_extension(self):
        """Test DocumentType detection defaults to TEXT for unknown extensions."""
        mock_path = Mock(spec=Path)
        mock_path.suffix = ".unknown"
        
        result = DocumentLoader.detect_document_type(mock_path)
        assert result == DocumentType.TEXT

    def test_parse_metadata_yaml_frontmatter(self):
        """Test frontmatter extraction from markdown with YAML frontmatter."""
        content = """---
title: Test Document
author: Test Author
tags: [test, documentation]
---

# Main Content

This is the main document content.
"""
        
        metadata, clean_content = DocumentLoader.parse_metadata(content, DocumentType.MARKDOWN)
        
        expected_metadata = {
            "title": "Test Document",
            "author": "Test Author",
            "tags": ["test", "documentation"]
        }
        expected_content = """# Main Content

This is the main document content.
"""
        
        assert metadata == expected_metadata
        assert clean_content == expected_content

    def test_parse_metadata_no_frontmatter(self):
        """Test metadata parsing when no frontmatter is present."""
        content = "# Test Document\n\nJust regular content without frontmatter."
        
        metadata, clean_content = DocumentLoader.parse_metadata(content, DocumentType.MARKDOWN)
        
        assert metadata == {}
        assert clean_content == content

    def test_parse_metadata_invalid_yaml_frontmatter(self):
        """Test handling of invalid YAML frontmatter."""
        content = """---
invalid: yaml: content: [unclosed
---

# Main Content
"""
        
        metadata, clean_content = DocumentLoader.parse_metadata(content, DocumentType.MARKDOWN)
        
        # Should fallback to original content when YAML parsing fails
        assert metadata == {}
        assert clean_content == content

    def test_parse_metadata_json_document(self):
        """Test metadata extraction from JSON documents."""
        json_data = {
            "metadata": {
                "title": "API Documentation",
                "version": "1.0"
            },
            "content": {
                "data": "some content"
            }
        }
        content = json.dumps(json_data)
        
        metadata, clean_content = DocumentLoader.parse_metadata(content, DocumentType.JSON)
        
        expected_metadata = {
            "title": "API Documentation",
            "version": "1.0"
        }
        
        assert metadata == expected_metadata
        assert clean_content == content  # JSON content remains unchanged

    def test_prepare_for_ingestion_complete_workflow(self):
        """Test complete document preparation workflow."""
        test_content = """---
title: Test Document
author: Test Author
---

# Main Content

This is test content.
"""
        additional_metadata = {"source": "test_upload", "category": "documentation"}
        
        mock_path = Mock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.suffix = ".md"
        mock_path.name = "test.md"
        
        with patch("builtins.open", mock_open(read_data=test_content)):
            clean_content, doc_type, combined_metadata = DocumentLoader.prepare_for_ingestion(
                mock_path, additional_metadata
            )
        
        expected_clean_content = """# Main Content

This is test content.
"""
        expected_metadata = {
            "original_filename": "test.md",
            "file_extension": ".md",
            "title": "Test Document",
            "author": "Test Author",
            "source": "test_upload",
            "category": "documentation"
        }
        
        assert clean_content == expected_clean_content
        assert doc_type == DocumentType.MARKDOWN
        assert combined_metadata == expected_metadata

    def test_prepare_for_ingestion_without_additional_metadata(self):
        """Test document preparation without additional metadata."""
        test_content = "Simple text content."
        
        mock_path = Mock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.suffix = ".txt"
        mock_path.name = "simple.txt"
        
        with patch("builtins.open", mock_open(read_data=test_content)):
            clean_content, doc_type, combined_metadata = DocumentLoader.prepare_for_ingestion(mock_path)
        
        expected_metadata = {
            "original_filename": "simple.txt",
            "file_extension": ".txt"
        }
        
        assert clean_content == test_content
        assert doc_type == DocumentType.TEXT
        assert combined_metadata == expected_metadata

    def test_prepare_for_ingestion_file_not_found(self):
        """Test prepare_for_ingestion error handling for missing files."""
        mock_path = Mock(spec=Path)
        mock_path.exists.return_value = False
        mock_path.__str__ = Mock(return_value="/path/to/missing.txt")
        
        with pytest.raises(FileNotFoundError):
            DocumentLoader.prepare_for_ingestion(mock_path)