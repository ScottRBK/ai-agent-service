"""Tests for DocumentProcessor class - evaluation-specific document handling"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from app.evaluation.document_processor import DocumentProcessor
from app.models.resources.knowledge_base import DocumentType


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""

    @patch('app.evaluation.document_processor.settings')
    @patch('app.evaluation.document_processor.DocumentLoader.load_file')
    def test_load_document_with_evaluation_path_resolution(self, mock_load_file, mock_settings):
        """Test that DocumentProcessor resolves paths using EVALUATION_INPUT_DIR"""
        # Setup
        mock_settings.EVALUATION_INPUT_DIR = "/evaluation/input"
        mock_load_file.return_value = "test document content"
        
        # Test relative path resolution
        result = DocumentProcessor.load_document("test_document.md")
        
        # Verify path resolution and delegation
        mock_load_file.assert_called_once()
        called_path = mock_load_file.call_args[0][0]
        assert str(called_path) == "/evaluation/input/test_document.md"
        assert result == "test document content"

    @patch('app.evaluation.document_processor.DocumentLoader.load_file')
    def test_load_document_with_absolute_path(self, mock_load_file):
        """Test that absolute paths bypass evaluation directory resolution"""
        # Setup
        mock_load_file.return_value = "absolute path content"
        absolute_path = "/absolute/path/to/document.txt"
        
        # Test absolute path handling
        result = DocumentProcessor.load_document(absolute_path)
        
        # Verify absolute path is used directly
        mock_load_file.assert_called_once()
        called_path = mock_load_file.call_args[0][0]
        assert str(called_path) == absolute_path
        assert result == "absolute path content"

    @patch('app.evaluation.document_processor.DocumentLoader.parse_metadata')
    def test_parse_frontmatter_delegates_to_parent(self, mock_parse_metadata):
        """Test that frontmatter parsing properly delegates to parent DocumentLoader"""
        # Setup
        test_content = "---\ntitle: Test\n---\nContent here"
        expected_metadata = {"title": "Test"}
        expected_content = "Content here"
        mock_parse_metadata.return_value = (expected_metadata, expected_content)
        
        # Test frontmatter parsing
        metadata, content = DocumentProcessor.parse_frontmatter(test_content)
        
        # Verify delegation to parent with correct parameters
        mock_parse_metadata.assert_called_once_with(test_content, DocumentType.TEXT)
        assert metadata == expected_metadata
        assert content == expected_content