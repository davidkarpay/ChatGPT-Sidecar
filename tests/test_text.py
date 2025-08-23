"""Tests for text processing and chunking utilities."""

import pytest
from app.text import chunk_text


class TestTextChunking:
    """Test text chunking functionality."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "This is a test. " * 100  # Create text longer than default chunk size
        
        chunks = chunk_text(text)
        
        assert len(chunks) > 1  # Should create multiple chunks
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all('text' in chunk for chunk in chunks)
        assert all('start_char' in chunk for chunk in chunks)
        assert all('end_char' in chunk for chunk in chunks)
    
    def test_chunk_text_with_custom_size(self):
        """Test chunking with custom chunk size."""
        text = "A" * 2000  # 2000 character text
        
        chunks = chunk_text(text, size=500, overlap=50)
        
        # Should create multiple chunks
        assert len(chunks) >= 3
        
        # Check chunk sizes (except possibly the last one)
        for chunk in chunks[:-1]:
            assert len(chunk['text']) <= 500 + 100  # Allow some buffer
        
        # Check overlap
        if len(chunks) > 1:
            first_chunk = chunks[0]['text']
            second_chunk = chunks[1]['text']
            
            # There should be some overlap
            assert first_chunk[-25:] in second_chunk[:75]  # Approximate overlap check
    
    def test_chunk_text_short_text(self):
        """Test chunking with text shorter than chunk size."""
        text = "This is a short text."
        
        chunks = chunk_text(text, size=1000)
        
        assert len(chunks) == 1
        assert chunks[0]['text'] == text
        assert chunks[0]['start_char'] == 0
        assert chunks[0]['end_char'] == len(text)  # end_char is exclusive in implementation
    
    def test_chunk_text_empty_string(self):
        """Test chunking with empty string."""
        text = ""
        
        chunks = chunk_text(text)
        
        assert len(chunks) == 0
    
    def test_chunk_text_whitespace_only(self):
        """Test chunking with whitespace-only text."""
        text = "   \n\t  \n  "
        
        chunks = chunk_text(text)
        
        # Should handle gracefully, either return empty or single chunk
        assert len(chunks) <= 1
    
    def test_chunk_text_preserves_positions(self):
        """Test that chunk positions are correct."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 50  # 1300 characters
        
        chunks = chunk_text(text, size=400, overlap=50)
        
        # Check that positions are consistent
        for i, chunk in enumerate(chunks):
            start = chunk['start_char']
            end = chunk['end_char']
            
            # Extract text using positions
            extracted = text[start:end]
            assert extracted == chunk['text']
            
            # Check continuity (with overlap)
            if i > 0:
                prev_chunk = chunks[i - 1]
                # Should have some overlap or be contiguous
                assert start <= prev_chunk['end_char'] + 1
    
    def test_chunk_text_unicode_handling(self):
        """Test chunking with Unicode characters."""
        text = "Hello 世界! " * 100 + "This contains émojis 🚀🌟 and açcénts."
        
        chunks = chunk_text(text, size=200)
        
        assert len(chunks) > 0
        
        # Check that Unicode is preserved
        combined_text = ''.join(chunk['text'] for chunk in chunks)
        # Allow for some overlap differences
        assert all(char in combined_text for char in set(text) if char != ' ')
    
    def test_chunk_text_newline_handling(self):
        """Test chunking with newlines and paragraphs."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3." * 20
        
        chunks = chunk_text(text, size=100, overlap=20)
        
        assert len(chunks) > 1
        
        # Newlines should be preserved
        for chunk in chunks:
            if '\n' in text[chunk['start_char']:chunk['end_char'] + 1]:
                assert '\n' in chunk['text']
    
    def test_chunk_text_very_long_words(self):
        """Test chunking with very long words."""
        # Create a very long word
        long_word = "a" * 2000
        text = f"Short word {long_word} another short word"
        
        chunks = chunk_text(text, size=500)
        
        assert len(chunks) >= 1
        
        # Should handle long words gracefully (either split or keep in one chunk)
        total_length = sum(len(chunk['text']) for chunk in chunks)
        assert total_length >= len(text) - 100  # Allow for some overlap
    
    def test_chunk_text_consistent_overlap(self):
        """Test that overlap is consistent across chunks."""
        text = "Word " * 500  # Many repeated words
        overlap = 30
        
        chunks = chunk_text(text, size=200, overlap=overlap)
        
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]
                
                # Check that there's actual overlap
                current_end = current_chunk['text'][-overlap:]
                next_start = next_chunk['text'][:overlap]
                
                # Should have some common words due to overlap
                current_words = current_end.split()
                next_words = next_start.split()
                
                if current_words and next_words:
                    # At least some overlap should exist
                    overlap_exists = any(word in next_words for word in current_words[-3:])
                    assert overlap_exists, f"No overlap found between chunks {i} and {i+1}"
    
    def test_chunk_text_boundary_conditions(self):
        """Test chunking at boundary conditions."""
        # Test with chunk size exactly equal to text length
        text = "Exactly sized text for chunk"
        chunk_size = len(text)
        
        chunks = chunk_text(text, size=chunk_size)
        
        assert len(chunks) == 1
        assert chunks[0]['text'] == text
        
        # Test with chunk size one less than text length
        chunks = chunk_text(text, size=chunk_size - 1, overlap=0)
        
        assert len(chunks) >= 1
    
    def test_chunk_text_zero_overlap(self):
        """Test chunking with zero overlap."""
        text = "ABCD " * 100  # 500 characters
        
        chunks = chunk_text(text, size=100, overlap=0)
        
        if len(chunks) > 1:
            # Check that chunks don't overlap
            for i in range(len(chunks) - 1):
                current_end = chunks[i]['end_char']
                next_start = chunks[i + 1]['start_char']
                
                # Next chunk should start after current chunk ends
                assert next_start >= current_end
    
    def test_chunk_text_large_overlap(self):
        """Test chunking with large overlap (close to chunk size)."""
        text = "This is a test sentence. " * 50
        chunk_size = 200
        overlap = 180  # Very large overlap
        
        chunks = chunk_text(text, size=chunk_size, overlap=overlap)
        
        assert len(chunks) >= 1
        
        # Should still create multiple chunks despite large overlap
        if len(text) > chunk_size:
            assert len(chunks) > 1
    
    def test_chunk_text_token_estimation(self):
        """Test that token estimation is included if available."""
        text = "This is a test sentence with multiple words."
        
        chunks = chunk_text(text)
        
        for chunk in chunks:
            # Token estimation might be included
            if 'token_estimate' in chunk:
                assert isinstance(chunk['token_estimate'], int)
                assert chunk['token_estimate'] > 0
    
    def test_chunk_text_metadata_preservation(self):
        """Test that chunk metadata is properly set."""
        text = "Sample text for metadata testing."
        
        chunks = chunk_text(text, size=100)
        
        for i, chunk in enumerate(chunks):
            # Check required fields
            assert 'text' in chunk
            assert 'start_char' in chunk
            assert 'end_char' in chunk
            
            # Check types
            assert isinstance(chunk['text'], str)
            assert isinstance(chunk['start_char'], int)
            assert isinstance(chunk['end_char'], int)
            
            # Check logical consistency
            assert chunk['start_char'] <= chunk['end_char']
            assert len(chunk['text']) == chunk['end_char'] - chunk['start_char']
    
    def test_chunk_text_edge_case_single_character(self):
        """Test chunking with single character."""
        text = "A"
        
        chunks = chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0]['text'] == "A"
        assert chunks[0]['start_char'] == 0
        assert chunks[0]['end_char'] == 1
    
    def test_chunk_text_performance_large_text(self):
        """Test chunking performance with large text."""
        import time
        
        # Create a large text (about 1MB)
        text = "This is a performance test sentence. " * 30000
        
        start_time = time.time()
        chunks = chunk_text(text, size=1000, overlap=100)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # 5 seconds max
        assert len(chunks) > 100  # Should create many chunks
        
        # Verify integrity of first and last chunks
        assert chunks[0]['start_char'] == 0
        assert chunks[-1]['end_char'] <= len(text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])