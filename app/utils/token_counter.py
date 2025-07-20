# app/utils/token_counter.py
import tiktoken
from typing import List, Dict
from app.utils.logging import logger


class TokenCounter:
    """Utility for counting tokens in conversation history."""
    
    def __init__(self, model: str = "gpt-4"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception as e:
            logger.warning(f"Could not initialize tiktoken for model {model}: {e}")
            # Fallback to cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a single text string."""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def count_conversation_tokens(self, conversation: List[Dict]) -> int:
        """Count total tokens in conversation history."""
        total_tokens = 0
        for message in conversation:
            content = message.get("content", "")
            if isinstance(content, str):
                total_tokens += self.count_tokens(content)
        return total_tokens