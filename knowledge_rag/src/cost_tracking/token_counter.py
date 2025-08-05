"""
Token Counter for OpenAI Models

Uses tiktoken for accurate token counting that matches OpenAI's billing.
"""

import logging
from typing import List, Dict, Any, Optional
import json

# Try to import tiktoken, provide fallback if not available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available. Using fallback token counting (less accurate)")

from .pricing_config import get_model_pricing


class TokenCounter:
    """Accurate token counting for OpenAI models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._encoders = {}  # Cache for encoders
        
        # Model to encoding mapping
        self.model_encodings = {
            "gpt-4": "cl100k_base",
            "gpt-4-32k": "cl100k_base", 
            "gpt-4-turbo": "cl100k_base",
            "gpt-4-turbo-preview": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-3.5-turbo-16k": "cl100k_base",
            "gpt-3.5-turbo-1106": "cl100k_base"
        }
    
    def _get_encoder(self, model_name: str):
        """Get or create tiktoken encoder for model"""
        if not TIKTOKEN_AVAILABLE:
            return None
            
        # Handle model aliases and versions
        base_model = model_name
        for alias in ["gpt-4-0613", "gpt-4-0314"]:
            if model_name.startswith(alias):
                base_model = "gpt-4"
                break
        for alias in ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-0301"]:
            if model_name.startswith(alias):
                base_model = "gpt-3.5-turbo"
                break
        
        if base_model not in self._encoders:
            try:
                encoding_name = self.model_encodings.get(base_model, "cl100k_base")
                self._encoders[base_model] = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                self.logger.warning(f"Failed to get encoder for {base_model}: {e}")
                return None
                
        return self._encoders[base_model]
    
    def count_tokens_in_text(self, text: str, model_name: str = "gpt-3.5-turbo") -> int:
        """
        Count tokens in a text string
        
        Args:
            text: Text to count tokens for
            model_name: OpenAI model name
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
            
        encoder = self._get_encoder(model_name)
        if encoder:
            try:
                return len(encoder.encode(text))
            except Exception as e:
                self.logger.warning(f"tiktoken encoding failed: {e}, using fallback")
        
        # Fallback: rough estimation (less accurate)
        return self._fallback_token_count(text)
    
    def count_tokens_in_messages(self, messages: List[Dict[str, str]], model_name: str = "gpt-3.5-turbo") -> int:
        """
        Count tokens in a list of chat messages
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_name: OpenAI model name
            
        Returns:
            Total token count for the messages
        """
        if not messages:
            return 0
            
        encoder = self._get_encoder(model_name)
        if not encoder:
            # Fallback for all message content
            total_text = " ".join(msg.get("content", "") for msg in messages)
            return self._fallback_token_count(total_text)
        
        try:
            # Use tiktoken's method for counting message tokens
            # This accounts for the message structure overhead
            return self._count_message_tokens_with_tiktoken(messages, model_name, encoder)
        except Exception as e:
            self.logger.warning(f"Message token counting failed: {e}, using fallback")
            total_text = " ".join(msg.get("content", "") for msg in messages)
            return self._fallback_token_count(total_text)
    
    def _count_message_tokens_with_tiktoken(self, messages: List[Dict[str, str]], model_name: str, encoder) -> int:
        """Count tokens in messages using tiktoken, accounting for chat format overhead"""
        
        # Token overhead per message (varies by model)
        if "gpt-3.5-turbo" in model_name:
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-4" in model_name:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            tokens_per_message = 3
            tokens_per_name = 1
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if value:  # Only count non-empty values
                    num_tokens += len(encoder.encode(str(value)))
                    if key == "name":
                        num_tokens += tokens_per_name
        
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    
    def count_tokens_in_response(self, response_text: str, model_name: str = "gpt-3.5-turbo") -> int:
        """
        Count tokens in an API response
        
        Args:
            response_text: The response text from OpenAI
            model_name: OpenAI model name
            
        Returns:
            Number of tokens in the response
        """
        return self.count_tokens_in_text(response_text, model_name)
    
    def _fallback_token_count(self, text: str) -> int:
        """
        Fallback token counting when tiktoken is not available
        Uses rough estimation: ~4 characters per token
        """
        if not text:
            return 0
        
        # Simple heuristic: average of 4 characters per token
        # This is less accurate but better than nothing
        char_count = len(text)
        
        # Adjust for different types of content
        # Code and structured data tend to have more tokens per character
        if any(indicator in text.lower() for indicator in ['json', '{', '}', '[', ']', 'def ', 'class ', 'import ']):
            # More tokens for code/structured content
            return max(1, int(char_count / 3))
        else:
            # Standard text
            return max(1, int(char_count / 4))
    
    def estimate_max_response_tokens(self, model_name: str, input_tokens: int) -> int:
        """
        Estimate maximum response tokens based on model context window
        
        Args:
            model_name: OpenAI model name
            input_tokens: Number of input tokens
            
        Returns:
            Estimated maximum response tokens
        """
        pricing = get_model_pricing(model_name)
        if not pricing:
            return 1000  # Default fallback
        
        context_window = pricing.get("context_window", 4096)
        
        # Reserve some tokens for safety margin
        safety_margin = 100
        max_response = context_window - input_tokens - safety_margin
        
        return max(100, max_response)  # At least 100 tokens
    
    def get_token_stats(self, text: str, model_name: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """
        Get detailed token statistics for text
        
        Args:
            text: Text to analyze
            model_name: OpenAI model name
            
        Returns:
            Dictionary with token statistics
        """
        token_count = self.count_tokens_in_text(text, model_name)
        char_count = len(text)
        
        return {
            "token_count": token_count,
            "character_count": char_count,
            "characters_per_token": char_count / token_count if token_count > 0 else 0,
            "model": model_name,
            "uses_tiktoken": TIKTOKEN_AVAILABLE,
            "word_count": len(text.split()) if text else 0
        } 