
"""
Tokenizers using OpenAI models.

This module contains tokenizers for various OpenAI models. Tokenizers convert text into token IDs that can be processed by embedding models.

Available Tokenizers:
- Ada
"""

from transformers import GPT2Tokenizer

def get_openai_ada_tokenizer():
    """
    Returns a tokenizer for the OpenAI Ada model.
    
    Returns:
        tokenizer (transformers.PreTrainedTokenizer): The Ada tokenizer.
    """
    return GPT2Tokenizer.from_pretrained('openai-gpt')

