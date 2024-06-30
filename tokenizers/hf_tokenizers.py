
"""
Tokenizers using Hugging Face models.

This module contains tokenizers for various Hugging Face models. Tokenizers convert text into token IDs that can be processed by embedding models.

Available Tokenizers:
- MiniLM
- BERT
"""

from transformers import AutoTokenizer

def get_hf_minilm_tokenizer():
    """
    Returns a tokenizer for the Hugging Face MiniLM model.
    
    Returns:
        tokenizer (transformers.PreTrainedTokenizer): The MiniLM tokenizer.
    """
    return AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def get_hf_bert_tokenizer():
    """
    Returns a tokenizer for the Hugging Face BERT model.
    
    Returns:
        tokenizer (transformers.PreTrainedTokenizer): The BERT tokenizer.
    """
    return AutoTokenizer.from_pretrained('bert-base-uncased')

