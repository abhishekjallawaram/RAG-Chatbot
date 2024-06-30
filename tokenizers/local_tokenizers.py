
"""
Tokenizers for local models.

This module contains tokenizers for various local models. Tokenizers convert text into token IDs that can be processed by embedding models.

Available Tokenizers:
- MiniLM
- BERT
- RoBERTa
- DistilBERT
- T5
- GPT-2
"""

from transformers import AutoTokenizer

def get_local_minilm_tokenizer():
    """
    Returns a tokenizer for the local MiniLM model.
    
    Returns:
        tokenizer (transformers.PreTrainedTokenizer): The MiniLM tokenizer.
    """
    return AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def get_local_bert_tokenizer():
    """
    Returns a tokenizer for the local BERT model.
    
    Returns:
        tokenizer (transformers.PreTrainedTokenizer): The BERT tokenizer.
    """
    return AutoTokenizer.from_pretrained('bert-base-uncased')

def get_local_roberta_tokenizer():
    """
    Returns a tokenizer for the local RoBERTa model.
    
    Returns:
        tokenizer (transformers.PreTrainedTokenizer): The RoBERTa tokenizer.
    """
    return AutoTokenizer.from_pretrained('roberta-base')

def get_local_distilbert_tokenizer():
    """
    Returns a tokenizer for the local DistilBERT model.
    
    Returns:
        tokenizer (transformers.PreTrainedTokenizer): The DistilBERT tokenizer.
    """
    return AutoTokenizer.from_pretrained('distilbert-base-uncased')

def get_local_t5_tokenizer():
    """
    Returns a tokenizer for the local T5 model.
    
    Returns:
        tokenizer (transformers.PreTrainedTokenizer): The T5 tokenizer.
    """
    return AutoTokenizer.from_pretrained('t5-base')

def get_local_gpt2_tokenizer():
    """
    Returns a tokenizer for the local GPT-2 model.
    
    Returns:
        tokenizer (transformers.PreTrainedTokenizer): The GPT-2 tokenizer.
    """
    return AutoTokenizer.from_pretrained('gpt2')

