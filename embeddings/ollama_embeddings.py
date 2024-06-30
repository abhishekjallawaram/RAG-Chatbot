#pip install ollama

import ollama
import numpy as np

"""
Function to create embeddings using Ollama 'ollama-base' model.

Args:
    texts (list of str): List of text documents to create embeddings for.
    api_key (str): Ollama API key.

Returns:
    np.ndarray: Embeddings for the given texts.
"""

def create_ollama_base_embeddings(texts, api_key):
    """
    Creates embeddings using the Ollama 'ollama-base' model.
    
    Pros:
    - Enterprise-grade embeddings.
    - Tailored for specific domains.
    - Reliable and efficient.
    
    Cons:
    - Costs associated with usage.
    - Limited to available models.
    
    Costs:
    - API usage costs, typically subscription-based.
    
    Use Cases:
    - Specialized applications in finance, healthcare, legal sectors where domain-specific embeddings are beneficial.
    
    Args:
    texts (list of str): List of texts to embed.
    api_key (str): Ollama API key.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    client = ollama.Client(api_key=api_key)
    embeddings = []
    for text in texts:
        response = client.embed_text(text, model="ollama-base")
        embeddings.append(response['embedding'])
    return np.array(embeddings)

def create_ollama_finance_embeddings(texts, api_key):
    """
    Creates embeddings using the Ollama 'ollama-finance' model.
    
    Pros:
    - Specialized for financial domain.
    - Reliable and efficient.
    
    Cons:
    - Costs associated with usage.
    - Limited to financial applications.
    
    Costs:
    - API usage costs, typically subscription-based.
    
    Use Cases:
    - Financial applications requiring domain-specific embeddings.
    
    Args:
    texts (list of str): List of texts to embed.
    api_key (str): Ollama API key.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    client = ollama.Client(api_key=api_key)
    embeddings = []
    for text in texts:
        response = client.embed_text(text, model="ollama-finance")
        embeddings.append(response['embedding'])
    return np.array(embeddings)

def create_ollama_healthcare_embeddings(texts, api_key):
    """
    Creates embeddings using the Ollama 'ollama-healthcare' model.
    
    Pros:
    - Specialized for healthcare domain.
    - Reliable and efficient.
    
    Cons:
    - Costs associated with usage.
    - Limited to healthcare applications.
    
    Costs:
    - API usage costs, typically subscription-based.
    
    Use Cases:
    - Healthcare applications requiring domain-specific embeddings.
    
    Args:
    texts (list of str): List of texts to embed.
    api_key (str): Ollama API key.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    client = ollama.Client(api_key=api_key)
    embeddings = []
    for text in texts:
        response = client.embed_text(text, model="ollama-healthcare")
        embeddings.append(response['embedding'])
    return np.array(embeddings)