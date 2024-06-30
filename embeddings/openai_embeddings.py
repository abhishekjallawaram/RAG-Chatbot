#pip install openai

import openai
import numpy as np
"""
Function to create embeddings using OpenAI Ada model.

Args:
    texts (list of str): List of text documents to create embeddings for.
    api_key (str): OpenAI API key.

Returns:
    np.ndarray: Embeddings for the given texts.
"""
def create_openai_ada_embeddings(texts, api_key):
    """
    Creates embeddings using the OpenAI 'text-embedding-ada-002' model.
    
    Pros:
    - High-quality embeddings.
    - Easy to integrate with existing applications.
    - Continuously updated with new improvements.
    
    Cons:
    - Costs can add up with extensive use.
    - Dependency on external service.
    
    Costs:
    - API usage costs, charged per request.
    
    Use Cases:
    - High-stakes applications needing accurate embeddings.
    - Enterprises and large-scale deployments.
    
    Args:
    texts (list of str): List of texts to embed.
    api_key (str): OpenAI API key.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    openai.api_key = api_key
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings)

def create_openai_babbage_embeddings(texts, api_key):
    """
    Creates embeddings using the OpenAI 'text-similarity-babbage-001' model.
    
    Pros:
    - Good trade-off between performance and cost.
    
    Cons:
    - May not be as powerful as larger models.
    
    Use Cases:
    - Applications needing balanced performance and cost.
    
    Args:
    texts (list of str): List of texts to embed.
    api_key (str): OpenAI API key.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    openai.api_key = api_key
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=text, model="text-similarity-babbage-001")
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings)

def create_openai_curie_embeddings(texts, api_key):
    """
    Creates embeddings using the OpenAI 'text-similarity-curie-001' model.
    
    Pros:
    - Strong performance for many tasks.
    
    Cons:
    - Higher cost compared to smaller models.
    
    Use Cases:
    - Applications needing high-quality embeddings without using the largest models.
    
    Args:
    texts (list of str): List of texts to embed.
    api_key (str): OpenAI API key.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    openai.api_key = api_key
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=text, model="text-similarity-curie-001")
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings)

def create_openai_davinci_embeddings(texts, api_key):
    """
    Creates embeddings using the OpenAI 'text-similarity-davinci-001' model.
    
    Pros:
    - Highest quality embeddings offered by OpenAI.
    
    Cons:
    - Highest cost among OpenAI models.
    
    Use Cases:
    - Applications requiring the best possible performance.
    
    Args:
    texts (list of str): List of texts to embed.
    api_key (str): OpenAI API key.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    openai.api_key = api_key
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=text, model="text-similarity-davinci-001")
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings)

