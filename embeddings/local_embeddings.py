#pip install transformers torch
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

"""
Local Embeddings
----------------
Pros:
- No dependency on external APIs.
- Full control over data and model.
- Cost-effective for large-scale processing.

Cons:
- Requires significant local resources (CPU/GPU, memory).
- Maintenance and updates are manual.

Costs:
- Hardware costs for running models.
- Energy and operational costs.

Use Cases:
- Research and development environments.
- Applications with strict data privacy requirements.
"""

def create_local_minilm_embeddings(texts):
    """
    Creates embeddings using the local 'all-MiniLM-L6-v2' model.
    
    Pros:
    - Lightweight and fast.
    - Suitable for general-purpose embeddings.
    
    Cons:
    - May not capture domain-specific nuances.
    
    Use Cases:
    - General NLP tasks, prototyping, applications needing lightweight models.
    
    Args:
    texts (list of str): List of texts to embed.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1).numpy()
    
    return np.array(embeddings)

def create_local_bert_embeddings(texts):
    """
    Creates embeddings using the local 'bert-base-uncased' model.
    
    Pros:
    - Well-known, robust model.
    - Captures general semantic meanings well.
    
    Cons:
    - Larger and slower than MiniLM.
    
    Use Cases:
    - Applications needing robust general-purpose embeddings.
    
    Args:
    texts (list of str): List of texts to embed.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1).numpy()
    
    return np.array(embeddings)

def create_local_roberta_embeddings(texts):
    """
    Creates embeddings using the local 'roberta-base' model.
    
    Pros:
    - Improved performance over BERT in some cases.
    - Captures nuanced meanings.
    
    Cons:
    - Computationally intensive.
    
    Use Cases:
    - Applications needing nuanced text understanding.
    
    Args:
    texts (list of str): List of texts to embed.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    model_name = 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1).numpy()
    
    return np.array(embeddings)

def create_local_distilbert_embeddings(texts):
    """
    Creates embeddings using the local 'distilbert-base-uncased' model.
    
    Pros:
    - Lighter and faster than BERT.
    - Good trade-off between performance and speed.
    
    Cons:
    - Slightly less accurate than BERT.
    
    Use Cases:
    - Applications needing fast and efficient embeddings.
    
    Args:
    texts (list of str): List of texts to embed.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1).numpy()
    
    return np.array(embeddings)

def create_local_t5_embeddings(texts):
    """
    Creates embeddings using the local 't5-base' model.
    
    Pros:
    - Versatile model capable of multiple NLP tasks.
    
    Cons:
    - Larger and slower compared to specialized embedding models.
    
    Use Cases:
    - Applications requiring versatile NLP capabilities.
    
    Args:
    texts (list of str): List of texts to embed.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    model_name = 't5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1).numpy()
    
    return np.array(embeddings)

def create_local_gpt2_embeddings(texts):
    """
    Creates embeddings using the local 'gpt2' model.
    
    Pros:
    - Well-known generative model.
    - Can generate coherent text.
    
    Cons:
    - Not specifically designed for embeddings.
    
    Use Cases:
    - Generative applications needing contextual embeddings.
    
    Args:
    texts (list of str): List of texts to embed.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1).numpy()
    
    return np.array(embeddings)
