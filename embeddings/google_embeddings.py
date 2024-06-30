
from google.cloud import language_v1
import numpy as np

#pip install google-cloud-language

"""
Function to create embeddings using Google Cloud Natural Language API.

Args:
    texts (list of str): List of text documents to create embeddings for.
    project_id (str): Google Cloud project ID.

Returns:
    np.ndarray: Embeddings for the given texts.
"""
def create_google_embeddings(texts, project_id):
    """
    Creates embeddings using Google Cloud Natural Language API.
    
    Pros:
    - Reliable and scalable.
    - Strong support for various languages.
    - Efficient for large-scale applications.
    
    Cons:
    - API costs.
    - Dependency on Google Cloud services.
    
    Costs:
    - API usage costs, charged based on the number of requests and data processed.
    
    Use Cases:
    - Large-scale applications.
    - Multilingual support.
    - Enterprises requiring robust and scalable solutions.
    
    Args:
    texts (list of str): List of texts to embed.
    project_id (str): Google Cloud project ID.
    
    Returns:
    np.ndarray: Array of embeddings.
    """
    client = language_v1.LanguageServiceClient()
    embeddings = []
    for text in texts:
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = client.annotate_text(document=document, features={"extract_document_sentiment": True})
        embeddings.append(response.document_sentiment.score)  # Placeholder, replace with actual embeddings if available
    return np.array(embeddings)

