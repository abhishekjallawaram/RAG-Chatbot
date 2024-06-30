import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

def create_embeddings(texts, api_key):
    tokenizer = HuggingFaceEmbeddings(api_key=api_key)
    embeddings = [tokenizer.embed(text) for text in texts]
    return np.array(embeddings)
