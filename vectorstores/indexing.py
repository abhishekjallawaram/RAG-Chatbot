# indexing.py
import numpy as np
from langchain_community.vectorstores import FAISS, Pinecone, Weaviate, Milvus, Chroma, LanceDB
from langchain_openai import OpenAIEmbeddings

def index_with_faiss(documents, embeddings):
    """
    Index documents with FAISS.
    
    Args:
        documents (list): List of document objects.
        embeddings (object): Embedding model to use.
    """
    db = FAISS.from_documents(documents, embeddings)
    return db

def index_with_pinecone(documents, embeddings, api_key):
    """
    Index documents with Pinecone.
    
    Args:
        documents (list): List of document objects.
        embeddings (object): Embedding model to use.
        api_key (str): Pinecone API key.
    """
    db = Pinecone.from_documents(documents, embeddings, api_key)
    return db

def index_with_weaviate(documents, embeddings, api_key):
    """
    Index documents with Weaviate.
    
    Args:
        documents (list): List of document objects.
        embeddings (object): Embedding model to use.
        api_key (str): Weaviate API key.
    """
    db = Weaviate.from_documents(documents, embeddings, api_key)
    return db

def index_with_milvus(documents, embeddings, api_key):
    """
    Index documents with Milvus.
    
    Args:
        documents (list): List of document objects.
        embeddings (object): Embedding model to use.
        api_key (str): Milvus API key.
    """
    db = Milvus.from_documents(documents, embeddings, api_key)
    return db

def index_with_chroma(documents, embeddings):
    """
    Index documents with Chroma.
    
    Args:
        documents (list): List of document objects.
        embeddings (object): Embedding model to use.
    """
    db = Chroma.from_documents(documents, embeddings)
    return db

def index_with_lance(documents, embeddings, db_path):
    """
    Index documents with LanceDB.
    
    Args:
        documents (list): List of document objects.
        embeddings (object): Embedding model to use.
        db_path (str): Path to LanceDB database.
    """
    db = lancedb.connect(db_path)
    table = db.create_table("my_table", data=[{"vector": embeddings.embed_query(doc["content"]), "text": doc["content"], "id": doc["id"]} for doc in documents], mode="overwrite")
    return table

def store_embeddings(embeddings, file_path):
    """
    Store embeddings to a file.
    
    Args:
        embeddings (np.ndarray): Embeddings to store.
        file_path (str): Path to the file.
    """
    np.save(file_path, embeddings)

def load_embeddings(file_path):
    """
    Load embeddings from a file.
    
    Args:
        file_path (str): Path to the file.
    
    Returns:
        np.ndarray: Loaded embeddings.
    """
    return np.load(file_path)

# Example usage
if __name__ == "__main__":
    documents = [...]  # Load or define your documents
    embeddings = OpenAIEmbeddings(api_key="your_openai_api_key")

    # Index documents with FAISS
    faiss_db = index_with_faiss(documents, embeddings)
    # Index documents with Pinecone
    pinecone_db = index_with_pinecone(documents, embeddings, api_key="your_pinecone_api_key")

    # Store and load embeddings for reusability
    store_embeddings(embeddings, "embeddings.npy")
    loaded_embeddings = load_embeddings("embeddings.npy")
