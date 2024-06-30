import os
from loaders.document_loaders import load_documents, load_pdf, load_text
from loaders.text_splitters import split_texts
from embeddings.embedding_models import create_embeddings
from vectorstores.vectorstores import create_vectorstore
from retrieval.retrievers import create_retriever
from chains.conversational_chain import setup_conversational_chain, get_answer
from utils.config import get_huggingface_api_key

def setup_retrieval_chain(directory):
    documents = load_documents(directory)
    texts = [doc['text'] for doc in documents]
    chunks = split_texts(texts)
    embeddings = create_embeddings(chunks, api_key=get_huggingface_api_key())
    vectorstore, metadata = create_vectorstore(embeddings, chunks)
    retriever = create_retriever(vectorstore, metadata)
    return retriever

if __name__ == "__main__":
    directory = 'readme_files'
    retriever = setup_retrieval_chain(directory)
    conversational_chain = setup_conversational_chain(retriever)
    
    question = "What is the purpose of this project?"
    answer = get_answer(question, conversational_chain)
    print(answer)
