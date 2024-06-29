import os
import numpy as np
import faiss
from langchain.document_loaders import SimpleDirectoryReader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI, GoogleGenerativeAI, HuggingFaceLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import Memory
from dotenv import load_dotenv
import torch
from PyPDF2 import PdfFileReader
import io

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Load tokenizer and model for embeddings
tokenizer = HuggingFaceEmbeddings(api_key=HUGGINGFACE_API_KEY)

def load_documents(directory):
    loader = SimpleDirectoryReader(directory)
    return loader.load()

def load_pdf(file):
    pdf_loader = PyPDFLoader(file)
    return pdf_loader.load()

def load_text(file):
    text_loader = TextLoader(file)
    return text_loader.load()

def split_texts(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return text_splitter.split_texts(texts)

def create_embeddings(texts):
    embeddings = [tokenizer.embed(text) for text in texts]
    return np.array(embeddings)

def create_vectorstore(vectors, texts):
    vectorstore = FAISS(vectors.shape[1])
    for i, vector in enumerate(vectors):
        vectorstore.add(vector, metadata={'text': texts[i]})
    return vectorstore

def create_retriever(vectorstore):
    return vectorstore.as_retriever()

def setup_retrieval_chain(directory):
    documents = load_documents(directory)
    texts = [doc['text'] for doc in documents]
    chunks = split_texts(texts)
    embeddings = create_embeddings(chunks)
    vectorstore = create_vectorstore(embeddings, chunks)
    retriever = create_retriever(vectorstore)
    return retriever

def setup_conversational_chain(retriever):
    memory = Memory()
    llm = LangChain(llm_name="openai", api_key=OPENAI_API_KEY)
    conversational_chain = ConversationalRetrievalChain(llm=llm, retriever=retriever, memory=memory)
    return conversational_chain

def get_answer(question, conversational_chain):
    return conversational_chain.run(question)
