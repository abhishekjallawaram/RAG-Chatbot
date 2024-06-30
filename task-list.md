# RAG Development Checklist

## 1. Environment Setup
- [x] Create a virtual environment
  - [x] `python -m venv venv`
  - [x] `source venv/bin/activate`
- [x] Install required dependencies
  - [x] `pip install -r requirements.txt`
- [x] Set up .env file with API keys
  - [x] Add `OPENAI_API_KEY`
  - [x] Add `GOOGLE_API_KEY`
  - [x] Add `HUGGINGFACE_API_KEY`

## 2. Data Ingestion
### 2.1 Document Loaders
- [x] Create document loaders for:
  - [x] PDFs
    - [x] Implement PDF loader using PyPDF2
  - [x] Text files
    - [x] Implement text file loader
  - [x] Markdown files
    - [x] Implement markdown file loader
  - [x] HTML files
    - [x] Implement HTML file loader
  - [x] Word documents
    - [x] Implement Word document loader
  - [x] Excel spreadsheets
    - [x] Implement Excel spreadsheet loader

#### 2.2 Text Splitting
- [x] Implement text splitting methods
  - [x] Recursive Character Text Splitter
    - [x] Define chunk size and overlap
    - [x] Implement function `split_texts_recursive`
  - [x] Sentence Splitter
    - [x] Implement function `split_texts_sentence`
  - [x] Paragraph Splitter
    - [x] Implement function `split_texts_paragraph`
  - [x] Token Splitter
    - [x] Define chunk size and overlap
    - [x] Implement function `split_texts_token`
- [x] Integrate text splitting methods with document loaders


## 3. Embedding Generation
### 3.1 Hugging Face Models
- [x] Create embeddings using:
  - [x] MiniLM
  - [x] BERT
  - [x] RoBERTa
  - [x] DistilBERT
  - [x] T5
  - [x] GPT-2
- [x] Write functions for each Hugging Face model

### 3.2 OpenAI Models
- [x] Create embeddings using:
  - [x] Ada
- [x] Write functions for OpenAI models

### 3.3 Google Models
- [x] Create embeddings using:
  - [x] Google Cloud Natural Language API
- [x] Write functions for Google models

### 3.4 Ollama Models
- [x] Create embeddings using:
  - [x] Ollama Base
  - [x] Ollama Finance
  - [x] Ollama Healthcare
- [x] Write functions for each Ollama model

### 3.5 Local Models
- [x] Create embeddings using:
  - [x] Local MiniLM
  - [x] Local BERT
  - [x] Local RoBERTa
  - [x] Local DistilBERT
  - [x] Local T5
  - [x] Local GPT-2
- [x] Write functions for each local model

## 4. Tokenization
- [x] Implement tokenizers for all models used in embedding generation
  - [x] Hugging Face tokenizers
  - [x] OpenAI tokenizers
  - [x] Google tokenizers
  - [x] Ollama tokenizers
  - [x] Local model tokenizers

## 5. Vector Store Setup
### 5.1 Vector Store Integration
- [x] Integrate FAISS
  - [x] Install FAISS library
  - [x] Set up FAISS vector store
  - [x] Index embeddings in FAISS
  - [ ] Test retrieval with FAISS

- [ ] Integrate Pinecone
  - [ ] Sign up for Pinecone and obtain API key
  - [ ] Install Pinecone library
  - [ ] Set up Pinecone vector store
  - [ ] Index embeddings in Pinecone
  - [ ] Test retrieval with Pinecone

- [ ] Integrate Weaviate
  - [ ] Install Weaviate library
  - [ ] Set up Weaviate vector store
  - [ ] Index embeddings in Weaviate
  - [ ] Test retrieval with Weaviate

- [ ] Integrate Milvus
  - [ ] Install Milvus library
  - [ ] Set up Milvus vector store
  - [ ] Index embeddings in Milvus
  - [ ] Test retrieval with Milvus

- [ ] Integrate Chroma
  - [ ] Install Chroma library
  - [ ] Set up Chroma vector store
  - [ ] Index embeddings in Chroma
  - [ ] Test retrieval with Chroma

- [ ] Integrate Lance
  - [ ] Install LanceDB library
  - [ ] Set up Lance vector store
  - [ ] Index embeddings in Lance
  - [ ] Test retrieval with Lance

- [ ] Integrate other vector stores as needed
  - [ ] Research additional vector store options
  - [ ] Evaluate pros and cons of each option
  - [ ] Implement integration for selected vector stores
  - [ ] Test retrieval with new vector stores


### 5.2 Indexing
- [ ] Index documents with generated embeddings
  - [ ] Create a separate module for indexing, e.g., indexing.py
  - [ ] Implement functions to index embeddings for each vector store:
    - [ ] FAISS
    - [ ] Pinecone
    - [ ] Weaviate
    - [ ] Milvus
    - [ ] Chroma
    - [ ] Lance
  - [ ] Ensure proper handling of document metadata during indexing
- [ ] Store embeddings for reusability
  - [ ] Save embeddings to disk or a database
  - [ ] Implement functions to load stored embeddings
  - [ ] Integrate the loading and saving mechanisms with the indexing functions


## 6. Retrieval Mechanism
- [ ] Implement retrieval methods for:
  - [ ] Dense Retrieval
    - [ ] Implement dense retrieval using FAISS
    - [ ] Implement dense retrieval using Pinecone
    - [ ] Implement dense retrieval using Weaviate
    - [ ] Implement dense retrieval using Milvus
    - [ ] Implement dense retrieval using Chroma
    - [ ] Implement dense retrieval using LanceDB
  - [ ] Sparse Retrieval
    - [ ] Implement sparse retrieval using BM25
    - [ ] Implement sparse retrieval using Elasticsearch
    - [ ] Implement sparse retrieval using Solr
  - [ ] Multi-Vector Retrieval
    - [ ] Implement multi-vector retrieval using ColBERT
    - [ ] Implement multi-vector retrieval using DPR
    - [ ] Implement multi-vector retrieval using TCT-ColBERT
  - [ ] Advanced Retrieval Types
    - [ ] Vectorstore
      - [ ] Create embeddings for each piece of text
    - [ ] ParentDocument
      - [ ] Index multiple chunks for each document
      - [ ] Retrieve whole parent document
    - [ ] Multi Vector
      - [ ] Create multiple vectors for each document
      - [ ] Index relevant information
    - [ ] Self Query
      - [ ] Use an LLM to transform user input into a search query
      - [ ] Use an LLM to transform user input into a metadata filter
    - [ ] Contextual Compression
      - [ ] Post-processing step to extract relevant information from retrieved documents
    - [ ] Time-Weighted Vectorstore
      - [ ] Fetch documents based on semantic similarity
      - [ ] Fetch documents based on recency
    - [ ] Multi-Query Retriever
      - [ ] Use an LLM to generate multiple queries from the original one
    - [ ] Ensemble
      - [ ] Combine multiple retrieval methods
    - [ ] Long-Context Reorder
      - [ ] Reorder retrieved documents for long-context models


## 7. Query Handling
### 7.1 Question Answering
- [ ] Implement QA system to handle user queries

### 7.2 Conversational Search
- [ ] Implement conversational search capabilities

## 8. Response Generation
- [ ] Generate responses using:
  - [ ] OpenAI GPT-3
  - [ ] Google Generative AI
  - [ ] Hugging Face Transformers

## 9. Integration with Langchain
- [ ] Integrate document loaders, text splitters, and embeddings with Langchain
- [ ] Set up conversational retrieval chain using Langchain

## 10. User Interface
### 10.1 Streamlit Application
- [ ] Set up file upload functionality
- [ ] Implement query input and response display

## 11. Testing and Evaluation
- [ ] Test individual components
- [ ] Evaluate end-to-end system
- [ ] Optimize performance and accuracy

## 12. Documentation
- [ ] Write detailed documentation for each component
- [ ] Create a comprehensive README

## 13. Deployment
- [ ] Set up deployment pipeline
- [ ] Deploy the application to a cloud service

## 14. Maintenance
- [ ] Monitor system performance
- [ ] Update models and dependencies as needed
