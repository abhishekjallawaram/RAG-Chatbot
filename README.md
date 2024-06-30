# RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot powered by Langchain, OpenAI, Google Generative AI, and Hugging Face. It allows users to upload README, PDF, or text files and ask questions about their content. The chatbot retrieves relevant information from the files and generates accurate, context-aware responses.

## Table of Contents
- [RAG Chatbot](#rag-chatbot)
  - [Table of Contents](#table-of-contents)
  - [File Structure](#file-structure)
  - [Setup](#setup)
  - [Usage](#usage)
  - [RAG Process](#rag-process)
    - [Document Loading](#document-loading)
    - [Text Splitting](#text-splitting)
    - [Vectorization](#vectorization)
    - [Similarity Comparison](#similarity-comparison)
    - [Querying](#querying)
    - [Response Generation](#response-generation)
    - [Langchain Integration](#langchain-integration)
  - [Scraping Data](#scraping-data)
  - [Conclusion](#conclusion)

## File Structure

For a detailed file structure, please refer to the [file_structure.md](file_structure.md).

## Setup

1. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file and add your API keys:
    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    GOOGLE_API_KEY=your_google_api_key
    HUGGINGFACE_API_KEY=your_huggingface_api_key
    ```

4. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## Usage

- Upload README, PDF, or text files through the Streamlit interface.
- Ask questions about the uploaded files and get context-aware answers.
- Optionally, load pre-indexed files from a directory.

## RAG Process

The RAG process involves several steps to ensure accurate and context-aware responses. Here’s a detailed breakdown of each step:

### Document Loading

The first step is to load documents into the system. This can include README files, PDFs, or text files. Various libraries and approaches can be used:

- **PyMuPDF**: For loading PDFs.
- **Plain Text Loader**: For text files.
- **Markdown Loader**: For README files.
- **Langchain Document Loaders**: Provides support for various document formats and sources.

### Text Splitting

Once the documents are loaded, they are split into smaller chunks to manage memory efficiently and improve retrieval performance. Common approaches include:

- **Sentence Splitter**: Splits documents into sentences.
- **Paragraph Splitter**: Splits documents into paragraphs.
- **Token Splitter**: Splits documents based on a specified token count.
- **Langchain Text Splitters**: Provides flexible text splitting based on different criteria.

### Vectorization

Vectorization converts text into numerical representations (embeddings) that can be processed by machine learning models. Different embedding models and tokenizers can be used:

- **OpenAI Embeddings**: Using OpenAI's models for high-quality embeddings (paid).
- **Hugging Face Models**: Open-source models like BERT, RoBERTa for embeddings.
- **Sentence Transformers**: Models like `all-MiniLM-L6-v2` from Hugging Face.
- **Tokenizers**:
  - **Byte-Pair Encoding (BPE)**: Commonly used with models like GPT-2.
  - **WordPiece**: Used by models like BERT.
  - **SentencePiece**: Used by models like T5.

### Similarity Comparison

Similarity comparison helps in finding the most relevant chunks of text based on the query. Various methods and vector databases can be used:

- **Vector Databases**:
  - **FAISS**: An open-source library for efficient similarity search.
  - **Pinecone**: A managed vector database service (paid).
  - **Weaviate**: An open-source vector search engine.
  
- **Similarity Metrics**:
  - **Cosine Similarity**: Measures the cosine of the angle between two vectors.
  - **Dot Product**: Measures the dot product of two vectors.
  - **Euclidean Distance**: Measures the straight-line distance between two vectors.

### Querying

Querying involves fetching the relevant chunks based on the similarity comparison. This step retrieves the most relevant pieces of information that will be used to generate the response. Different approaches include:

- **Keyword Matching**: Basic approach for finding relevant documents.
- **Semantic Search**: Uses vector embeddings to find semantically similar documents.

### Response Generation

The final step is generating the response using the retrieved information. This involves using generative models:

- **OpenAI GPT-3**: High-quality text generation (paid).
- **Google Generative AI**: Advanced text generation capabilities (paid).
- **Hugging Face Transformers**: Open-source models like GPT-2, T5 for text generation.
- **Langchain Response Generators**: Integrates various models for generating responses.

### Langchain Integration

Langchain provides a flexible framework for integrating different components of the RAG process:

- **Document Loaders**: Supports various document formats and sources.
- **Text Splitters**: Flexible text splitting based on different criteria.
- **Embeddings and Vector Stores**: Integration with different embedding models and vector databases.
- **Retrieval and Querying**: Efficient retrieval and querying mechanisms.
- **Response Generation**: Integration with various generative models for response generation.

## Scraping Data

For scraping horoscope data, a separate script `scraper.py` is included in the `data` directory. Detailed documentation for scraping data can be found [here](data/README.md).

## Conclusion

This project provides a powerful RAG chatbot that leverages advanced AI models and retrieval techniques to deliver accurate and context-aware responses. By supporting various file types and incorporating a scraping script for horoscope data, it offers a comprehensive solution for data retrieval and question answering.
