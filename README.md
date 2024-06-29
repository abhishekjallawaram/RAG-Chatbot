# RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot powered by Langchain, OpenAI, Google Generative AI, and Hugging Face. It allows users to upload README, PDF, or text files and ask questions about their content. The chatbot retrieves relevant information from the files and generates accurate, context-aware responses.

## File Structure

- **.env**: Contains API keys.
- **app.py**: Streamlit application file.
- **create_index.py**: Script to create FAISS index from a directory of files.
- **requirements.txt**: List of dependencies.
- **readme_files/**: Directory containing sample README files.
- **uploads/**: Directory where uploaded files are saved.
- **utils.py**: Utility functions for document loading, text splitting, embedding, and retrieval.

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
