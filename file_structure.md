RAG-Chatbot/
├── .env                  # Contains API keys
├── .gitignore            # Specifies files and directories to be ignored by git
├── app.py                # Streamlit application file
├── requirements.txt      # List of dependencies
├── readme_files/         # Directory containing sample README files
├── uploads/              # Directory where uploaded files are saved
├── data/                 # Directory for scraped horoscope data and the scraping script
│   ├── scraper.py        # Script to scrape horoscope data
│   └── README.md         # Documentation for scraping data
├── loaders/              # Module for loading documents
│   ├── __init__.py
│   ├── document_loaders.py # Functions to load different types of documents
│   └── text_splitters.py   # Functions to split text into smaller chunks
├── tokenizers/           # Module for tokenizers
│   ├── __init__.py
│   ├── hf_tokenizers.py    # Tokenizers using Hugging Face models
│   ├── openai_tokenizers.py # Tokenizers using OpenAI models
│   ├── google_tokenizers.py # Tokenizers using Google models
│   ├── ollama_tokenizers.py # Tokenizers using Ollama models
│   ├── local_tokenizers.py  # Tokenizers for local models
├── embeddings/           # Module for creating embeddings
│   ├── __init__.py
│   ├── hf_embeddings.py # Tokenizers using Hugging Face models
│   ├── openai_embeddings.py # Tokenizers using OpenAI models
│   ├── google_embeddings.py # Tokenizers using Google models
│   ├── ollama_embeddings.py # Tokenizers using Ollama models
│   ├── local_embeddings.py # Tokenizers for local models
├── vectorstores/         # Module for managing vector stores
│   ├── __init__.py
│   └── vectorstores.py # Functions to create and manage vectorDBs
│   └── indexing.py       # Functions for indexing documents and storing embeddings stores
├── retrieval/            # Module for setting up retrievers
│   ├── __init__.py
│   └── retrievers.py      # Functions to create retrievers
├── chains/               # Module for setting up conversational chains
│   ├── __init__.py
│   └── conversational_chain.py # Functions to set up conversational retrieval chains
├── utils/                # Utility functions
│   ├── __init__.py
│   └── config.py          # Configuration functions to load API keys
└── main.py               # Main script to set up the retrieval chain and handle queries
```