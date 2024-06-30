from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTextSplitter, ParagraphTextSplitter, TokenTextSplitter

def split_texts_recursive(texts, chunk_size=512, chunk_overlap=50):
    """
    Split texts into chunks using RecursiveCharacterTextSplitter.

    Args:
        texts (list of str): List of text documents to split.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Number of characters to overlap between chunks.

    Returns:
        list of str: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_texts(texts)

def split_texts_sentence(texts):
    """
    Split texts into chunks using SentenceTextSplitter.

    Args:
        texts (list of str): List of text documents to split.

    Returns:
        list of str: List of text chunks.
    """
    text_splitter = SentenceTextSplitter()
    return text_splitter.split_texts(texts)

def split_texts_paragraph(texts):
    """
    Split texts into chunks using ParagraphTextSplitter.

    Args:
        texts (list of str): List of text documents to split.

    Returns:
        list of str: List of text chunks.
    """
    text_splitter = ParagraphTextSplitter()
    return text_splitter.split_texts(texts)

def split_texts_token(texts, chunk_size=512, chunk_overlap=50):
    """
    Split texts into chunks using TokenTextSplitter.

    Args:
        texts (list of str): List of text documents to split.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Number of tokens to overlap between chunks.

    Returns:
        list of str: List of text chunks.
    """
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_texts(texts)

def handle_text_splitting(texts):
    """
    Handle text splitting for a given list of texts.

    Args:
        texts (list of str): List of text documents to split.

    Returns:
        dict: Dictionary containing different types of text chunks.
    """
    try:
        chunks_recursive = split_texts_recursive(texts)
        chunks_sentence = split_texts_sentence(texts)
        chunks_paragraph = split_texts_paragraph(texts)
        chunks_token = split_texts_token(texts)
        
        return {
            "recursive": chunks_recursive,
            "sentence": chunks_sentence,
            "paragraph": chunks_paragraph,
            "token": chunks_token
        }
    except Exception as e:
        print(f"Error occurred during text splitting: {e}")
        return {}

def main():
    # Example usage of the text splitting functions
    texts = ["This is a sample document. It contains multiple sentences.", "Here is another document for testing."]
    
    split_results = handle_text_splitting(texts)
    
    # Output results for each splitting method
    for method, chunks in split_results.items():
        print(f"\nChunks using {method} splitter:")
        for chunk in chunks:
            print(chunk)

if __name__ == "__main__":
    main()
