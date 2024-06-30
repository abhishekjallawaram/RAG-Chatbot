import os
from langchain.document_loaders import SimpleDirectoryReader, PyPDFLoader, TextLoader
from langchain.document_loaders.html_loader import HTMLLoader
from langchain.document_loaders.word_loader import WordLoader
from langchain.document_loaders.excel_loader import ExcelLoader

def load_documents(directory):
    """
    Load documents from a directory.

    Args:
        directory (str): Directory path containing documents.

    Returns:
        list: Loaded documents.
    """
    loader = SimpleDirectoryReader(directory)
    return loader.load()

def load_pdf(file):
    """
    Load a PDF document.

    Args:
        file (str): File path to the PDF document.

    Returns:
        list: Loaded PDF document.
    """
    pdf_loader = PyPDFLoader(file)
    return pdf_loader.load()

def load_text(file):
    """
    Load a text document.

    Args:
        file (str): File path to the text document.

    Returns:
        list: Loaded text document.
    """
    text_loader = TextLoader(file)
    return text_loader.load()

def load_markdown(file):
    """
    Load a markdown document.

    Args:
        file (str): File path to the markdown document.

    Returns:
        list: Loaded markdown document.
    """
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    return [{"content": content}]

def load_html(file):
    """
    Load an HTML document.

    Args:
        file (str): File path to the HTML document.

    Returns:
        list: Loaded HTML document.
    """
    html_loader = HTMLLoader(file)
    return html_loader.load()

def load_word(file):
    """
    Load a Word document.

    Args:
        file (str): File path to the Word document.

    Returns:
        list: Loaded Word document.
    """
    word_loader = WordLoader(file)
    return word_loader.load()

def load_excel(file):
    """
    Load an Excel spreadsheet.

    Args:
        file (str): File path to the Excel spreadsheet.

    Returns:
        list: Loaded Excel spreadsheet.
    """
    excel_loader = ExcelLoader(file)
    return excel_loader.load()
