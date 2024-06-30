import os
from dotenv import load_dotenv

load_dotenv()

def get_openai_api_key():
    return os.getenv('OPENAI_API_KEY')

def get_google_api_key():
    return os.getenv('GOOGLE_API_KEY')

def get_huggingface_api_key():
    return os.getenv('HUGGINGFACE_API_KEY')
