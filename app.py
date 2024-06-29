import streamlit as st
from utils import setup_retrieval_chain, setup_conversational_chain, load_pdf, load_text, split_texts, create_embeddings, create_vectorstore, create_retriever, get_answer
import numpy as np
import os

st.title('RAG Chatbot')

uploaded_files = st.file_uploader("Upload README, PDF, or Text files", accept_multiple_files=True)
if uploaded_files:
    texts = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            texts.extend(load_pdf(uploaded_file))
        elif uploaded_file.type == "text/plain":
            texts.extend(load_text(uploaded_file))
        else:
            st.error("Unsupported file type")

    chunks = split_texts(texts)
    embeddings = create_embeddings(chunks)
    vectorstore = create_vectorstore(embeddings, chunks)
    retriever = create_retriever(vectorstore)
    conversational_chain = setup_conversational_chain(retriever)

    question = st.text_input('Ask a question about the uploaded files:')
    if question:
        answer = get_answer(question, conversational_chain)
        st.write(answer)
else:
    st.info('Upload files to begin.')

if st.button('Load from pre-indexed directory'):
    retriever = setup_retrieval_chain('readme_files')
    conversational_chain = setup_conversational_chain(retriever)

    question = st.text_input('Ask a question about the pre-indexed files:')
    if question:
        answer = get_answer(question, conversational_chain)
        st.write(answer)
