import streamlit as st
from main import setup_retrieval_chain, setup_conversational_chain, get_answer

st.title("RAG Chatbot")

uploaded_files = st.file_uploader("Upload README, PDF, or Text files", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(f"uploads/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())

    directory = 'uploads'
    retriever = setup_retrieval_chain(directory)
    conversational_chain = setup_conversational_chain(retriever)

    question = st.text_input("Ask a question about the uploaded files:")
    if question:
        answer = get_answer(question, conversational_chain)
        st.write(answer)
