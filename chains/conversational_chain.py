from langchain.llms import OpenAI, GoogleGenerativeAI, HuggingFaceLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import Memory
from utils.config import get_openai_api_key

def setup_conversational_chain(retriever):
    memory = Memory()
    llm = OpenAI(api_key=get_openai_api_key())
    conversational_chain = ConversationalRetrievalChain(llm=llm, retriever=retriever, memory=memory)
    return conversational_chain

def get_answer(question, conversational_chain):
    return conversational_chain.run(question)
