import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st


os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")


# Define model
llm = OllamaLLM(model="llama3.2")

# Prompt Setup
prompt = ChatPromptTemplate([
    ("system", "You are helfull assistant. Give answer the question"),
    ("user", "{question}")
])

# Streamlit setup
st.title("Langchain Demo With Ollama")
text_input = st.text_input("Ask your question")

chain = prompt | llm | StrOutputParser()

if text_input:
    st.write(chain.invoke({"question": text_input}))

