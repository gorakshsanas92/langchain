import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

model = ChatGroq(model="gemma2-9b-it")

# Set Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistance"),
        ("user", "My name is gaurav"),
        ("user", "tell me about {topic}")
    ]
)

parser = StrOutputParser()

chain = prompt | model | parser

st.title("Chat Bot")
input_text = st.text_input("Please enter your question")

if input_text:
    response = chain.invoke({"topic": input_text})
    st.write(response)