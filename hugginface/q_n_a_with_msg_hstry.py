import os
import bs4
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser

# 

load_dotenv()


os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

llm = ChatGroq(model='gemma2-9b-it')

# Load Document
loader = WebBaseLoader(
    web_path="https://www.ibm.com/think/topics/langchain",
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('cms-richtext '))
    )
)

docs = loader.load()

# Make document in chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs_splitter = splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector store
vectorstore = Chroma.from_documents(docs_splitter, embeddings)

# Retriever
retriever = vectorstore.as_retriever()

system_prompt = (
    "You are assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer the question."
    "if you don't know the answer say that you don't know."
    "\n\n"
    "{context}"
)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# Setup Prompt with chat history

contextualize_q_system_prompt = (
    "Given chat history and latest user question"
    "which might reference context in the chat history"
    "formulated a standalone question which can be understood"
    "without chat history. Do not answer the question,"
    "just formulate it if needed and otherwise return it as it is"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

parser = StrOutputParser()

# question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt, output_parser=parser)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# response = rag_chain.invoke({"input": "Prompt templates"})

history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt, output_parser=parser)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []

question = "what is LangChain agents"
response = rag_chain.invoke({"input": question, "chat_history": chat_history})

chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=response['answer'])
    ]
)

question2 = "give detail in simple word"
response2 = rag_chain.invoke({"input": question2, "chat_history": chat_history})


chat_history.extend(
    [
        HumanMessage(content=question2),
        AIMessage(content=response2['answer'])
    ]
)

print(chat_history)




