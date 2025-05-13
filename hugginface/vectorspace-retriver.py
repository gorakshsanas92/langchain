import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

llm = ChatGroq(model="gemma2-9b-it")

# Document Loader
loader = PyPDFLoader("tax.pdf")
document = loader.load()

# Emddening
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorStore = Chroma.from_documents(document, embedding=embeddings)

# result = vectorStore.similarity_search("Income tax")

# Retriever
retriever = vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant give information about tax with given context"),
        ("user", "Give the answer using the provided context only. {question} context: {context}")
    ]
)

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke("tax calculation")

print(response.content)

