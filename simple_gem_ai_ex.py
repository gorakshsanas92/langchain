import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain


os.environ["OPENAI_API_KEY"]    = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

# llm
llm = ChatOpenAI(model="gpt-4.1")

# Load Document
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()

# Split the text
text_spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
documents = text_spliter.split_documents(docs)

# Embeding (Vector BD)
embeddings = OpenAIEmbeddings()

vectordb = FAISS.from_documents(documents, embeddings)


# Query
# query = "The quality and development speed of AI applications depends"
# result = vectordb.similarity_search(query)
# print(result)

# Retrival Chain, Document chain
from langchain.chains.combine_documents import create_stuff_documents_chain
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context:
    <context>{context}</context>
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)

# result = document_chain.invoke({
#     "input": "LangSmith is a platform for building",
#     "context": [Document(page_content="LangSmith is a platform for building production-grade LLM applications. It allows you to closely monitor and evaluate your application, so you can ship quickly and with confidence.")]
# })

# print(result)


# Get the information from vectorDB
retriver = vectordb.as_retriever()
retriver_chain = create_retrieval_chain(retriver, document_chain)

result = retriver_chain.invoke({"input": "The quality and development speed of AI applications depends"})
print(result)