import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["OPENAI_API_KEY"]    = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

llm = ChatOpenAI(model="gpt-4.1")

# result = llm.invoke("Hello Hi")
# print(result)

prompt = ChatPromptTemplate(
    [
        ("system", "You are an expert AI Engineer. Provide me answer based on questions"),
        ("user", "{input}")
    ]
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"input": "What is langchain?"})

print(result)

