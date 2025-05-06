import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))

message = [
    SystemMessage(content="Translate following into French"),
    HumanMessage(content="Hello")
]

result = llm.invoke(message)
parser = StrOutputParser()

response = parser.invoke(result)

print(response)