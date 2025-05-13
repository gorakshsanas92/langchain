import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# Load the Groq API key from the environment variable
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="gemma2-9b-it")

prompt = ChatPromptTemplate(
    [
        ("system", "Translate following into {language}"),
        ("user", "{input}")
    ]
)

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"input": "Hello, how are you?", "language": "French"})

print(result)

# message = [
#     SystemMessage(content="Translate following into French"),
#     HumanMessage(content="Hello")
# ]

# result = llm.invoke(message)
# parser = StrOutputParser()

# response = parser.invoke(result)

# print(response)