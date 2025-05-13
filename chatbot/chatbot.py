import os
from dotenv import load_dotenv

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

llm = ChatGroq(model="gemma2-9b-it")

sessions = {}

def getSessionHistory(session_id: str)->BaseChatMessageHistory:
    if session_id not in sessions:
        sessions[session_id] = ChatMessageHistory()
    return sessions[session_id]

withMessageHistory = RunnableWithMessageHistory(llm, getSessionHistory)

print("---> ", withMessageHistory)

config = {"configurable": {"session_id": "chat1"}}

result = withMessageHistory.invoke(
    [HumanMessage(content="My name is Gaurav and I am Software Developer")],
    config=config,
)

print(result.content)

result2 = withMessageHistory.invoke(
    [HumanMessage(content="What is my name?")],
    config=config,
)

print(result2.content)