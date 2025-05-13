import os
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langserve import add_routes

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

llm = ChatGroq(model='gemma2-9b-it')

prompt = ChatPromptTemplate(
    [
        ("system", "Translate the following text from English to {language}:"),
        ("user", "{input}")
    ]
)

parser = StrOutputParser()

# chain = prompt|llm|parser

chain = LLMChain(prompt=prompt, llm=llm, output_parser=parser)

app = FastAPI(
    title="Groq API",
    description="Groq API",
    version="1.0"
)


add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
