import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
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

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
NOTICE_PATH = PROJECT_ROOT / "hugginface" / "tax.pdf"

app = FastAPI(title="Groq API", description="Groq API", version="1.0")


add_routes(
    app,
    chain,
    path="/chain",
)


@app.get(
    "/income-tax-notice",
    summary="Download the income tax notice PDF",
    response_description="PDF file containing the income tax notice",
)
async def download_income_tax_notice() -> FileResponse:
    """
    Download the income tax notice.

    Returns the `tax.pdf` file located in the `hugginface` directory.
    """
    if not NOTICE_PATH.exists():
        raise HTTPException(status_code=404, detail="Income tax notice not found.")

    return FileResponse(
        path=NOTICE_PATH,
        media_type="application/pdf",
        filename="income-tax-notice.pdf",
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
