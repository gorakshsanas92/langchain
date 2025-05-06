from langchain_community.document_loaders import WebBaseLoader
from bs4 import SoupStrainer

loader = WebBaseLoader(
    web_path="https://www.britannica.com/biography/A-P-J-Abdul-Kalam",
    bs_kwargs=dict(parse_only=SoupStrainer(class_=("topic-paragraph", "h1"))))
document = loader.load()

print(document)