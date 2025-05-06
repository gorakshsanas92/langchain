from langchain_community.document_loaders import TextLoader

loader = TextLoader('simple.txt')
# Load Document
document = loader.load()

print(document)