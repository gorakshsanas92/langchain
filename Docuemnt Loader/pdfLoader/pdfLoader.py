from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('sample.pdf')
document = loader.load()

print(document)