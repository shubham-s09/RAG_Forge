from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyPDFLoader(
    "data/documents/machine_learning.pdf"
)

documents = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)


print("Pages:", len(documents))
print("Chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0].page_content)