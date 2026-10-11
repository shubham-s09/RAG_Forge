from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Load PDF
loader = PyPDFLoader(
    "data/documents/machine_learning.pdf"
)

documents = loader.load()


# 2. Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)


# 3. Create embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Convert chunks into vectors and store them in FAISS
vector_db = FAISS.from_documents(
    chunks,
    embedding
)


# 5. Save FAISS database
vector_db.save_local("faiss_index")


print("Pages:", len(documents))
print("Chunks:", len(chunks))
print("FAISS database created successfully!")