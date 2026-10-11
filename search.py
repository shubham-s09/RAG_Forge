from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load our saved FAISS database
vector_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# Ask a question
query = "What is overfitting?"


# Search for the 3 most relevant chunks
results = vector_db.similarity_search(
    query,
    k=3
)


# Display results
print("Question:", query)

for i, result in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(result.page_content)
    print("Metadata:", result.metadata)