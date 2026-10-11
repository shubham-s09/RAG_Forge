from langchain_huggingface import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Machine learning is a branch of artificial intelligence."

embedding = model.embed_query(text)

print("Embedding dimension:", len(embedding))
print("First 10 numbers:", embedding[:10])