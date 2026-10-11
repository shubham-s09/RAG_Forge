import os

from dotenv import load_dotenv
from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from sentence_transformers import CrossEncoder


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# 2. CREATE GROQ CLIENT
# ==========================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 4. LOAD EXISTING FAISS DATABASE
# ==========================================

vector_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# ==========================================
# 5. LOAD RE-RANKER MODEL
# ==========================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ==========================================
# 6. CONVERSATION HISTORY
# ==========================================

messages = []


# ==========================================
# 7. START CHATBOT
# ==========================================

while True:

    # Ask user a question
    question = input(
        "\nAsk a question (type 'exit' to quit): "
    )

    # Exit chatbot
    if question.lower() == "exit":
        print("Goodbye!")
        break


    # ==========================================
    # 8. REWRITE FOLLOW-UP QUESTION
    # ==========================================

    if messages:

        history = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in messages[-4:]
        )

        rewrite_prompt = f"""
Rewrite the user's latest question into a standalone search query.

Use the conversation history to understand references such as:
"it", "that", "this", "they", "them", etc.

Do NOT answer the question.

Return ONLY the rewritten search query.

Conversation:
{history}

Latest question:
{question}
"""

        rewrite_response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": rewrite_prompt
                }
            ]
        )

        search_query = (
            rewrite_response.choices[0]
            .message.content
            .strip()
        )

        print(
            f"\n[DEBUG] Rewritten search query: "
            f"{search_query}"
        )

    else:

        search_query = question


    # ==========================================
    # 9. SEARCH FAISS
    # ==========================================

    # Retrieve 10 candidate chunks
    candidates = vector_db.similarity_search(
        search_query,
        k=10
    )


    # ==========================================
    # 10. RE-RANK THE CANDIDATES
    # ==========================================

    # Create question-document pairs
    pairs = [
        [search_query, document.page_content]
        for document in candidates
    ]


    # Calculate relevance scores
    scores = reranker.predict(pairs)


    # Combine scores with documents
    ranked_results = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )


    # Keep the best 3 chunks
    results = [
        document
        for score, document in ranked_results[:3]
    ]


    # ==========================================
    # 11. DISPLAY RETRIEVAL SCORES
    # ==========================================

    print("\n========== RETRIEVAL SCORES ==========")

    for score, document in ranked_results[:3]:

        page = document.metadata.get(
            "page_label",
            "Unknown"
        )

        print(
            f"Score: {score:.4f} | Page: {page}"
        )


    # ==========================================
    # 12. CREATE CONTEXT
    # ==========================================

    context = "\n\n".join(
        document.page_content
        for document in results
    )


    # ==========================================
    # 13. CREATE RAG PROMPT
    # ==========================================

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the information
provided in the context below.

IMPORTANT:
- Do not use outside knowledge.
- Do not guess.
- Do not invent information.
- If the answer cannot be found in the context, say:

"I could not find the answer in the provided document."

Context:
{context}

User Question:
{question}

Answer:
"""


    # ==========================================
    # 14. ADD QUESTION TO MEMORY
    # ==========================================

    messages.append({
        "role": "user",
        "content": prompt
    })


    # ==========================================
    # 15. ASK GROQ
    # ==========================================

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages
    )


    # ==========================================
    # 16. GET ANSWER
    # ==========================================

    answer = response.choices[0].message.content


    # ==========================================
    # 17. SAVE ANSWER TO MEMORY
    # ==========================================

    messages.append({
        "role": "assistant",
        "content": answer
    })


    # ==========================================
    # 18. DISPLAY ANSWER
    # ==========================================

    print("\n========== ANSWER ==========")
    print(answer)


    # ==========================================
    # 19. DISPLAY SOURCES
    # ==========================================

    print("\n========== SOURCES ==========")

    for i, result in enumerate(results):

        page = result.metadata.get(
            "page_label",
            "Unknown"
        )

        print(
            f"Source {i + 1}: Page {page}"
        )