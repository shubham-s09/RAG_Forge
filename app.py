import os

import streamlit as st

from dotenv import load_dotenv
from groq import Groq

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from sentence_transformers import CrossEncoder


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RAG Forge",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# LOAD AI MODELS
# =========================================================

@st.cache_resource
def load_models():

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return client, embeddings, reranker


client, embeddings, reranker = load_models()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

def create_vector_database(uploaded_files):

    documents = []

    # -----------------------------------------------------
    # Extract text from every uploaded PDF
    # -----------------------------------------------------

    for uploaded_file in uploaded_files:

        pdf_reader = PdfReader(uploaded_file)

        for page_number, page in enumerate(
            pdf_reader.pages,
            start=1
        ):

            text = page.extract_text()

            if text and text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_number
                        }
                    )
                )

    # -----------------------------------------------------
    # Split documents into chunks
    # -----------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(
        documents
    )

    # -----------------------------------------------------
    # Create FAISS vector database
    # -----------------------------------------------------

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_db, len(chunks)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    st.write(
        "Upload your documents and build "
        "a searchable knowledge base."
    )

    # -----------------------------------------------------
    # PDF UPLOAD
    # -----------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    # -----------------------------------------------------
    # SHOW UPLOADED DOCUMENTS
    # -----------------------------------------------------

    if uploaded_files:

        st.write("### 📄 Selected Documents")

        for file in uploaded_files:

            st.write(
                f"📄 {file.name}"
            )

        st.write(
            f"**{len(uploaded_files)} document(s) selected**"
        )

        # -------------------------------------------------
        # BUILD KNOWLEDGE BASE
        # -------------------------------------------------

        if st.button(
            "🔨 Build Knowledge Base",
            use_container_width=True
        ):

            with st.spinner(
                "Processing documents..."
            ):

                (
                    st.session_state.vector_db,
                    st.session_state.chunk_count
                ) = create_vector_database(
                    uploaded_files
                )

                st.session_state.uploaded_files = [
                    file.name
                    for file in uploaded_files
                ]

                # Clear old conversation
                st.session_state.messages = []

            st.success(
                "Knowledge base created!"
            )

    # -----------------------------------------------------
    # KNOWLEDGE BASE STATUS
    # -----------------------------------------------------

    st.divider()

    st.subheader("📊 Knowledge Base Status")

    if st.session_state.vector_db is not None:

        st.success("🟢 Ready")

        st.metric(
            "📄 Documents",
            len(
                st.session_state.uploaded_files
            )
        )

        st.metric(
            "🧩 Chunks",
            st.session_state.chunk_count
        )

    else:

        st.info(
            "No knowledge base created yet."
        )

    # -----------------------------------------------------
    # SHOW CURRENT DOCUMENTS
    # -----------------------------------------------------

    if st.session_state.uploaded_files:

        st.divider()

        st.subheader(
            "📚 Current Documents"
        )

        for filename in (
            st.session_state.uploaded_files
        ):

            st.write(
                f"📄 {filename}"
            )

    # -----------------------------------------------------
    # RAG PIPELINE
    # -----------------------------------------------------

    st.divider()

    st.subheader("🧠 RAG Pipeline")

    st.write("1️⃣ PDF Extraction")
    st.write("2️⃣ Chunking")
    st.write("3️⃣ Embeddings")
    st.write("4️⃣ FAISS Retrieval")
    st.write("5️⃣ Re-ranking")
    st.write("6️⃣ Groq LLM")
    st.write("7️⃣ Source Citations")

    # -----------------------------------------------------
    # CLEAR CHAT
    # -----------------------------------------------------

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    # -----------------------------------------------------
    # CLEAR KNOWLEDGE BASE
    # -----------------------------------------------------

    if st.button(
        "🔄 Clear Knowledge Base",
        use_container_width=True
    ):

        st.session_state.vector_db = None

        st.session_state.uploaded_files = []

        st.session_state.chunk_count = 0

        st.session_state.messages = []

        st.rerun()


# =========================================================
# MAIN PAGE
# =========================================================

st.title("🤖 RAG Forge")

st.caption(
    "Conversational AI over your own documents"
)


# =========================================================
# KNOWLEDGE BASE MESSAGE
# =========================================================

if st.session_state.vector_db is None:

    st.info(
        "👈 Upload one or more PDF documents "
        "from the sidebar and click "
        "**Build Knowledge Base** to begin."
    )

else:

    st.success(
        f"🟢 Knowledge base ready — "
        f"{len(st.session_state.uploaded_files)} "
        f"document(s) | "
        f"{st.session_state.chunk_count} chunks"
    )


# =========================================================
# DISPLAY PREVIOUS CHAT
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # -----------------------------------------------
        # DISPLAY SOURCES
        # -----------------------------------------------

        if (
            message["role"] == "assistant"
            and "sources" in message
            and message["sources"]
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.write(
                        f"📄 {source['file']} "
                        f"— Page {source['page']}"
                    )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # CHECK KNOWLEDGE BASE
    # -----------------------------------------------------

    if st.session_state.vector_db is None:

        st.warning(
            "Please upload PDF files and "
            "build the knowledge base first."
        )

        st.stop()

    # -----------------------------------------------------
    # DISPLAY USER QUESTION
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # -----------------------------------------------------
    # QUERY REWRITING
    # -----------------------------------------------------

    if len(st.session_state.messages) > 1:

        history_messages = (
            st.session_state.messages[-5:]
        )

        history = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in history_messages
        )

        rewrite_prompt = f"""
Rewrite the user's latest question into a
standalone search query.

Use the conversation history to understand
references such as:

"it"
"that"
"this"
"they"
"them"

Do NOT answer the question.

Return ONLY the rewritten search query.

Conversation:
{history}

Latest question:
{question}
"""

        rewrite_response = (
            client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": rewrite_prompt
                    }
                ]
            )
        )

        search_query = (
            rewrite_response
            .choices[0]
            .message
            .content
            .strip()
        )

    else:

        search_query = question

    # -----------------------------------------------------
    # RETRIEVE CANDIDATES
    # -----------------------------------------------------

    candidates = (
        st.session_state.vector_db
        .similarity_search(
            search_query,
            k=10
        )
    )

    # -----------------------------------------------------
    # RE-RANK RESULTS
    # -----------------------------------------------------

    pairs = [
        [
            search_query,
            document.page_content
        ]

        for document in candidates
    ]

    scores = reranker.predict(
        pairs
    )

    ranked_results = sorted(
        zip(
            scores,
            candidates
        ),

        key=lambda x: x[0],

        reverse=True
    )

    # -----------------------------------------------------
    # TAKE TOP 3 RESULTS
    # -----------------------------------------------------

    results = [
        document

        for score, document
        in ranked_results[:3]
    ]

    # -----------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------

    context = "\n\n".join(
        document.page_content

        for document in results
    )

    # -----------------------------------------------------
    # GENERATE ANSWER
    # -----------------------------------------------------

    prompt = f"""
You are a helpful document assistant.

Answer the user's question using ONLY the
information provided in the context below.

IMPORTANT:

- Do not use outside knowledge.
- Do not guess.
- Do not invent information.
- If the answer cannot be found in the context,
say:

"I could not find the answer in the provided documents."

Context:

{context}

User Question:

{question}

Answer:
"""

    # -----------------------------------------------------
    # ASSISTANT RESPONSE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching your documents..."
        ):

            response = (
                client.chat.completions.create(
                    model="openai/gpt-oss-20b",

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

        st.markdown(answer)

        # -------------------------------------------------
        # COLLECT SOURCES
        # -------------------------------------------------

        source_pages = []

        for result in results:

            source = result.metadata.get(
                "source",
                "Unknown"
            )

            page = result.metadata.get(
                "page",
                "Unknown"
            )

            source_info = {
                "file": source,
                "page": page
            }

            if source_info not in source_pages:

                source_pages.append(
                    source_info
                )

        # -------------------------------------------------
        # DISPLAY SOURCES
        # -------------------------------------------------

        if source_pages:

            with st.expander(
                "📚 Sources"
            ):

                for source in source_pages:

                    st.write(
                        f"📄 {source['file']} "
                        f"— Page {source['page']}"
                    )

    # -----------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_pages
        }
    )