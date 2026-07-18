import streamlit as st
import tempfile

from loaders.pdf_loader import load_pdf
from splitters.text_splitter import split_documents
from embeddings.embedding import get_embeddings
from vectorstores.faiss_store import create_vector_db
from retrievers.retriever import get_retriever
from llm.groq_llm import get_llm
from rag.rag_chain import create_rag_chain


# -----------------------------
# Page Title
# -----------------------------
st.title("📚 RAG Chatbot")
st.write("Upload a PDF and ask questions from it.")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📄 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF",
    type="pdf"
)
# Clear Chat Button
if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# -----------------------------
# Load RAG Pipeline
# -----------------------------
@st.cache_resource
def load_rag(pdf_path):

    documents = load_pdf(pdf_path)

    chunks = split_documents(documents)

    embeddings = get_embeddings()

    vector_db = create_vector_db(chunks, embeddings)

    retriever = get_retriever(vector_db)

    llm = get_llm()

    rag_chain = create_rag_chain(llm, retriever)

    return rag_chain


rag_chain = None


# -----------------------------
# Build RAG after PDF Upload
# -----------------------------
if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:

        temp_file.write(uploaded_file.read())

        temp_pdf_path = temp_file.name

    rag_chain = load_rag(temp_pdf_path)

    st.sidebar.success("✅ PDF Uploaded Successfully")


# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# -----------------------------
# Chat Input
# -----------------------------
question = st.chat_input("Ask your question...")


if question:

    if rag_chain is None:

        st.warning("⚠️ Please upload a PDF first.")

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = rag_chain.invoke(question)

            answer = response["result"]

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            st.write(answer)