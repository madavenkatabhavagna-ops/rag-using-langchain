# RAG Using LangChain

## Project Overview
This project is a Retrieval-Augmented Generation (RAG) application that answers questions from a PDF document.

## Technologies Used
- Python
- LangChain
- HuggingFace Embeddings
- FAISS Vector Database
- Groq LLM

## Features
- Load PDF
- Split into chunks
- Create embeddings
- Store vectors in FAISS
- Ask questions about the PDF
- Generate answers using Groq

## Project Flow

PDF
↓
PyPDFLoader
↓
Text Splitter
↓
Embeddings
↓
FAISS Vector Database
↓
Retriever
↓
Groq LLM
↓
Answer

## Future Improvements
- Streamlit UI
- Docker
- Pinecone
- ChromaDB
