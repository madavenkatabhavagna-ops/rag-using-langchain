
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

loader = PyPDFLoader("Funny_Stories_for_RAG.pdf")

documents = loader.load()

print("PDF Loaded Successfully")
print("Number of Pages:", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Number of Chunks:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")

vector_db = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS Vector Database Created")

retriever = vector_db.as_retriever()

print("Retriever Ready")

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile"
)

print("Groq LLM Connected")

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

print("RAG Ready")

while True:

    question = input("\nAsk your question (type 'exit' to stop): ")

    if question.lower() == "exit":
        print("Thank you!")
        break

    response = rag_chain.invoke(question)

    print("\nAnswer:")
    print(response["result"])