from loaders.pdf_loader import load_pdf
from splitters.text_splitter import split_documents
from embeddings.embedding import get_embeddings
from vectorstores.faiss_store import create_vector_db
from retrievers.retriever import get_retriever
from llm.groq_llm import get_llm
from rag.rag_chain import create_rag_chain

# Step 1: Load PDF
documents = load_pdf("data/Funny_Stories_for_RAG.pdf")

# Step 2: Split PDF
chunks = split_documents(documents)

# Step 3: Load Embedding Model
embeddings = get_embeddings()

# Step 4: Create Vector Database
vector_db = create_vector_db(chunks, embeddings)

# Step 5: Create Retriever
retriever = get_retriever(vector_db)

# Step 6: Load LLM
llm = get_llm()

# Step 7: Create RAG Chain
rag_chain = create_rag_chain(llm, retriever)

print("RAG Chatbot Ready!")

while True:

    question = input("\nAsk your question (type 'exit' to stop): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    response = rag_chain.invoke(question)

    print("\nAnswer:")
    print(response["result"])