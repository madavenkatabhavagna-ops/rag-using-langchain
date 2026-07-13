from langchain_community.vectorstores import FAISS

def create_vector_db(chunks, embeddings):

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_db