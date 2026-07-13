from langchain.chains import RetrievalQA

def create_rag_chain(llm, retriever):

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return rag_chain