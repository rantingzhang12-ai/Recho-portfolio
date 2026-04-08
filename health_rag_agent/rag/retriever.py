from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def rerank_docs(query, docs):
    scored = [(doc.page_content.count(query), doc) for doc in docs]
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:3]]


def get_retriever(docs, vector_store):
    vector_ret = vector_store.as_retriever(search_kwargs={"k": 5})

    bm25_ret = BM25Retriever.from_documents(docs)
    bm25_ret.k = 5

    return EnsembleRetriever(
        retrievers=[vector_ret, bm25_ret],
        weights=[0.7, 0.3]
    )
