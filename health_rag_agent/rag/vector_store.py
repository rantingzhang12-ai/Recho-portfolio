# rag/vector_store.py

from langchain_community.vectorstores import Milvus
from rag.embeddings import get_embeddings
from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME

def build_vector_store(docs):

    """
    构建向量数据库
    """
    embeddings = get_embeddings()

    vector_store = Milvus.from_documents(
        docs,
        embedding=embeddings,
        connection_args={
            "host": MILVUS_HOST,  # 待填写
            "port": MILVUS_PORT
        },
        collection_name=COLLECTION_NAME
    )

    return vector_store

def load_vector_store():
    """
    加载已存在的向量数据库（app.py 使用）
    """
    embeddings = get_embeddings()

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={
            "host": MILVUS_HOST,
            "port": MILVUS_PORT
        },
        collection_name=COLLECTION_NAME
    )

    return vector_store
