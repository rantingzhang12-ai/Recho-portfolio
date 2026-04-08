"""
    对向量存储库中的操作
"""

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

import config as config

class VectorStoreService(object):
    def __init__(self,embedding):
        self.embedding = embedding

        # 构建向量库的检索对象
        self.vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,  # 数据库表，名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.CHROMADB_PATH,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs = {"k":config.similarity_threshold})

if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    res = retriever.invoke("我的体重120斤，身高；165，尺码推荐")
    print(res)
