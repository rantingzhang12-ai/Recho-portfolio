# ingest.py

from rag.loader import load_documents
from rag.splitter import split_documents
from rag.vector_store import build_vector_store

def main():
    print("🚀 开始构建向量库...")

    print("🚀 开始加载文档...")
    documents = load_documents("data")

    print(f"📄 文档数量: {len(documents)}")

    print("✂️ 开始切分文档...")
    split_docs = split_documents(documents)

    print(f"🔹 切分后文本块数量: {len(split_docs)}")

    print("🧠 开始构建向量库（Milvus）...")
    build_vector_store(split_docs)
    # build_vector_store()
    # 内部会：
    # 创建 collection
    # 清空旧数据（或覆盖）
    # 插入所有向量

    print("✅ 向量库构建完成！")

if __name__ == "__main__":
    main()
