# rag/loader.py
#
import os
import re
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)


def load_documents(data_path: str):
    documents = []

    print("扫描目录:", os.path.abspath(data_path))

    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)

            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.lower().endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file.lower().endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                continue

            docs = loader.load()    # 调用加载器
            # 相对路径
            relative_path = os.path.relpath(file_path, data_path)

            patient_id = relative_path.split(os.sep)[0]
            date_match = re.search(r"\d{4}-\d{2}", file)

            for doc in docs:
                doc.metadata["patient_id"] = patient_id
                doc.metadata["date"] = date_match.group() if date_match else "unknown"
                doc.metadata["source"] = file

            documents.extend(docs)

    return documents


# if __name__ == "__main__":
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     DATA_PATH = os.path.join(BASE_DIR, "data")
#
#     documents = load_documents(DATA_PATH)
#
#     print("共加载文档数:", len(documents))
#     for doc in documents:
#         print(doc.metadata)
