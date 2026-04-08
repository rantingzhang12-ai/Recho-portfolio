from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    """
    文档切分（RAG关键步骤）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=30,       # 分段之间允许重叠字符数
        separators=["\n\n", "\n", "。", "！", " "]
    )
    return splitter.split_documents(documents)
