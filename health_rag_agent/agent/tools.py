# agent/tools.py
# 工具定义（给Agent调用）

from langchain_community.tools import Tool
from rag.retriever import get_retriever,rerank_docs
from utils.pdf_generator import generate_pdf

def create_tools(docs,vector_store):

    retriever = get_retriever(docs,vector_store)

    def rag_search(query: str):
        """
        RAG检索工具
        """
        docs = retriever.get_relevant_documents(query)

        results = rerank_docs(query, docs)

        return "\n".join([d.page_content for d in results])

    def save_pdf_tool(content: str):
        """
        PDF生成工具
        """
        return generate_pdf(content)

    tools = [
        Tool(
            name="Health_RAG_Search",
            func=rag_search,
            description="用于检索健康档案信息"
        ),
        Tool(
            name="Save_PDF",
            func=save_pdf_tool,
            description="用于将健康报告保存为PDF"
        )
    ]

    return tools
