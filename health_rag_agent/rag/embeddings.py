# rag/embeddings.py

# from langchain_openai import OpenAIEmbeddings
# from config import OPENAI_API_KEY, OPENAI_BASE_URL
#
# def get_embeddings():
#     return OpenAIEmbeddings(
#         api_key=OPENAI_API_KEY,
#         base_url=OPENAI_BASE_URL,
#         model="text-embedding-v1",
#         tiktoken_enabled=False
#     )

from langchain_community.embeddings import DashScopeEmbeddings
import os

def get_embeddings():
    return DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

