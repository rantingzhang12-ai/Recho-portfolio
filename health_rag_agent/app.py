# app.py

import os

# 🔥 强制关闭所有代理
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

from fastapi import FastAPI
from starlette.responses import StreamingResponse

from rag.loader import load_documents
from rag.splitter import split_documents
from rag.vector_store import build_vector_store,load_vector_store
from agent.tools import create_tools
from agent.agent import create_agent
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:63342",
    "http://127.0.0.1:63342"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 1. 加载Milvus（不重新embedding）
print("🚀 加载向量库...")
vector_store = load_vector_store()

# 🚀 2. BM25仍然需要原始文本
documents = load_documents("data")
split_docs = split_documents(documents)


# 🚀 3. 初始化Agent
print("🧠 初始化Agent...")
tools = create_tools(split_docs,vector_store)
agent = create_agent(tools)


@app.get("/chat")
def generate_report(query: str):
    try:
        result = agent.invoke({"input": query})
        return {
            "success": True,
            "data": result["output"] # 可以改流式输出
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/stream")
def stream(query: str):

    def generator():
        for chunk in agent.stream(query):
            yield chunk

    return StreamingResponse(generator(), media_type="text/plain")
