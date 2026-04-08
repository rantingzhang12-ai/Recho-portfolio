import os
# 阿里云百炼 OpenAI兼容接口
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ⚠️ 关键：改成百炼的 base_url
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 使用的模型（通义千问）
MODEL_NAME = "qwen3-max"   # 或 qwen-turbo / qwen-max

# Milvus配置
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

COLLECTION_NAME = "health_rag"
