"""配置文档"""

md5_path = "./md5.text"

COLLECTION_NAME = "KeFu_rag"

CHROMADB_PATH = "./chromadb"

chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n","。",".","!","?","！","？"," ",""]

max_split_char = 1000

similarity_threshold = 1        # 检索返回匹配文本的数量

embedding_model = "text-embedding-v4"

chat_model = "qwen3-max"

session_config={
        "configurable":{
            "session_id":"user_001",
        }
    }
