"""
知识库操作：防止重复存储相同文档 导致冗余
"""
import hashlib
import os.path
from datetime import datetime

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config as cf

def check_md5(md5_str:str):
    # 检查md5字符串是否已经存在（被处理过）

    if not os.path.exists(cf.md5_path):
        # 文件不存在，未处理过
        open(cf.md5_path,'w',encoding="utf-8").close()
        return False
    else:
        for line in open(cf.md5_path, 'r', encoding="utf-8").readlines():
            line = line.strip() # 处理字符串前后的空格和回车
            if md5_str == line:
                return True     # 表示在文件中已记录处理过

    return False

def save_md5(md5_str:str):
    # 将传入的md5字符串记录到文件内保存，知道是否已读取
    with open(cf.md5_path,'a',encoding="utf-8") as f:
        f.write(md5_str + "\n")

    # print(f"{md5_str} saved")

def get_string_md5(input_str:str,encoding="utf-8"):
    """将传入的字符串转为md5"""

    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding = encoding)
    md5_obj = hashlib.md5()     # 得到md5对象
    md5_obj.update(str_bytes)   # 更新内容（传入即将要转换的字节数组）
    return md5_obj.hexdigest()  # 得到16进制字符串

# 核心干活的类
class KnowledgeBaseService(object):
    def __init__(self):
        # 如果数据库文件夹不存在则创建，如果存在则跳过
        os.makedirs(cf.CHROMADB_PATH,exist_ok=True)

        self.chroma = Chroma(
            collection_name=cf.COLLECTION_NAME, # 数据库表，名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=cf.CHROMADB_PATH,   # 数据库本地存储文件夹路径
        )      # 向量存储实例，chroma向量库对象

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=cf.chunk_size,
            chunk_overlap= cf.chunk_overlap,
            separators=cf.separators,       # 自然段落划分符号
            length_function=len,
        )    # 文本分割器对象

    def upload_by_str(self,data:str,filename):
        """将传入的字符串，分割 -> 向量化 -> 存入向量数据库"""
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        # 分割
        # 设置一个文本分割阈值
        if len(data) > cf.max_split_char:
            knowledge_chunks:list[str] = self.splitter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata = {
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"小张"
        }

        # 内容加载到向量库中
        self.chroma.add_texts(
            # Iterable -> 迭代器：list / tuple
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        save_md5(md5_hex)

        return "[成功]内容已经成功载入向量库！"


if __name__ == '__main__':
    # r1 = get_string_md5("周杰伦")
    # r2 = get_string_md5("周杰伦")
    # r3 = get_string_md5("周杰伦2")

    # 每一步要进行阶段性的测试
    DB_service = KnowledgeBaseService()
    r = DB_service.upload_by_str("周杰伦","testfile")
    print(r)
    # print(r1,r2,r3)
    # if check_md5(r):
    #     save_md5(r)
