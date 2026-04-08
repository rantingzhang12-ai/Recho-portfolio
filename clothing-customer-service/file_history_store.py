"""
文件长期存储对话历史
"""
import json
import os
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict

def get_history(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id,storage_path):
        self.storage_path = storage_path      # 不同会话id的存储文件的，所在文件夹路径
        self.session_id = session_id          # 会话id
        # 完整的文件路径
        self.file_path = os.path.join(self.storage_path, self.session_id)

        # 确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

    # 获取历史消息
    @property
    def messages (self) -> list[BaseMessage]:
        try:
            with open(os.path.join(self.storage_path, self.session_id),"r",encoding = "utf-8",) as f:
                messages_data = json.load(f) # 读取文件内所有内容
            return messages_from_dict(messages_data)   # 将字典转换回BaseMessage对象
        except FileNotFoundError:
            return []

    # 添加历史会话记录
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages)          # Existing messages
        all_messages.extend(messages)               # Add new messages

        # 将数据写入本地文件
        # 类对象写入文件 -> 一堆二进制
        # 要先将BaseMessage消息转换为字典再（借助json模块以json格式写入文件）
        # 官方的message_to_dict：可以将单个消息对象（Base Message实例） -> 字典
        serialized = [message_to_dict (message) for message in all_messages]

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(serialized,f)

    # 清理文件内容
    def clear(self) -> None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)   # 写入空

if __name__ == "__main__":
    # 清空历史记录
    get_history("user_001").clear()
