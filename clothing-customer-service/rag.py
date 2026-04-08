# rag.py

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, format_document, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda

import config as config
from file_history_store import get_history

from vector_store import VectorStoreService

# 打印完整提示词
def print_prompt(full_prompt):
    print("="*20, full_prompt.to_string(), "\n"+ "="*20)
    return full_prompt


class RagService(object):
    def __init__(self):
        # 向量知识库对象
        self.vector_service = VectorStoreService(
            embedding = DashScopeEmbeddings(model=config.embedding_model)
        )

        # 提示词模板
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，参考资料:{context}"),
                ("system","并根据以下历史对话内容，简洁和专业地回答用户问题，对话历史如下："),
                MessagesPlaceholder("history"),
                ("human", "用户提问: {input}")
            ]
        )

        # 聊天模型
        self.chat_model = ChatTongyi(model=config.chat_model)

        # 初始链
        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()

        def format_document(docs:list[Document]):
            if not docs:
                return "无相关参考资料"
            # context = "[" + " ; ".join(d.page_content for d in docs) + "]"
            format_str = ""
            for doc in docs:
                format_str += f"文档片段：{doc.page_content}\n 文档元数据：{doc.metadata}\n\n"

            return format_str

        def format_for_retriever(value:dict):
            input_str = value["input"]
            return input_str

        def format_for_prompt(value):
            # {input,context ,history}
            new_value = {}
            new_value["input"] = value["input"]["input"]        # 末尾加逗号会变成元组放入字典中
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = ({"input":RunnablePassthrough(),"context": RunnableLambda(format_for_retriever) | retriever | format_document } | RunnableLambda(format_for_prompt) | self.prompt_template | self.chat_model | StrOutputParser())

        # 增强后，有记忆的链
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key= "input",
            history_messages_key= "history",
        )

        return conversation_chain

if __name__ == "__main__":
    session_config={
        "configurable":{
            "session_id":"user_001",
        }
    }

    # 注意：增强链的invoke要求输入为dict
    res = RagService().chain.invoke({"input":"根据我的身高和体重进行尺码推荐"}, session_config)
    print(res)
