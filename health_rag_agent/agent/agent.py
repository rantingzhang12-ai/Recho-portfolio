from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.agents import initialize_agent, AgentType
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from langchain.memory import ConversationBufferWindowMemory

def create_agent(tools):
    """
    使用OpenAI兼容接口调用通义千问
    """

    llm = ChatTongyi(
        model=MODEL_NAME, # qwen3-max
        temperature=0.3,
        streaming=False
    )

    # 基于内存存放的历史对话内容
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k = 3,  # 记忆最新的三轮对话
        return_messages=True
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        verbose=True
    )

    return agent
