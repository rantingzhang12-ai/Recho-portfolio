import time
from rag import RagService
import streamlit as st
import config as cfg

# 标题
st.title("智能女装客服")
st.divider()    # 分隔符


if "message" not in st.session_state:
    st.session_state["message"] = [{"role":"assistant", "content":"你好，有什么可以帮助你？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

# 用于每次刷新后全部显示聊天历史记录
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入栏
prompt = st.chat_input()

if prompt:

    # 在页面输出用户提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    ai_res_list = []
    with st.spinner("🚀AI思考中..."):
        # 通过迭代器实现流式输出   res_stream就是一个流式迭代器
        res_stream = st.session_state["rag"].chain.stream({"input":prompt},cfg.session_config)

        # 抓包函数：中间缓存一个列表存放完整回复
        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk     # 返回源文本格式

        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})
