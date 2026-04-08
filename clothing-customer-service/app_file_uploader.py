
"""基于 streamlit 完成 WEB网页上传服务"""
import time
import streamlit as st
# streamlit 当web页面元素发生变化，则代码重新重头执行一遍
from knowledge_base import KnowledgeBaseService

st.title("知识库更新服务")

# file_uploader 文件上传框
uploader_file = st.file_uploader(
    "请上传txt文件",
    type=["txt"],
    accept_multiple_files=False,        # 不接受一次性上传多文件
)

# session_state 就是一个字典,可以设置状态 key 只要run程序不停，就不会每次都跟着页面一起刷新
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    # 提取文件信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size  / 1024

    st.subheader(f"文件名：{file_name}")                    # 文字显示
    st.write(f"格式：{file_type} | 大小：{file_size:.2f}KB ")        #小字显示

    # 获取文件内容,bytes - > utf-8(中文str)
    text = uploader_file.getvalue().decode("utf-8")
    # st.write(text[:100])
    st.text_area("文件内容预览", text[:100], height=150)        # （前100字）

    with st.spinner("载入知识库中..."):        # 转圈动画
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)
