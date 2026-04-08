# 服装客服知识库问答项目

这是一个基于 LangChain + Streamlit + Chroma 的轻量级 RAG 示例项目，面向服装/女装客服场景。项目提供了两个主要能力：

- 将文本知识上传到本地向量库
- 基于知识库进行多轮问答，并保留本地聊天记录

项目适合用于学习以下能力：

- RAG 基础流程：文本切分、向量化、检索、问答
- 使用 Streamlit 快速搭建交互式页面
- 使用 Chroma 做本地向量数据库持久化
- 使用文件保存对话历史

## 项目功能

- `app_file_uploader.py`
  - 提供知识库上传页面
  - 支持上传单个 `.txt` 文件
  - 自动读取内容、切分文本并写入 Chroma
  - 使用 MD5 去重，避免重复导入相同内容

- `app_qa.py`
  - 提供客服问答页面
  - 从本地向量库中检索相关知识片段
  - 将检索结果与历史对话一起交给大模型生成回复
  - 通过 Streamlit 聊天组件实现流式展示

- `file_history_store.py`
  - 以文件形式保存每个会话的历史消息
  - 当前默认会话 ID 为 `user_001`

## 技术栈

- Python
- Streamlit
- LangChain
- Chroma
- DashScope Embeddings
- Tongyi / Qwen 模型

## 目录结构

```text
clothing-customer-service/
├─ app_file_uploader.py      # 知识库上传页面
├─ app_qa.py                 # 问答页面
├─ knowledge_base.py         # 文本切分、去重、入库逻辑
├─ rag.py                    # RAG 链路封装
├─ vector_store.py           # Chroma 检索封装
├─ file_history_store.py     # 本地聊天记录存储
├─ config.py                 # 项目配置
├─ requirements.txt          # 依赖列表
├─ chromadb/                 # Chroma 持久化目录
├─ chat_history/             # 本地聊天记录目录
└─ md5.text                  # 已导入文本内容的 MD5 记录
```

## 运行原理

### 1. 知识入库

上传 `.txt` 文件后，系统会：

1. 读取文本内容
2. 计算文本 MD5，检查是否重复导入
3. 按配置对文本进行切分
4. 使用嵌入模型将文本转换为向量
5. 存入本地 Chroma 向量库

### 2. 问答流程

用户提问后，系统会：

1. 用提问内容去 Chroma 检索相关片段
2. 读取本地保存的历史对话
3. 将“用户问题 + 检索上下文 + 历史消息”一起拼接成提示词
4. 调用大模型生成回复
5. 将本轮问答写入本地历史记录

## 环境要求

- Python 3.10 及以上
- 可访问通义千问 / DashScope 服务
- 已配置对应 API Key

## 安装依赖

在项目目录下执行：

```bash
pip install -r requirements.txt
```

根据代码实际引用，若环境中尚未安装，通常还需要补充以下依赖：

```bash
pip install langchain-chroma
```

## 配置说明

项目中的主要配置位于 `config.py`：

- `COLLECTION_NAME`：Chroma 集合名称
- `CHROMADB_PATH`：向量库本地存储目录
- `chunk_size`：分块大小
- `chunk_overlap`：分块重叠长度
- `embedding_model`：嵌入模型名称
- `chat_model`：对话模型名称
- `session_config`：默认会话配置

此外，代码使用了：

- `DashScopeEmbeddings`
- `ChatTongyi`

因此运行前需要准备对应的阿里云百炼 / DashScope 鉴权信息。实际配置方式可根据你的本地 SDK 环境变量习惯进行设置。

## 启动方式

### 1. 启动知识库上传页面

```bash
streamlit run app_file_uploader.py
```

### 2. 启动客服问答页面

```bash
streamlit run app_qa.py
```

## 使用说明

### 上传知识

1. 启动 `app_file_uploader.py`
2. 上传一个 `.txt` 文件
3. 等待系统完成切分、向量化和入库

### 开始问答

1. 启动 `app_qa.py`
2. 在聊天框输入服装客服相关问题
3. 系统会基于已导入知识进行回答

## 当前实现特点

- 适合演示或课程练习使用
- 本地持久化，部署简单
- 支持最基础的多轮记忆
- 支持知识重复导入检测

## 已知限制

- 当前知识上传仅支持 `.txt` 文件
- 默认会话 ID 固定为 `user_001`
- `requirements.txt` 中未完整列出全部实际依赖
- 项目目录名为 `clothing-customer-service`，如果后续希望与 Python 包命名统一，可再考虑改名

## 后续可优化方向

- 支持 PDF、Word、Markdown 等更多文档格式
- 增加多用户 / 多会话管理
- 增加知识删除、重建索引等管理能力
- 增加更细粒度的检索参数配置
- 引入更完善的提示词模板与引用来源展示

## 适用场景

- 服装客服知识库问答 Demo
- RAG 教学示例
- LangChain + Streamlit 入门练习
- 本地私有知识库原型验证
