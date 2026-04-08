# 健康档案 RAG Agent 项目

这是一个结合 RAG 检索、Agent 工具调用、Milvus 向量库和 FastAPI 接口的健康档案问答项目。它面向个人健康档案或体检资料场景，支持将健康相关文档导入知识库，并通过 Agent 完成问答与报告生成。

项目的核心目标是：

- 对健康档案进行检索增强问答
- 结合向量检索与关键词检索提升召回效果
- 使用 Agent 调用检索工具与 PDF 生成工具
- 通过 FastAPI 对外提供接口

## 项目功能

- 文档导入
  - 扫描 `data/` 目录中的健康档案文件
  - 支持 `PDF`、`DOCX`、`TXT`
  - 自动提取患者 ID、文档日期等元数据

- 向量库构建
  - 对文档进行切分
  - 使用 DashScope Embeddings 生成向量
  - 写入 Milvus 向量数据库

- 混合检索
  - 向量检索：语义召回
  - BM25 检索：关键词召回
  - EnsembleRetriever：组合两种召回结果

- Agent 能力
  - 调用健康档案检索工具回答问题
  - 调用 PDF 工具保存生成内容
  - 保留有限轮数的对话记忆

- API 服务
  - `/chat`：返回普通 JSON 响应
  - `/stream`：返回流式文本响应

## 技术栈

- Python
- FastAPI
- LangChain
- Milvus
- DashScope Embeddings
- Tongyi / Qwen 模型
- ReportLab

## 目录结构

```text
health_rag_agent/
├─ app.py                    # FastAPI 服务入口
├─ ingest.py                 # 文档入库脚本
├─ config.py                 # 全局配置
├─ index.html                # 前端页面示例
├─ requirements.txt          # 依赖列表
├─ agent/
│  ├─ agent.py               # Agent 初始化
│  └─ tools.py               # 工具定义
├─ rag/
│  ├─ loader.py              # 文档加载
│  ├─ splitter.py            # 文本切分
│  ├─ embeddings.py          # 嵌入模型封装
│  ├─ vector_store.py        # Milvus 向量库读写
│  ├─ retriever.py           # 混合检索与简单重排
│  └─ milvus_connect.py      # Milvus 相关辅助文件
├─ utils/
│  └─ pdf_generator.py       # PDF 生成工具
└─ data/
   └─ patient_001/           # 示例健康档案数据
```

## 工作流程

### 1. 文档入库

执行 `ingest.py` 后，系统会：

1. 扫描 `data/` 目录下的文档
2. 按文件类型选择加载器
3. 提取文档内容与元数据
4. 切分为多个文本块
5. 生成向量并写入 Milvus

### 2. 在线问答

启动 `app.py` 后，系统会：

1. 连接已有的 Milvus 向量库
2. 再次读取原始文档，供 BM25 检索使用
3. 初始化检索工具和 PDF 工具
4. 创建带短期记忆的 Agent
5. 对外提供问答接口

### 3. 检索策略

项目使用的是“混合检索”：

- 向量检索负责找语义相近内容
- BM25 负责找关键词高度匹配内容
- 最后通过简单重排逻辑选出更相关的结果

这种方式比纯向量检索更适合健康档案、病例记录、指标问答等既依赖语义又依赖关键字段的场景。

## 环境要求

- Python 3.10 及以上
- 本地可用的 Milvus 服务
- 可访问 DashScope / 通义千问服务
- 已配置 API Key

## 安装依赖

在项目目录下执行：

```bash
pip install -r requirements.txt
```

## 配置说明

配置文件位于 `config.py`，主要参数如下：

- `OPENAI_API_KEY`：从环境变量读取的 API Key
- `OPENAI_BASE_URL`：兼容 OpenAI 的 DashScope 接口地址
- `MODEL_NAME`：使用的对话模型，默认是 `qwen3-max`
- `MILVUS_HOST`：Milvus 主机地址
- `MILVUS_PORT`：Milvus 端口
- `COLLECTION_NAME`：Milvus 集合名

另外，`rag/embeddings.py` 中的嵌入模型通过环境变量读取：

- `DASHSCOPE_API_KEY`

因此在运行前，建议至少配置：

```bash
set OPENAI_API_KEY=你的Key
set DASHSCOPE_API_KEY=你的Key
```

如果你在 PowerShell 中运行，可以使用：

```powershell
$env:OPENAI_API_KEY="你的Key"
$env:DASHSCOPE_API_KEY="你的Key"
```

## 启动步骤

### 1. 启动 Milvus

请先确保本地 Milvus 已正常运行，并与 `config.py` 中的地址一致。

### 2. 构建向量库

```bash
python ingest.py
```

### 3. 启动 API 服务

```bash
uvicorn app:app --reload
```

默认启动后可访问本地接口，例如：

- `GET /chat?query=最近的健康情况怎么样`
- `GET /stream?query=帮我总结这份健康档案`

## 接口说明

### `/chat`

请求示例：

```text
GET /chat?query=请总结患者最近的健康状态
```

返回示例：

```json
{
  "success": true,
  "data": "......"
}
```

### `/stream`

请求示例：

```text
GET /stream?query=生成一份健康报告
```

该接口返回流式文本，适合前端逐步展示回答内容。

## 示例数据说明

项目中自带了 `data/patient_001/2024-01-健康档案.pdf` 作为示例数据，便于快速验证文档加载、切分、入库和问答流程。

## 当前实现特点

- 同时具备 RAG 和 Agent 两层能力
- 检索链路相对清晰，适合教学和原型演示
- 支持 PDF 导出工具调用
- 支持健康档案类多格式文档加载

## 已知限制

- 代码中 `OPENAI_API_KEY` 与 `ChatTongyi` / DashScope 的使用方式存在一定命名混用，更适合在后续统一配置命名
- `/stream` 接口当前直接调用 `agent.stream(query)`，流式行为是否稳定还取决于底层 Agent 执行方式
- `pdf_generator.py` 当前生成能力较基础，仅输出简单段落内容
- `requirements.txt` 为演示级配置，实际部署时建议补充版本约束

## 后续可优化方向

- 增加患者维度过滤与权限控制
- 支持更规范的引用来源展示
- 增加结构化报告模板
- 使用更可靠的 rerank 模型替代当前简单重排逻辑
- 将前端页面与 FastAPI 服务整合成完整应用

## 适用场景

- 健康档案智能问答 Demo
- 医疗/体检文档 RAG 原型验证
- LangChain Agent + Milvus 学习项目
- 个人健康数据管理与报告生成实验项目
