# RAG Backend - FastAPI

基于 FastAPI 的 RAG (Retrieval-Augmented Generation) 系统后端服务。

## 功能特性

### 1. 文档管理
- ✅ 支持多种文档格式上传 (TXT/PDF/DOCX/MD)
- ✅ 文档解析和内容提取
- ✅ 文档列表、详情、删除等操作
- ✅ 批量删除文档

### 2. 文档切分
- ✅ 按字符切分
- ✅ 按句子切分
- ✅ 按段落切分
- ✅ 自定义规则切分 (支持正则表达式)
- ✅ 可配置切分长度和重叠长度

### 3. 向量嵌入
- ✅ 支持多种嵌入模型:
  - Sentence Transformers (BGE、text2vec 等)
  - BGE (FlagEmbedding)
  - OpenAI Embeddings
- ✅ 批量向量化
- ✅ 可配置批处理大小和设备

### 4. 向量数据库
- ✅ 支持 FAISS (本地高性能)
- ✅ 支持 Milvus (分布式)
- ✅ 支持多种索引类型 (HNSW/IVF/PQ)
- ✅ 向量添加、搜索、保存和加载

### 5. 检索
- ✅ 向量相似度搜索
- ✅ 支持多种相似度算法 (余弦/欧氏/点积)
- ✅ 可配置 Top-K 和相似度阈值
- ✅ 关键词匹配和高亮

### 6. RAG 生成
- ✅ 结合检索和生成
- ✅ 支持 OpenAI GPT 模型
- ✅ 支持 Anthropic Claude 模型
- ✅ 可配置生成参数 (温度、Top-P 等)
- ✅ 上下文构建和 Prompt 优化

## 安装

### 1. 创建虚拟环境

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并修改配置:

```bash
cp .env.example .env
```

编辑 `.env` 文件，根据需要修改配置:

```env
# FastAPI 配置
APP_NAME=RAG Backend
APP_VERSION=1.0.0
DEBUG=True
HOST=0.0.0.0
PORT=8000

# 跨域配置
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]

# 文件上传配置
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=10485760
ALLOWED_EXTENSIONS=[".txt",".pdf",".docx",".md"]

# 向量数据库配置
VECTOR_DB_TYPE=faiss
VECTOR_DB_DIR=./vector_db
FAISS_INDEX_TYPE=HNSW
FAISS_DIMENSION=768

# 嵌入模型配置
EMBEDDING_MODEL_TYPE=sentence-transformers
EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DEVICE=cpu

# LLM 配置 (可选)
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

## 运行

### 开发模式 (带热重载)

```bash
python main.py
```

或使用 uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 生产模式

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API 文档

启动服务后，访问以下地址查看 API 文档:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 端点

### 文档管理

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/documents/upload` | 上传文档 |
| GET | `/documents/list` | 获取文档列表 |
| GET | `/documents/{doc_id}` | 获取文档详情 |
| DELETE | `/documents/{doc_id}` | 删除文档 |
| POST | `/documents/batch-delete` | 批量删除文档 |

### 文档切分

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/chunking/split` | 切分文档 |
| POST | `/chunking/embed` | 向量化片段 |

### 向量嵌入

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/embedding/load` | 加载嵌入模型 |
| GET | `/embedding/status` | 获取模型状态 |

### 向量数据库

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/vector-db/init` | 初始化向量数据库 |
| GET | `/vector-db/status` | 获取数据库状态 |
| POST | `/vector-db/save` | 保存数据库 |

### 检索

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/retrieval/search` | 执行检索 |

### RAG 生成

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/rag/generate` | RAG 生成回答 |

## 使用示例

### 1. 上传文档

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@document.pdf"
```

### 2. 切分文档

```bash
curl -X POST "http://localhost:8000/chunking/split?doc_id=xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "paragraph",
    "length": 500,
    "overlap": 50,
    "custom_rule": ""
  }'
```

### 3. 加载嵌入模型

```bash
curl -X POST "http://localhost:8000/embedding/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "sentence-transformers",
    "model_name": "BAAI/bge-small-zh-v1.5",
    "batch_size": 32,
    "device": "cpu"
  }'
```

### 4. 向量化片段

```bash
curl -X POST "http://localhost:8000/chunking/embed?doc_id=xxx"
```

### 5. 执行检索

```bash
curl -X POST "http://localhost:8000/retrieval/search?query=RAG是什么" \
  -H "Content-Type: application/json" \
  -d '{
    "top_k": 5,
    "similarity_threshold": 0.6,
    "algorithm": "cosine"
  }'
```

### 6. RAG 生成回答

```bash
curl -X POST "http://localhost:8000/rag/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "RAG的核心流程是什么?",
    "retrieval_config": {
      "top_k": 5,
      "similarity_threshold": 0.6,
      "algorithm": "cosine"
    },
    "generation_config": {
      "llm_provider": "openai",
      "llm_model": "gpt-3.5-turbo",
      "temperature": 0.7,
      "max_tokens": 2000,
      "top_p": 0.9,
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0
    }
  }'
```

## 项目结构

```
backend/
├── main.py                 # FastAPI 应用入口
├── config.py               # 配置管理
├── models.py               # Pydantic 数据模型
├── requirements.txt        # Python 依赖
├── .env.example           # 环境变量示例
├── api/                   # API 路由
│   ├── __init__.py
│   ├── documents.py       # 文档管理 API
│   ├── chunking.py        # 文档切分 API
│   ├── embedding.py       # 向量嵌入 API
│   ├── vector_db.py       # 向量数据库 API
│   ├── retrieval.py       # 检索 API
│   └── rag.py            # RAG 生成 API
├── services/              # 业务逻辑
│   ├── __init__.py
│   ├── document_parser.py # 文档解析
│   ├── document_manager.py # 文档管理
│   ├── chunker.py         # 文档切分
│   ├── embedding.py       # 向量嵌入
│   ├── vector_db.py       # 向量数据库
│   ├── retriever.py       # 检索器
│   └── rag_generator.py   # RAG 生成器
└── utils/                 # 工具函数
    ├── __init__.py
    ├── logger.py          # 日志配置
    └── file_utils.py      # 文件工具
```

## 技术栈

- **Web 框架**: FastAPI
- **文档处理**: pypdf, python-docx, markdown
- **向量嵌入**: sentence-transformers, FlagEmbedding, openai
- **向量数据库**: FAISS, Milvus
- **LLM**: OpenAI, Anthropic
- **其他**: Pydantic, numpy, scipy

## 注意事项

1. 首次运行时，嵌入模型会自动下载，可能需要一些时间
2. 使用 GPU 可以显著提高向量嵌入速度，在 `.env` 中设置 `EMBEDDING_DEVICE=cuda`
3. 如果使用 OpenAI 或其他 LLM，需要配置相应的 API Key
4. FAISS 索引文件会保存在 `vector_db` 目录
5. 上传的文件会保存在 `uploads` 目录

## 许可证

MIT License