# RAG 智能问答系统

一个功能完整的检索增强生成（RAG）系统，支持文档上传、智能切分、向量化存储、语义检索和智能问答。

## ✨ 特性

### 📄 文档管理
- 支持多格式文档上传（TXT、PDF、DOCX、Markdown）
- 文档预览和内容查看
- 批量导入和删除
- 文档分类管理

### 🔪 智能切分
- 4种切分策略：字符、句子、段落、自定义规则
- 可配置切分长度和重叠度
- 切分质量评估和建议
- 片段编辑功能（删除、合并）

### 🧠 向量化处理
- 支持多种嵌入模型：
  - Sentence Transformers
  - BGE（中文优化）
  - OpenAI Embeddings
- 本地模型缓存，避免重复下载
- 批量向量化处理

### 🗃️ 向量数据库
- 支持3种向量数据库：
  - FAISS（本地高性能）
  - Milvus（分布式）
  - Qdrant（云原生）
- 支持多种索引类型（HNSW、IVF、PQ、Flat）
- 持久化存储

### 🔍 智能检索
- 多种相似度算法（余弦、欧氏、点积）
- Top-K检索配置
- 相似度阈值过滤
- 检索结果可视化

### 💬 智能问答
- 完整的RAG流程
- 支持多种LLM：
  - OpenAI GPT系列
  - Anthropic Claude系列
- 上下文构建和Prompt工程
- 答案来源引用

### 🔒 安全特性
- JWT身份认证
- API速率限制
- CORS跨域配置
- 文件类型和大小验证

## 🏗️ 技术架构

### 前端技术栈
- **框架**: React 19.2.0 + TypeScript 5.9.3
- **构建工具**: Vite 7.2.4
- **路由**: React Router DOM 7.13.0
- **数据可视化**: ECharts 6.0.0
- **拖拽排序**: SortableJS 1.15.6

### 后端技术栈
- **Web框架**: FastAPI 0.115.0
- **ASGI服务器**: Uvicorn 0.32.0
- **数据验证**: Pydantic 2.10.3
- **认证**: JWT (python-jose)
- **速率限制**: SlowAPI 0.1.9

### 文档处理
- **PDF**: pypdf 5.1.0
- **Word**: python-docx 1.1.2
- **Markdown**: markdown 3.7
- **文件检测**: python-magic 0.4.27

### 向量嵌入
- **Sentence Transformers**: sentence-transformers 2.7.0
- **BGE模型**: FlagEmbedding 1.2.7
- **OpenAI**: openai 1.57.0

### 向量数据库
- **FAISS**: faiss-cpu 1.8.0
- **Milvus**: pymilvus 2.4.9
- **Qdrant**: qdrant-client 1.12.1

### LLM集成
- **OpenAI**: openai 1.57.0
- **Anthropic**: anthropic 0.42.0

## 📁 项目结构

```
rag/
├── backend/                    # 后端服务
│   ├── api/                   # API路由
│   │   ├── documents.py      # 文档管理API
│   │   ├── chunking.py       # 文档切分API
│   │   ├── embedding.py      # 向量嵌入API
│   │   ├── vector_db.py      # 向量数据库API
│   │   ├── retrieval.py      # 检索API
│   │   └── rag.py            # RAG生成API
│   ├── services/             # 业务逻辑层
│   │   ├── document_manager.py
│   │   ├── chunker.py
│   │   ├── embedding.py
│   │   ├── vector_db.py
│   │   ├── retriever.py
│   │   └── rag_generator.py
│   ├── data/                 # 数据目录
│   │   ├── docs/            # 上传的文档
│   │   └── models/          # 嵌入模型缓存
│   ├── vector_db/            # 向量数据库
│   │   ├── faiss_index
│   │   ├── faiss_metadata.json
│   │   └── documents.json
│   ├── uploads/              # 临时上传目录
│   ├── logs/                 # 日志文件
│   ├── config.py            # 配置管理
│   ├── main.py              # 应用入口
│   └── requirements.txt     # Python依赖
│
├── frontend/                  # 前端应用
│   ├── src/
│   │   ├── pages/           # 页面组件
│   │   │   ├── Login.tsx
│   │   │   ├── Chat.tsx
│   │   │   ├── RagDocManage.tsx
│   │   │   ├── Chunk.tsx
│   │   │   ├── Embedding.tsx
│   │   │   ├── Retrieval.tsx
│   │   │   ├── Generate.tsx
│   │   │   └── History.tsx
│   │   ├── components/      # 通用组件
│   │   ├── contexts/        # React Context
│   │   ├── services/        # API服务
│   │   ├── types/           # TypeScript类型
│   │   └── utils/           # 工具函数
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.10+
- **Node.js**: 18+
- **npm**: 9+

### 后端安装

```bash
cd backend

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，配置必要的参数

# 启动服务
python main.py
```

后端服务将在 `http://localhost:8000` 启动。

### 前端安装

```bash
cd frontend

# 安装依赖
npm install

# 配置环境变量
# 编辑 .env 文件，设置 API_BASE_URL
echo "VITE_API_BASE_URL=http://localhost:8000" > .env

# 启动开发服务器
npm run dev
```

前端应用将在 `http://localhost:5173` 启动。

## ⚙️ 配置说明

### 后端配置 (.env)

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
UPLOAD_DIR=./data/docs
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

# LLM 配置
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

### 前端配置 (.env)

```env
VITE_API_BASE_URL=http://localhost:8000
```

## 📖 使用指南

### 1. 登录系统

默认账号：
- 用户名: `admin`
- 密码: `123456`

### 2. 上传文档

- 进入"文档管理"页面
- 点击"上传文档"按钮
- 选择要上传的文件（支持TXT、PDF、DOCX、Markdown）
- 等待文档解析完成

### 3. 切分文档

- 进入"文档切分"页面
- 选择要切分的文档
- 配置切分策略：
  - **字符切分**: 按固定字符数切分
  - **句子切分**: 保持句子完整性
  - **段落切分**: 保持段落结构
  - **自定义规则**: 使用正则表达式
- 设置切分长度和重叠度
- 点击"执行切分"

### 4. 向量化处理

- 切分完成后，自动进行向量化
- 或进入"向量索引"页面手动向量化
- 选择嵌入模型（BGE、Sentence Transformers、OpenAI）
- 等待向量化完成

### 5. 配置检索参数

- 进入"查询检索"页面
- 配置检索参数：
  - Top-K: 返回结果数量
  - 相似度阈值: 过滤低相似度结果
  - 相似度算法: 余弦/欧氏/点积
- 保存配置

### 6. 智能问答

- 进入"智能对话"页面
- 输入问题
- 系统自动：
  1. 检索相关文档片段
  2. 构建上下文
  3. 调用LLM生成答案
  4. 返回答案和引用来源

## 🔧 API文档

启动后端服务后，访问 `http://localhost:8000/docs` 查看完整的API文档。

### 主要API端点

#### 文档管理
```
POST   /api/documents/upload          # 上传文档
GET    /api/documents/list            # 获取文档列表
GET    /api/documents/{doc_id}        # 获取文档详情
DELETE /api/documents/{doc_id}        # 删除文档
```

#### 文档切分
```
POST   /api/chunking/split            # 切分文档
POST   /api/chunking/embed            # 向量化片段
```

#### 向量嵌入
```
POST   /api/embedding/load            # 加载嵌入模型
GET    /api/embedding/status          # 获取模型状态
```

#### 智能检索
```
POST   /api/retrieval/search          # 执行检索
```

#### RAG生成
```
POST   /api/rag/generate              # RAG生成回答
```

## 🔄 工作流程

```
文档上传
    ↓
文档解析存储
    ↓
智能切分（字符/句子/段落/自定义）
    ↓
向量化处理（BGE/ST/OpenAI）
    ↓
存储到向量数据库（FAISS/Milvus/Qdrant）
    ↓
用户提问
    ↓
查询向量化
    ↓
向量检索（Top-K搜索）
    ↓
构建上下文
    ↓
LLM生成答案
    ↓
返回答案+引用
```

## 🛠️ 生产部署

### 使用Uvicorn（推荐）

```bash
cd backend

# 启动生产服务器
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 使用Docker

```bash
# 构建后端镜像
docker build -f backend/Dockerfile -t rag-backend .

# 构建前端镜像
docker build -f frontend/Dockerfile -t rag-frontend .

# 使用Docker Compose启动
docker-compose up -d
```

### 使用Nginx（前端）

```bash
cd frontend
npm run build

# 将dist目录部署到Nginx
sudo cp -r dist/* /var/www/html/
```

## 📊 性能优化

### 模型缓存
系统自动缓存下载的嵌入模型到 `./data/models/` 目录，避免重复下载。

### 批量处理
向量化支持批量处理，默认批大小为32，可根据硬件配置调整。

### 索引选择
- **HNSW**: 高性能近似搜索，适合大规模数据
- **IVF**: 倒排文件索引，平衡精度和速度
- **Flat**: 精确搜索，适合小规模数据

### 检索优化
- 使用合适的相似度阈值过滤低质量结果
- 根据数据量调整Top-K值
- 选择合适的索引类型

## 🔐 安全建议

1. **修改默认密码**: 生产环境请修改JWT密钥和用户凭证
2. **启用HTTPS**: 使用SSL/TLS加密通信
3. **配置防火墙**: 限制API访问端口
4. **定期更新**: 及时更新依赖包和安全补丁
5. **API密钥管理**: 不要将API密钥提交到版本控制

## 🐛 故障排除

### 模型加载失败
- 检查网络连接
- 确认磁盘空间充足
- 尝试手动下载模型到 `./data/models/` 目录

### 向量化速度慢
- 减少批大小
- 使用GPU加速（修改 `EMBEDDING_DEVICE=gpu`）
- 选择更小的嵌入模型

### 检索结果不准确
- 调整相似度阈值
- 尝试不同的切分策略
- 优化Prompt模板
- 使用更大的嵌入模型

### 前端无法连接后端
- 检查CORS配置
- 确认后端服务正常运行
- 检查API_BASE_URL配置

## 📝 开发指南

### 添加新的嵌入模型

1. 在 `backend/services/embedding.py` 中实现新的模型类
2. 在 `EmbeddingModelType` 枚举中添加新类型
3. 在 `EmbeddingService` 中添加加载逻辑

### 添加新的向量数据库

1. 在 `backend/services/vector_db.py` 中实现新的数据库类
2. 在 `VectorDBType` 枚举中添加新类型
3. 在 `VectorDatabaseManager` 中添加创建逻辑

### 添加新的页面

1. 在 `frontend/src/pages/` 中创建新的页面组件
2. 在 `frontend/src/App.tsx` 中添加路由
3. 在 `frontend/src/services/api.ts` 中添加API调用

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题或建议，请通过Issue联系我们。

---

**最后更新**: 2026-01-29