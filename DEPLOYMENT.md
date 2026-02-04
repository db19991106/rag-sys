# RAG 系统部署文档

本文档提供了 RAG (Retrieval-Augmented Generation) 系统的详细部署指南。

## 1. 系统要求

### 硬件要求
- **CPU**: 至少 4 核，推荐 8 核以上
- **内存**: 至少 16GB，推荐 32GB 以上
- **磁盘**: 至少 50GB 可用空间
- **GPU**: 可选，用于加速模型推理

### 软件要求
- **操作系统**: Ubuntu 20.04+/CentOS 7+/Windows 10+
- **Python**: 3.8-3.12
- **Node.js**: 16.0+ (用于前端)
- **Git**: 2.0+

## 2. 安装步骤

### 2.1 克隆代码库

```bash
git clone <repository-url>
cd rag
```

### 2.2 安装后端依赖

```bash
# 进入后端目录
cd backend

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Linux/Mac
source venv/bin/activate
# Windows
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装额外的切分依赖（如果需要）
chmod +x install_chunking_deps.sh
./install_chunking_deps.sh
```

### 2.3 安装前端依赖

```bash
# 进入前端目录
cd ../frontend

# 安装依赖
npm install
```

## 3. 配置说明

### 3.1 后端配置

在 `backend` 目录下创建 `.env` 文件，基于 `.env.example` 模板：

```env
# 应用配置
APP_NAME=RAG Backend
APP_VERSION=1.0.0
DEBUG=False
HOST=0.0.0.0
PORT=8000

# 文件上传配置
UPLOAD_DIR=./data/docs
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS=.txt,.pdf,.docx,.md,.csv,.json,.html,.xml,.pptx,.xlsx

# 向量数据库配置
VECTOR_DB_TYPE=faiss  # faiss, milvus, qdrant
VECTOR_DB_DIR=./vector_db
FAISS_DIMENSION=768

# 嵌入模型配置
EMBEDDING_MODEL_TYPE=sentence-transformers
EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu  # cpu, cuda

# LLM 配置
LLM_PROVIDER=local  # openai, anthropic, local
LLM_MODEL=Qwen2.5-7B-Instruct

# 本地 LLM 配置
LOCAL_LLM_MODEL_PATH=./data/models/Qwen2.5-7B-Instruct
LOCAL_LLM_DEVICE=cpu  # cpu, cuda

# JWT 配置
JWT_SECRET_KEY=your-secret-key-change-in-production-environment
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
AUDIT_LOG_FILE=./logs/audit.log
```

### 3.2 前端配置

在 `frontend` 目录下创建 `.env` 文件：

```env
# 开发环境
VITE_API_BASE_URL=http://localhost:8000/api

# 生产环境
# VITE_API_BASE_URL=https://your-api-domain/api
```

## 4. 模型下载

### 4.1 下载嵌入模型

嵌入模型会在首次运行时自动下载。默认使用 `BAAI/bge-small-zh-v1.5`。

### 4.2 下载本地 LLM 模型

使用提供的脚本下载本地 LLM 模型：

```bash
# 进入后端目录
cd backend

# 从 Hugging Face 下载
python download_model.py

# 或从 ModelScope 下载（国内推荐）
python download_model_modelscope.py
```

## 5. 启动服务

### 5.1 启动后端服务

```bash
# 进入后端目录
cd backend

# 激活虚拟环境
source venv/bin/activate

# 启动服务
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5.2 启动前端服务

```bash
# 进入前端目录
cd frontend

# 启动开发服务器
npm run dev
```

### 5.3 构建生产版本

```bash
# 构建前端
cd frontend
npm run build

# 构建产物会在 dist 目录
```

## 6. 服务管理

### 6.1 使用 systemd 管理后端服务

创建服务文件 `/etc/systemd/system/rag-backend.service`：

```ini
[Unit]
Description=RAG Backend Service
After=network.target

[Service]
User=your-user
WorkingDirectory=/path/to/rag/backend
ExecStart=/path/to/rag/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启用并启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-backend
sudo systemctl start rag-backend
```

### 6.2 使用 Nginx 反向代理

创建 Nginx 配置文件：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 前端静态文件
    location / {
        root /path/to/rag/frontend/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # 后端 API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## 7. 数据库初始化

首次启动时，系统会自动创建必要的目录结构和初始化向量数据库。

### 7.1 初始文档上传

1. 启动服务后，访问前端页面
2. 使用默认管理员账号登录：
   - 用户名: admin
   - 密码: 123456
3. 进入「文档管理」页面
4. 上传初始文档

## 8. 常见问题

### 8.1 依赖安装失败

- 确保 Python 版本正确
- 尝试升级 pip: `pip install --upgrade pip`
- 对于大模型依赖，可能需要更多内存

### 8.2 模型下载失败

- 检查网络连接
- 对于国内用户，推荐使用 ModelScope 下载
- 确保磁盘空间充足

### 8.3 服务启动失败

- 检查端口是否被占用
- 查看日志文件: `backend/logs/app.log`
- 确保配置文件正确

### 8.4 前端无法连接后端

- 检查后端服务是否运行
- 确认 API 地址配置正确
- 检查防火墙设置

## 9. 部署架构

### 9.1 开发环境

```
┌─────────────┐     ┌─────────────┐
│ 前端 (Vite) │────>│ 后端 (FastAPI) │
└─────────────┘     └─────────────┘
                          │
                          ▼
                  ┌─────────────┐
                  │ 向量数据库  │
                  └─────────────┘
                          │
                          ▼
                  ┌─────────────┐
                  │ 本地 LLM    │
                  └─────────────┘
```

### 9.2 生产环境

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Nginx     │────>│ 前端静态文件 │     │ 后端 (FastAPI) │
└─────────────┘     └─────────────┘     └─────────────┘
                          │                       │
                          └───────────────────────┘
                                          │
                                          ▼
                                  ┌─────────────┐
                                  │ 向量数据库  │
                                  └─────────────┘
                                          │
                                          ▼
                                  ┌─────────────┐
                                  │  LLM 服务   │
                                  └─────────────┘
```

## 10. 性能优化建议

### 10.1 硬件优化
- 使用 SSD 存储
- 对于生产环境，推荐使用 GPU 加速
- 增加内存以支持更大的模型

### 10.2 配置优化
- 生产环境中设置 `DEBUG=False`
- 调整 `FAISS_INDEX_TYPE` 为 `HNSW` 以获得更好的检索性能
- 根据硬件情况调整批处理大小

### 10.3 部署优化
- 使用容器化部署（Docker）
- 考虑使用负载均衡器处理高并发
- 实现缓存机制减少重复计算

## 11. 安全注意事项

- 生产环境中修改默认密码
- 使用 HTTPS 保护传输数据
- 限制文件上传大小和类型
- 定期更新依赖包
- 配置适当的文件权限

## 12. 监控与维护

详见 [MAINTENANCE_GUIDE.md](MAINTENANCE_GUIDE.md) 文档。
