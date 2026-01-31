# 本地 LLM 配置指南

本指南将帮助你在 RAG 系统中配置本地 LLM 模型，避免使用 OpenAI API。

## 快速开始

### 方式一：使用自动配置脚本（推荐）

```bash
cd backend
python download_model.py
```

脚本会：
1. 检查并安装必要的依赖
2. 下载 Qwen2.5-0.5B-Instruct 模型
3. 提供配置说明

### 方式二：手动配置

#### 1. 安装依赖

```bash
pip install transformers torch huggingface_hub accelerate
```

#### 2. 下载模型

推荐使用 Qwen2.5-0.5B-Instruct（轻量级，适合 CPU）：

```bash
# 使用 huggingface-cli
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir ./data/models/Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir-use-symlinks False \
  --trust-remote-code

# 或使用 Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-0.5B-Instruct',
    local_dir='./data/models/Qwen/Qwen2.5-0.5B-Instruct',
    trust_remote_code=True
)
"
```

其他推荐模型：
- **Qwen2.5-1.5B-Instruct** (中等性能)
- **THUDM/chatglm3-6b** (对话模型)
- **01-ai/Yi-1.5-6B-Chat** (高质量)

#### 3. 配置环境变量

编辑 `.env` 文件：

```env
LLM_PROVIDER=local
LLM_MODEL=Qwen2.5-0.5B-Instruct
LOCAL_LLM_MODEL_PATH=./data/models/Qwen/Qwen2.5-0.5B-Instruct
LOCAL_LLM_DEVICE=cpu  # 如果有 GPU，改为 cuda
LOCAL_LLM_LOAD_IN_8BIT=False
```

或直接修改 `config.py`：

```python
llm_provider: str = "local"
local_llm_model_path: str = "./data/models/Qwen/Qwen2.5-0.5B-Instruct"
local_llm_device: str = "cpu"
```

#### 4. 重启服务

```bash
python main.py
```

## 模型选择建议

### CPU 环境
- **Qwen2.5-0.5B-Instruct** (推荐)
  - 大小: ~1GB
  - 显存: 不需要
  - 速度: 较快
  - 质量: 良好

- **Qwen2.5-1.5B-Instruct**
  - 大小: ~3GB
  - 显存: 不需要
  - 速度: 中等
  - 质量: 较好

### GPU 环境
- **Qwen2.5-7B-Instruct**
  - 大小: ~14GB
  - 显存: 8-12GB
  - 速度: 快
  - 质量: 优秀

- **THUDM/chatglm3-6b**
  - 大小: ~12GB
  - 显存: 6-8GB
  - 速度: 快
  - 质量: 优秀

## 性能优化

### CPU 优化
```env
LOCAL_LLM_DEVICE=cpu
# 使用量化减少内存占用
LOCAL_LLM_LOAD_IN_8BIT=True
```

### GPU 优化
```env
LOCAL_LLM_DEVICE=cuda
# 启用量化以减少显存占用
LOCAL_LLM_LOAD_IN_8BIT=True
```

### 生成参数调整
```env
LLM_TEMPERATURE=0.7      # 温度：0-1，越高越随机
LLM_MAX_TOKENS=2000      # 最大生成长度
```

## 故障排除

### 1. 下载失败

如果网络问题导致下载失败，可以：

```bash
# 使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com
python download_model.py
```

或手动下载：
1. 访问 https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
2. 下载所有文件到 `./data/models/Qwen/Qwen2.5-0.5B-Instruct/`

### 2. 内存不足

- 使用更小的模型（如 0.5B）
- 启用 8-bit 量化
- 减少最大生成长度

### 3. 生成速度慢

- 使用 GPU 而非 CPU
- 使用量化模型
- 减少最大生成长度

### 4. 模型加载错误

确保已安装所有依赖：
```bash
pip install transformers torch accelerate
```

检查模型路径是否正确：
```bash
ls ./data/models/Qwen/Qwen2.5-0.5B-Instruct/
```

## 测试配置

启动服务后，在智能对话页面输入问题测试：

```
RAG的核心流程是什么？
```

如果看到回答，说明配置成功！

## 其他 LLM 提供商

如果你想使用云端 LLM：

### OpenAI
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your_openai_api_key
```

### Anthropic
```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-haiku-20240307
LLM_API_KEY=your_anthropic_api_key
```

## 参考资料

- [Qwen 模型文档](https://huggingface.co/Qwen)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [本地 LLM 最佳实践](https://huggingface.co/docs/transformers/main_classes/quantization)