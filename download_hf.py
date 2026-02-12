import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id="BAAI/bge-reranker-base",
    local_dir="/root/autodl-tmp/rag/backend/data/models/bge-reranker-base",
    local_dir_use_symlinks=False
)
print(f"下载完成: {model_dir}")
