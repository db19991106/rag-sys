from modelscope import snapshot_download
import shutil
import os

# 使用 cache_dir 下载到缓存目录
cache_dir = "/root/autodl-tmp/rag/backend/data/models"
model_id = "BAAI/bge-reranker-base"

model_dir = snapshot_download(
    model_id=model_id,
    cache_dir=cache_dir,
)

print(f"模型下载完成: {model_dir}")

# 如果下载的目录名不对，重命名
target_dir = os.path.join(cache_dir, "bge-reranker-base")
if model_dir != target_dir and os.path.exists(model_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(model_dir, target_dir)
    print(f"已重命名为: {target_dir}")
