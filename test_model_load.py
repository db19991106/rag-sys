#!/usr/bin/env python3
# 测试直接加载本地Qwen3-Embedding-4B模型

import os
from pathlib import Path

# 禁用CodeCarbon
os.environ['ACCELERATE_DISABLE_CODE_CARBON'] = '1'

print("开始测试模型加载...")

model_path = "/root/autodl-tmp/rag/backend/data/models/Qwen3-Embedding-4B"
print(f"模型路径: {model_path}")

# 检查路径是否存在
path_obj = Path(model_path)
if path_obj.exists():
    print("✅ 模型路径存在")
else:
    print("❌ 模型路径不存在")

# 尝试加载模型
try:
    from sentence_transformers import SentenceTransformer
    print("✅ 成功导入SentenceTransformer")
    
    # 尝试直接加载本地路径
    print("尝试加载模型...")
    model = SentenceTransformer(model_path, local_files_only=True)
    print("✅ 模型加载成功!")
    print(f"模型维度: {model.get_sentence_embedding_dimension()}")
    
    # 测试编码
    test_text = "这是一个测试句子"
    embedding = model.encode(test_text)
    print(f"✅ 编码成功，嵌入向量长度: {len(embedding)}")
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
