#!/usr/bin/env python3
# 测试pandas导入问题

print("开始测试pandas导入...")

try:
    import pandas
    print(f"成功导入pandas，版本: {pandas.__version__}")
except Exception as e:
    print(f"pandas导入失败: {e}")

print("\n测试sentence-transformers导入...")

try:
    import os
    os.environ['ACCELERATE_DISABLE_CODE_CARBON'] = '1'
    
    from sentence_transformers import SentenceTransformer
    print("成功导入sentence-transformers")
    
    # 尝试加载一个小型模型测试
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"成功加载模型，维度: {model.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"sentence-transformers导入或模型加载失败: {e}")
    import traceback
    traceback.print_exc()
