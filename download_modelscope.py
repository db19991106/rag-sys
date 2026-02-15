import os
import sys
from modelscope.hub.api import HubApi
from modelscope.hub.file_download import model_file_download

# 1. 配置模型信息和目标路径
MODEL_ID = "BAAI/bge-m3"
TARGET_DIR = "/root/autodl-tmp/rag/backend/data/models/bge-m3"

# 2. 创建目标目录（确保存在）
os.makedirs(TARGET_DIR, exist_ok=True)
print(f"目标目录已确认/创建：{TARGET_DIR}")

# 3. 初始化 ModelScope API 并下载模型
try:
    api = HubApi()
    # 获取模型所有文件列表
    file_list = api.get_model_files(MODEL_ID)
    print(f"发现 {len(file_list)} 个模型文件，开始下载...")
    
    # 逐个下载文件到指定目录
    for file_info in file_list:
        # 跳过目录（Type: 'tree'）
        if file_info.get('Type') == 'tree':
            print(f"跳过目录：{file_info.get('Path')}")
            continue
        
        file_path = file_info.get('Path')
        print(f"正在下载：{file_path}")
        model_file_download(
            model_id=MODEL_ID,
            file_path=file_path,
            local_dir=TARGET_DIR
        )
    print("\n✅ 模型下载完成！目标路径：", TARGET_DIR)
    
    # 4. 验证下载结果
    print("\n📄 下载的核心文件列表：")
    core_files = ["config.json", "tokenizer.json", "pytorch_model.bin"]
    for f in core_files:
        f_path = os.path.join(TARGET_DIR, f)
        if os.path.exists(f_path):
            size = os.path.getsize(f_path) / (1024*1024*1024)  # 转GB
            print(f"  ✔️ {f} - 大小：{size:.2f} GB")
        else:
            print(f"  ❌ {f} - 未找到（可能下载失败）")
            
except Exception as e:
    print(f"\n❌ 下载出错：{str(e)}")
    sys.exit(1)