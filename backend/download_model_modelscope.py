#!/usr/bin/env python3
"""
从 ModelScope 下载 Qwen2.5-0.5B-Instruct 模型
"""

from modelscope import snapshot_download
import os
from pathlib import Path

def download_qwen_model():
    """从 ModelScope 下载 Qwen2.5-0.5B-Instruct 模型"""

    print("=" * 70)
    print("从 ModelScope 下载 Qwen2.5-0.5B-Instruct 模型")
    print("=" * 70)
    print()

    # 模型目录
    model_dir = Path("/root/autodl-tmp/rag/backend/data/models/Qwen2.5-0.5B-Instruct")
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"模型将下载到: {model_dir.absolute()}")
    print()
    print("模型信息:")
    print("  - 模型名称: Qwen/Qwen2.5-0.5B-Instruct")
    print("  - 模型大小: 约 1GB")
    print("  - 适用场景: 中文对话、RAG、问答")
    print()

    try:
        print("开始下载...")
        print("-" * 70)

        model_dir = snapshot_download(
            'Qwen/Qwen2.5-0.5B-Instruct',
            cache_dir='/root/autodl-tmp/rag/backend/data/models',
            revision='master'
        )

        print("-" * 70)
        print()
        print("=" * 70)
        print("✅ 模型下载完成!")
        print("=" * 70)
        print()
        print(f"模型路径: {model_dir}")
        print()

        # 检查下载的文件
        print("下载的文件:")
        print("-" * 70)
        for item in sorted(model_dir.iterdir()):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.name:40s} {size_mb:>8.2f} MB")
            elif item.is_dir():
                print(f"  {item.name}/ (目录)")
        print()

        print("=" * 70)
        print("下一步配置:")
        print("=" * 70)
        print()
        print("在 .env 文件中添加:")
        print("  LLM_PROVIDER=local")
        print("  LLM_MODEL=Qwen2.5-0.5B-Instruct")
        print(f"  LOCAL_LLM_MODEL_PATH={model_dir}")
        print("  LOCAL_LLM_DEVICE=cpu")
        print()
        print("或直接修改 config.py:")
        print("  llm_provider: str = \"local\"")
        print(f"  local_llm_model_path: str = \"{model_dir}\"")
        print("  local_llm_device: str = \"cpu\"")
        print()
        print("然后重启后端服务:")
        print("  python main.py")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ 下载失败!")
        print("=" * 70)
        print(f"错误信息: {str(e)}")
        print()
        print("可能的原因:")
        print("  1. 网络连接问题")
        print("  2. ModelScope 服务暂时不可用")
        print("  3. 磁盘空间不足")
        print()
        print("建议:")
        print("  1. 检查网络连接")
        print("  2. 检查磁盘空间: df -h")
        print("  3. 稍后重试")
        print()
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = download_qwen_model()

    if success:
        print("🎉 配置完成！现在可以开始使用本地 LLM 了。")
    else:
        print("❌ 下载失败，请检查错误信息后重试。")