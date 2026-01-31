#!/usr/bin/env python3
"""
下载和配置本地 LLM 模型
"""

import os
from pathlib import Path


def download_qwen_model():
    """下载 Qwen2.5-0.5B-Instruct 模型"""
    print("=" * 60)
    print("下载 Qwen2.5-0.5B-Instruct 模型")
    print("=" * 60)
    print()
    print("这是一个轻量级的中文模型，适合在 CPU 上运行")
    print("模型大小: 约 1GB")
    print()

    # 确保模型目录存在
    model_dir = Path("./data/models/Qwen/Qwen2.5-0.5B-Instruct")
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"模型将下载到: {model_dir.absolute()}")
    print()

    try:
        from huggingface_hub import snapshot_download

        print("开始下载...")
        snapshot_download(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            trust_remote_code=True
        )

        print()
        print("=" * 60)
        print("✅ 模型下载完成!")
        print("=" * 60)
        print()
        print("下一步:")
        print("1. 更新 .env 文件，设置:")
        print("   LLM_PROVIDER=local")
        print("   LOCAL_LLM_MODEL_PATH=./data/models/Qwen/Qwen2.5-0.5B-Instruct")
        print()
        print("2. 或者直接修改 config.py 中的配置")
        print()
        print("3. 重启后端服务")
        print()

    except ImportError:
        print("❌ 错误: 需要安装 huggingface_hub")
        print("请运行: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        return False

    return True


def install_requirements():
    """安装必要的依赖"""
    print("=" * 60)
    print("安装必要的依赖")
    print("=" * 60)
    print()

    requirements = [
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "huggingface_hub>=0.19.0",
        "accelerate>=0.24.0",
    ]

    import subprocess

    for req in requirements:
        print(f"安装 {req}...")
        try:
            subprocess.run(
                ["pip", "install", req],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✅ {req} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {req} 安装失败")
            print(e.stderr)
            return False

    print()
    print("✅ 所有依赖安装完成!")
    return True


def main():
    """主函数"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "RAG 本地 LLM 模型配置工具" + " " * 17 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 检查是否已安装依赖
    print("检查依赖...")
    missing_deps = []

    try:
        import transformers
        print(f"✅ transformers {transformers.__version__}")
    except ImportError:
        missing_deps.append("transformers")

    try:
        import torch
        print(f"✅ torch {torch.__version__}")
        print(f"   CUDA 可用: {torch.cuda.is_available()}")
    except ImportError:
        missing_deps.append("torch")

    try:
        import huggingface_hub
        print(f"✅ huggingface_hub {huggingface_hub.__version__}")
    except ImportError:
        missing_deps.append("huggingface_hub")

    print()

    if missing_deps:
        print("❌ 缺少以下依赖:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print()

        choice = input("是否现在安装? (y/n): ").strip().lower()
        if choice == 'y':
            if not install_requirements():
                print("❌ 依赖安装失败，请手动安装:")
                print("pip install transformers torch huggingface_hub accelerate")
                return
        else:
            print("请手动安装依赖后重试")
            return

    # 下载模型
    print()
    choice = input("是否现在下载 Qwen2.5-0.5B-Instruct 模型? (y/n): ").strip().lower()
    if choice == 'y':
        download_qwen_model()
    else:
        print("您可以稍后手动下载模型")
        print("推荐模型:")
        print("  - Qwen/Qwen2.5-0.5B-Instruct (轻量级，推荐)")
        print("  - Qwen/Qwen2.5-1.5B-Instruct (中等)")
        print("  - THUDM/chatglm3-6b (对话模型)")
        print()

    print()
    print("配置说明:")
    print("-" * 60)
    print("在 .env 文件中添加:")
    print()
    print("LLM_PROVIDER=local")
    print("LLM_MODEL=Qwen2.5-0.5B-Instruct")
    print("LOCAL_LLM_MODEL_PATH=./data/models/Qwen/Qwen2.5-0.5B-Instruct")
    print("LOCAL_LLM_DEVICE=cpu  # 如果有 GPU，改为 cuda")
    print()
    print("或者直接修改 config.py 中的默认配置")
    print()


if __name__ == "__main__":
    main()