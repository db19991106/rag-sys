#!/usr/bin/env python3
"""
下载和配置本地 LLM 模型 - 修复版
支持多种下载方式
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
        import huggingface_hub

        print("开始下载...")

        # 检查huggingface_hub版本
        version = huggingface_hub.__version__
        print(f"huggingface_hub 版本: {version}")

        # 新版本(>=0.20.0)不再支持trust_remote_code参数
        try:
            # 尝试新版本API
            snapshot_download(
                repo_id="Qwen/Qwen2.5-0.5B-Instruct",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                # 移除trust_remote_code参数
            )
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                # 旧版本API
                snapshot_download(
                    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                )
            else:
                raise

        print()
        print("=" * 60)
        print("✅ 模型下载完成!")
        print("=" * 60)
        print()
        print("下一步:")
        print("1. 更新 .env 文件，设置:")
        print("   LLM_PROVIDER=local")
        print("   LLM_MODEL=Qwen2.5-0.5B-Instruct")
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
        print()
        print("💡 尝试备选方案:")
        print("   1. 使用ModelScope(国内镜像): python download_model_modelscope.py")
        print("   2. 使用Ollama: ollama pull qwen2.5:0.5b")
        print(
            "   3. 手动下载: 访问 https://modelscope.cn/models/qwen/Qwen2.5-0.5B-Instruct"
        )
        return False

    return True


def download_from_modelscope():
    """从ModelScope下载（国内镜像，更快）"""
    print("=" * 60)
    print("从 ModelScope 下载模型（国内镜像）")
    print("=" * 60)
    print()

    model_dir = Path("./data/models/Qwen/Qwen2.5-0.5B-Instruct")
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"模型将下载到: {model_dir.absolute()}")
    print()

    try:
        from modelscope import snapshot_download

        print("开始下载...")
        snapshot_download(
            model_id="qwen/Qwen2.5-0.5B-Instruct",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )

        print()
        print("=" * 60)
        print("✅ 模型下载完成!")
        print("=" * 60)
        return True

    except ImportError:
        print("❌ 错误: 需要安装 modelscope")
        print("请运行: pip install modelscope")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        return False


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
                ["pip", "install", req], check=True, capture_output=True, text=True
            )
            print(f"✅ {req} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {req} 安装失败")
            print(e.stderr)
            return False

    print()
    print("✅ 所有依赖安装完成!")
    return True


def check_dependencies():
    """检查依赖是否已安装"""
    print()
    print("检查依赖...")
    print()

    all_ok = True

    # 检查 transformers
    try:
        import transformers

        print(f"✅ transformers {transformers.__version__}")
    except ImportError:
        print("❌ transformers 未安装")
        all_ok = False

    # 检查 torch
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"✅ torch {torch.__version__}")
        print(f"   CUDA 可用: {cuda_available}")
    except ImportError:
        print("❌ torch 未安装")
        all_ok = False

    # 检查 huggingface_hub
    try:
        import huggingface_hub

        print(f"✅ huggingface_hub {huggingface_hub.__version__}")
    except ImportError:
        print("❌ huggingface_hub 未安装")
        all_ok = False

    print()
    return all_ok


def main():
    """主函数"""
    print()
    print("╔" + "═" * 56 + "╗")
    print("║" + " " * 10 + "RAG 本地 LLM 模型配置工具" + " " * 21 + "║")
    print("╚" + "═" * 56 + "╝")
    print()

    # 检查依赖
    if not check_dependencies():
        print("❌ 缺少必要的依赖")
        response = input("是否现在安装依赖? (y/n): ")
        if response.lower() == "y":
            if not install_requirements():
                print("❌ 依赖安装失败，请手动安装")
                return
        else:
            print("请先安装依赖后再运行此脚本")
            return

    # 询问下载方式
    print()
    print("选择下载方式:")
    print("1. HuggingFace (国际源)")
    print("2. ModelScope (国内镜像，推荐)")
    print()

    choice = input("请选择 (1/2): ").strip()

    if choice == "1":
        success = download_qwen_model()
    elif choice == "2":
        success = download_from_modelscope()
    else:
        print("❌ 无效选择")
        return

    if success:
        print()
        print("配置说明:")
        print("-" * 60)
        print()
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
