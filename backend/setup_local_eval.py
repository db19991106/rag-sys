#!/usr/bin/env python3
"""
快速设置本地模型评估环境
一键配置脚本
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"命令: {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    return result.returncode == 0


def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "RAG本地模型快速设置工具" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 步骤1: 安装依赖
    print("📦 步骤1: 安装必要依赖")
    print("-" * 60)

    deps = ["ragas", "langchain-community", "huggingface_hub", "modelscope"]

    for dep in deps:
        print(f"\n安装 {dep}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", dep], capture_output=True
        )

    print("\n✅ 依赖安装完成")

    # 步骤2: 选择模型来源
    print("\n" + "=" * 60)
    print("🤖 步骤2: 选择本地模型方案")
    print("=" * 60)
    print()
    print("方案A: 使用Ollama（最简单，推荐）")
    print("  - 无需配置，直接运行")
    print("  - 自动管理模型")
    print("  - 适合快速测试")
    print()
    print("方案B: 从ModelScope下载（中文优化）")
    print("  - 下载到本地目录")
    print("  - 完全离线运行")
    print("  - 需要较长时间下载")
    print()

    choice = input("请选择 (A/B): ").strip().upper()

    if choice == "A":
        # Ollama方案
        print("\n" + "=" * 60)
        print("🐳 Ollama方案")
        print("=" * 60)

        # 检查Ollama是否已安装
        result = subprocess.run(["which", "ollama"], capture_output=True)
        if result.returncode != 0:
            print("⚠️  Ollama未安装")
            print()
            print("安装方法:")
            print("  Linux/macOS: curl -fsSL https://ollama.com/install.sh | sh")
            print("  Docker: docker run -d -p 11434:11434 ollama/ollama")
            print()
            input("安装完成后按回车继续...")

        # 拉取模型
        print("\n📥 拉取Qwen2.5模型...")
        print("运行: ollama pull qwen2.5:0.5b")

        result = subprocess.run(["ollama", "pull", "qwen2.5:0.5b"])
        if result.returncode != 0:
            print("❌ 模型拉取失败")
            return

        print("✅ 模型拉取完成")

        # 测试连接
        print("\n🧪 测试模型连接...")
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:0.5b", "你好"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✅ 模型连接正常")
        else:
            print("⚠️  模型测试未完成，但模型已下载")

        # 配置完成提示
        print("\n" + "=" * 60)
        print("✅ 设置完成！")
        print("=" * 60)
        print()
        print("运行评估命令:")
        print()
        print("  cd /root/autodl-tmp/rag/backend")
        print("  python local_ragas_integration.py \\")
        print("    --provider ollama \\")
        print("    --model qwen2.5:0.5b \\")
        print("    --mode batch")
        print()

    elif choice == "B":
        # ModelScope方案
        print("\n" + "=" * 60)
        print("📥 ModelScope方案")
        print("=" * 60)

        print("\n开始下载模型（约1GB，可能需要5-10分钟）...")
        print()

        result = subprocess.run([sys.executable, "download_model.py"])

        if result.returncode != 0:
            print("❌ 模型下载失败")
            return

        print("\n" + "=" * 60)
        print("✅ 设置完成！")
        print("=" * 60)
        print()
        print("运行评估命令:")
        print()
        print("  cd /root/autodl-tmp/rag/backend")
        print("  python project_local_ragas.py --mode batch")
        print()

    else:
        print("❌ 无效选择")
        return

    print("提示:")
    print("  - 评估结果将保存在 evaluation_results/ 目录")
    print("  - 首次运行可能需要几分钟加载模型")
    print("  - 如需帮助，查看 LOCAL_LLM_EVALUATION_GUIDE.md")
    print()


if __name__ == "__main__":
    main()
