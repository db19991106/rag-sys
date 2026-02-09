#!/usr/bin/env python3
"""
测试本地 LLM 模型加载和生成
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings

def test_model_loading():
    """测试模型加载"""
    print("=" * 70)
    print("测试本地 LLM 模型")
    print("=" * 70)
    print()

    # 显示配置
    print("当前配置:")
    print(f"  LLM Provider: {settings.llm_provider}")
    print(f"  LLM Model: {settings.llm_model}")
    print(f"  Model Path: {settings.local_llm_model_path}")
    print(f"  Device: {settings.local_llm_device}")
    print()

    # 检查模型路径
    model_path = Path(settings.local_llm_model_path)
    if not model_path.exists():
        print(f"❌ 错误: 模型路径不存在: {model_path}")
        return False

    print(f"✅ 模型路径存在: {model_path}")
    print()

    # 检查模型文件
    print("模型文件:")
    print("-" * 70)
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    all_files_exist = True
    for filename in required_files:
        file_path = model_path / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename:30s} {size_mb:>8.2f} MB")
        else:
            print(f"  ❌ {filename:30s} 缺失")
            all_files_exist = False

    print()

    if not all_files_exist:
        print("❌ 错误: 缺少必要的模型文件")
        return False

    print("✅ 所有必要的模型文件都存在")
    print()

    # 测试加载模型
    print("测试加载模型...")
    print("-" * 70)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        print("✅ Tokenizer 加载成功")

        print(f"加载模型...")
        model_kwargs = {
            "torch_dtype": torch.float16 if settings.local_llm_device == "cuda" else torch.float32,
        }

        if settings.local_llm_device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                **model_kwargs
            )
            model = model.to(settings.local_llm_device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                device_map="auto",
                **model_kwargs
            )

        print(f"✅ 模型加载成功 (设备: {settings.local_llm_device})")
        print()

        # 测试生成
        print("测试文本生成...")
        print("-" * 70)

        test_messages = [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好，请用一句话介绍一下RAG技术。"}
        ]

        text = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print(f"输入: {test_messages[1]['content']}")
        print()

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        print(f"输出: {response}")
        print()

        print("=" * 70)
        print("✅ 测试成功! 本地 LLM 可以正常工作")
        print("=" * 70)
        print()
        print("现在可以重启后端服务，开始使用本地 LLM 了:")
        print("  python main.py")
        print()

        return True

    except ImportError as e:
        print(f"❌ 错误: 缺少必要的依赖")
        print(f"   {str(e)}")
        print()
        print("请安装依赖:")
        print("  pip install transformers torch accelerate")
        return False

    except Exception as e:
        print(f"❌ 错误: 模型加载或生成失败")
        print(f"   {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)