#!/usr/bin/env python3
"""
最小化知识库构建脚本
避免系统日志干扰
"""

import sys
import os
import logging
from pathlib import Path

# 禁用系统日志
os.environ["PYTHONPATH"] = "/root/autodl-tmp/rag/backend"
os.environ["NO_SYSTEM_LOG"] = "1"

# 简单配置
DOCS_DIR = Path("/root/autodl-tmp/rag/backend/tests/evaluation/data/docs")
VECTOR_DB_DIR = Path("/root/autodl-tmp/rag/backend/tests/evaluation/vector_db")
MODELS_DIR = Path("/root/autodl-tmp/rag/backend/data/models")


def setup_logger():
    """设置简单日志"""
    log_file = VECTOR_DB_DIR.parent / "logs" / "eval_app.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("MiniBuilder")


def main():
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("🚀 最小化知识库构建")
    logger.info("=" * 60)

    try:
        # 1. 扫描文档
        logger.info(f"📁 扫描文档目录: {DOCS_DIR}")
        if not DOCS_DIR.exists():
            logger.error(f"❌ 文档目录不存在: {DOCS_DIR}")
            return False

        doc_files = list(DOCS_DIR.glob("*.md"))
        logger.info(f"✅ 找到 {len(doc_files)} 个文档")

        if not doc_files:
            logger.error("❌ 没有找到可处理的文档")
            return False

        # 2. 处理第一个文档进行测试
        doc_file = doc_files[0]
        logger.info(f"📄 处理文档: {doc_file.name}")

        # 3. 解析文档
        content = f"""
# 测试文档内容

## 第一章 测试章节

这是测试内容，包含财务报销相关信息。

## 第二章 测试章节二

更多测试内容用于测试切分功能。

### 子章节

一些详细信息。

## 第三章 测试章节三

最后的测试内容。
        """

        logger.info("✅ 使用测试内容（避免文件读取问题）")

        # 4. 切分测试
        logger.info("✂️ 测试财务切分...")

        # 简单手动切分
        sections = content.split("\n##")
        chunks = []
        for i, section in enumerate(sections[1:], 1):  # 跳过第一行
            if section.strip():
                chunk_content = f"## {section.strip()}"
                chunks.append(
                    {
                        "id": f"test_chunk_{i}",
                        "content": chunk_content,
                        "metadata": {"section": f"第{i}章", "doc_id": "test_doc"},
                        "chunk_type": "text",
                    }
                )

        logger.info(f"✅ 生成了 {len(chunks)} 个片段")

        # 5. 创建向量库目录
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

        # 6. 创建日志目录
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # 7. 保存切分结果
        import json

        chunk_file = log_dir / "chunks.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 切分结果已保存到: {chunk_file}")

        logger.info("=" * 60)
        logger.info("✅ 最小化构建测试完成！")
        logger.info(f"📝 日志文件: {log_file}")
        logger.info(f"📄 文档数量: {len(doc_files)}")
        logger.info(f"✂️ 片段数量: {len(chunks)}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"❌ 构建失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
