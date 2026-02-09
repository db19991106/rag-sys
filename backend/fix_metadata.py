#!/usr/bin/env python3
"""
修复向量数据库元数据
为所有向量添加正确的标题、文档名称和其他元数据
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/rag/backend")


def fix_metadata():
    """修复元数据，添加标题和文档信息"""

    metadata_path = Path("/root/autodl-tmp/rag/backend/vector_db/faiss_metadata.json")

    # 读取现有元数据
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"发现 {len(metadata)} 个向量需要修复")

    # 差旅费报销制度文档块 (baoxiao.md) - 17个向量
    baoxiao_mappings = [
        {
            "title": "差旅费报销制度-总则",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第一章",
        },
        {
            "title": "差旅费报销制度-总则",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第一章",
        },
        {
            "title": "差旅费报销制度-总则",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第一章",
        },
        {
            "title": "差旅费报销制度-出差审批",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第二章",
        },
        {
            "title": "差旅费报销制度-出差审批",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第二章",
        },
        {
            "title": "差旅费报销制度-费用标准",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第三章",
        },
        {
            "title": "差旅费报销制度-费用标准",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第三章",
        },
        {
            "title": "差旅费报销制度-费用标准",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第三章",
        },
        {
            "title": "差旅费报销制度-费用标准",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第三章",
        },
        {
            "title": "差旅费报销制度-费用标准",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第三章",
        },
        {
            "title": "差旅费报销制度-报销流程",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第四章",
        },
        {
            "title": "差旅费报销制度-报销流程",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第四章",
        },
        {
            "title": "差旅费报销制度-报销流程",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第四章",
        },
        {
            "title": "差旅费报销制度-附则",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第五章",
        },
        {
            "title": "差旅费报销制度-附则",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第五章",
        },
        {
            "title": "差旅费报销制度-附则",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第五章",
        },
        {
            "title": "差旅费报销制度-附则",
            "document": "baoxiao.md",
            "category": "财务制度",
            "section": "第五章",
        },
    ]

    print(f"所有 {len(metadata)} 个向量都是 baoxiao.md 文档向量")

    # 修复元数据
    updated_count = 0

    # 获取排序后的键
    sorted_ids = sorted(metadata.keys(), key=lambda x: int(x))

    for idx, vec_id in enumerate(sorted_ids):
        meta = metadata[vec_id]
        if idx < len(baoxiao_mappings):
            meta.update(baoxiao_mappings[idx])
            updated_count += 1

        # 添加通用字段 (使用正确的键名以兼容retriever)
        meta["document_name"] = meta.get(
            "document", "baoxiao.md"
        )  # retriever期望的键名
        meta["document_id"] = "baoxiao_doc_001"
        meta["chunk_id"] = f"chunk_{vec_id}"
        meta["chunk_num"] = int(vec_id) + 1
        meta["source"] = f"file://data/docs/baoxiao.md"
        meta["updated_at"] = "2026-02-08"

    # 保存修复后的元数据
    backup_path = metadata_path.with_suffix(".json.backup")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ 已创建备份: {backup_path}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ 已更新元数据: {metadata_path}")

    # 显示修复结果
    print(f"\n修复完成! 更新了 {updated_count}/{len(metadata)} 个向量")

    # 显示示例
    first_id = list(metadata.keys())[0]
    print(f"\n示例 (ID: {first_id}):")
    print(json.dumps(metadata[first_id], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    fix_metadata()
