"""
批量向量化脚本 - 修复未向量化的文档
"""

import requests
import json
import sys
import time


def get_all_documents():
    """获取所有文档列表"""
    try:
        response = requests.get("http://localhost:9000/documents/list")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ 获取文档列表失败: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 连接后端服务失败: {e}")
        print("💡 请确保后端服务已启动: python main.py")
        return []


def embed_document(doc_id, doc_name):
    """向量化单个文档"""
    try:
        print(f"  正在向量化: {doc_name}...", end=" ")
        response = requests.post(
            f"http://localhost:9000/chunking/embed?doc_id={doc_id}",
            timeout=300,  # 5分钟超时
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ 成功 ({result.get('message', '')})")
            return True
        else:
            print(f"❌ 失败 (HTTP {response.status_code})")
            try:
                error = response.json()
                print(f"     错误: {error.get('detail', '未知错误')}")
            except:
                print(f"     响应: {response.text[:100]}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ 超时 (>5分钟)")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("🚀 RAG 批量向量化工具")
    print("=" * 70)
    print()

    # 检查后端服务
    print("📝 步骤1: 检查后端服务...")
    documents = get_all_documents()

    if not documents:
        print("❌ 没有获取到文档，请检查：")
        print("   1. 后端服务是否已启动 (python main.py)")
        print("   2. 是否有文档已上传")
        return

    print(f"✅ 找到 {len(documents)} 个文档")
    print()

    # 过滤出需要向量化的文档（已切分但未索引）
    docs_to_embed = []
    for doc in documents:
        status = doc.get("status", "unknown")
        chunk_count = doc.get("chunk_count", 0)

        if status == "split" and chunk_count > 0:
            docs_to_embed.append(doc)
        elif status == "indexed":
            print(f"   ⏭️  跳过 {doc['name']} (已索引)")
        elif chunk_count == 0:
            print(f"   ⚠️  跳过 {doc['name']} (未切分)")

    print()
    print(f"📝 步骤2: 开始批量向量化 ({len(docs_to_embed)} 个文档)...")
    print()

    # 批量向量化
    success_count = 0
    fail_count = 0

    for i, doc in enumerate(docs_to_embed, 1):
        print(f"[{i}/{len(docs_to_embed)}] ", end="")

        if embed_document(doc["id"], doc["name"]):
            success_count += 1
        else:
            fail_count += 1

        # 短暂延迟，避免过载
        time.sleep(0.5)

    print()
    print("=" * 70)
    print("📊 批量向量化完成")
    print("=" * 70)
    print(f"✅ 成功: {success_count} 个文档")
    print(f"❌ 失败: {fail_count} 个文档")
    print()

    # 检查向量数据库状态
    print("📝 步骤3: 检查向量数据库状态...")
    try:
        response = requests.get("http://localhost:9000/vector-db/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ 向量数据库: {status.get('db_type', 'unknown')}")
            print(f"   总向量数: {status.get('total_vectors', 0)}")
            print(f"   维度: {status.get('dimension', 0)}")
            print(f"   状态: {status.get('status', 'unknown')}")
    except Exception as e:
        print(f"❌ 无法获取向量数据库状态: {e}")

    print()
    print("💡 提示: 现在可以测试问答功能了！")
    print("   前端页面: http://localhost:5173")


if __name__ == "__main__":
    main()
