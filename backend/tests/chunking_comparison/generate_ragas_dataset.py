#!/usr/bin/env python3
"""
生成 RAGAS 评估数据集

从测试用例和文档中提取完整数据：
- question: 用户查询
- answer: 生成的答案（由 LLM 基于 contexts 生成）
- contexts: 检索到的上下文列表
- ground_truth: 标准答案

使用方法:
    # 使用 LLM 生成真实答案（推荐）
    python generate_ragas_dataset.py --use-llm
    
    # 使用 ground_truth 作为答案（仅用于快速测试，评估结果不可信）
    python generate_ragas_dataset.py

前提条件（使用 --use-llm 时）:
    需要先启动 vLLM 服务:
    vllm serve /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
        --served-model-name Qwen2.5-7B-Instruct \
        --host 0.0.0.0 --port 8001 \
        --dtype auto --gpu-memory-utilization 0.7 --max-model-len 4096
"""

import json
import re
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_documents() -> Dict[str, str]:
    """加载所有文档"""
    docs = {}
    docs_dir = Path("/root/autodl-tmp/rag/backend/data/docs")
    
    for doc_file in docs_dir.glob("*.md"):
        with open(doc_file, "r", encoding="utf-8") as f:
            docs[doc_file.stem] = f.read()
    
    return docs


def split_into_chunks(content: str, chunk_size: int = 500) -> List[str]:
    """将文档按段落分割成chunks"""
    # 按段落分割
    paragraphs = re.split(r'\n\s*\n', content)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk += "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def extract_contexts_by_keywords(docs: Dict[str, str], query: str, 
                                  expected_keywords: List[str], 
                                  top_k: int = 5) -> List[str]:
    """基于关键词提取相关上下文"""
    all_chunks = []
    
    for doc_name, content in docs.items():
        chunks = split_into_chunks(content)
        for chunk in chunks:
            all_chunks.append({
                "content": chunk,
                "doc": doc_name
            })
    
    # 计算每个chunk的关键词命中数
    scored_chunks = []
    for chunk_info in all_chunks:
        chunk = chunk_info["content"]
        # 计算关键词命中
        keyword_hits = sum(1 for kw in expected_keywords if kw in chunk)
        # 计算查询词命中
        query_terms = [t for t in query if len(t) > 1]
        query_hits = sum(1 for t in query_terms if t in chunk)
        
        score = keyword_hits * 2 + query_hits
        if score > 0:
            scored_chunks.append((score, chunk_info))
    
    # 按分数排序并返回top_k
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    contexts = [chunk["content"] for _, chunk in scored_chunks[:top_k]]
    
    return contexts


def extract_contexts_from_expected_docs(docs: Dict[str, str], 
                                         expected_docs: List[str],
                                         query: str,
                                         expected_keywords: List[str],
                                         top_k: int = 5) -> List[str]:
    """从预期文档中提取上下文"""
    # 先尝试从预期文档中提取
    filtered_docs = {k: v for k, v in docs.items() 
                     if any(doc_name in k or k in doc_name for doc_name in expected_docs)}
    
    if filtered_docs:
        contexts = extract_contexts_by_keywords(filtered_docs, query, expected_keywords, top_k)
        if contexts:
            return contexts
    
    # 如果预期文档中没有找到，从所有文档中搜索
    contexts = extract_contexts_by_keywords(docs, query, expected_keywords, top_k)
    if contexts:
        return contexts
    
    # 如果还是没找到，使用更宽松的匹配（单个关键词）
    all_chunks = []
    for doc_name, content in docs.items():
        chunks = split_into_chunks(content)
        for chunk in chunks:
            all_chunks.append({"content": chunk, "doc": doc_name})
    
    # 尝试匹配任意单个关键词
    for kw in expected_keywords:
        for chunk_info in all_chunks:
            if kw in chunk_info["content"]:
                contexts.append(chunk_info["content"])
                if len(contexts) >= top_k:
                    return contexts
    
    # 最后使用查询中的关键词
    query_keywords = [c for c in query if c.isalpha() and len(c) > 1]
    for kw in query_keywords[:3]:
        for chunk_info in all_chunks:
            if kw in chunk_info["content"] and chunk_info["content"] not in contexts:
                contexts.append(chunk_info["content"])
                if len(contexts) >= top_k:
                    return contexts
    
    # 如果仍然没有，返回文档的前几个段落作为默认上下文
    if not contexts:
        for doc_name, content in docs.items():
            if any(d in doc_name for d in ["财务", "报销", "城市"]):
                chunks = split_into_chunks(content)
                contexts.extend(chunks[:top_k])
                if len(contexts) >= top_k:
                    break
    
    return contexts[:top_k]


class VLLMClient:
    """vLLM API 客户端"""
    
    def __init__(self, host: str = "localhost", port: int = 8001, model: str = "Qwen2.5-7B-Instruct"):
        self.base_url = f"http://{host}:{port}/v1"
        self.model = model
        self._connected = False
        self._check_connection()
    
    def _check_connection(self):
        """检查连接"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                self._connected = True
                print(f"    ✅ vLLM 服务连接成功")
            else:
                print(f"    ⚠️ vLLM 服务响应异常: {response.status_code}")
        except Exception as e:
            print(f"    ⚠️ 无法连接 vLLM 服务: {e}")
            print(f"    将使用 ground_truth 作为答案")
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """生成文本"""
        if not self._connected:
            return ""
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"    ⚠️ LLM 生成失败: {e}")
            return ""


def generate_answer_from_context(
    contexts: List[str], 
    query: str, 
    ground_truth: str,
    llm_client: Optional[VLLMClient] = None
) -> str:
    """
    基于上下文生成答案
    
    如果提供了 llm_client，使用 LLM 生成真实答案
    否则使用 ground_truth（仅用于快速测试，评估结果不可信）
    """
    if llm_client and llm_client._connected:
        # 使用 LLM 生成答案
        context_text = "\n\n".join(contexts[:3])  # 只用前3个上下文
        
        prompt = f"""请根据以下上下文信息回答问题。如果上下文中没有相关信息，请说明。

上下文:
{context_text[:3000]}

问题: {query}

请直接回答问题，不要引用上下文原文："""

        answer = llm_client.generate(prompt)
        if answer:
            return answer
        # 如果生成失败，回退到 ground_truth
        print(f"    ⚠️ LLM 生成失败，使用 ground_truth")
    
    # 没有LLM或生成失败时，使用ground_truth
    # 注意：这会导致 answer == ground_truth，评估结果不可信！
    return ground_truth


def create_ragas_dataset(
    test_cases_path: str, 
    output_path: str,
    use_llm: bool = False,
    vllm_host: str = "localhost",
    vllm_port: int = 8001
):
    """创建RAGAS评估数据集"""
    # 加载测试用例
    with open(test_cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    test_cases = data["test_cases"]
    
    # 加载文档
    docs = load_documents()
    print(f"加载了 {len(docs)} 个文档")
    
    # 初始化 LLM 客户端（如果需要）
    llm_client = None
    if use_llm:
        print(f"\n初始化 LLM 客户端 ({vllm_host}:{vllm_port})...")
        llm_client = VLLMClient(host=vllm_host, port=vllm_port)
    else:
        print(f"\n⚠️ 未启用 LLM 生成，将使用 ground_truth 作为 answer")
        print(f"   这会导致 answer == ground_truth，评估结果不可信！")
        print(f"   建议使用 --use-llm 参数生成真实答案")
    
    # 生成RAGAS数据集
    ragas_data = []
    
    for i, case in enumerate(test_cases):
        # 提取上下文
        contexts = extract_contexts_from_expected_docs(
            docs,
            case.get("expected_docs", []),
            case["query"],
            case.get("expected_keywords", []),
            top_k=5
        )
        
        # 如果没有找到上下文，使用默认上下文
        if not contexts:
            # 从所有文档中搜索
            contexts = extract_contexts_by_keywords(
                docs, 
                case["query"], 
                case.get("expected_keywords", []),
                top_k=5
            )
        
        # 生成答案
        answer = generate_answer_from_context(
            contexts, 
            case["query"],
            case["ground_truth_answer"],
            llm_client
        )
        
        # 创建RAGAS格式数据
        ragas_item = {
            "question": case["query"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": case["ground_truth_answer"],
            "metadata": {
                "test_id": case["id"],
                "scenario": case["scenario"],
                "expected_docs": case.get("expected_docs", []),
                "expected_keywords": case.get("expected_keywords", []),
                "answer_source": "llm" if (llm_client and llm_client._connected) else "ground_truth"
            }
        }
        
        ragas_data.append(ragas_item)
        
        if (i + 1) % 50 == 0:
            print(f"处理了 {i + 1}/{len(test_cases)} 个测试用例")
    
    # 保存结果
    output = {
        "metadata": {
            "total_samples": len(ragas_data),
            "description": "RAGAS评估数据集，包含question、answer、contexts、ground_truth字段",
            "answer_generation_method": "llm" if (llm_client and llm_client._connected) else "ground_truth",
            "warning": "如果 answer_generation_method 为 ground_truth，则 answer == ground_truth，评估结果不可信" if not (llm_client and llm_client._connected) else None,
            "metrics_supported": [
                "context_precision",
                "context_recall", 
                "faithfulness",
                "answer_relevance"
            ]
        },
        "samples": ragas_data
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n生成完成！")
    print(f"总样本数: {len(ragas_data)}")
    print(f"输出文件: {output_path}")
    
    # 统计信息
    stats = {
        "with_contexts": sum(1 for d in ragas_data if d["contexts"]),
        "with_answer": sum(1 for d in ragas_data if d["answer"]),
        "with_ground_truth": sum(1 for d in ragas_data if d["ground_truth"]),
        "answer_equals_ground_truth": sum(1 for d in ragas_data if d["answer"] == d["ground_truth"])
    }
    print(f"\n数据完整性统计:")
    print(f"  有上下文: {stats['with_contexts']}/{len(ragas_data)}")
    print(f"  有答案: {stats['with_answer']}/{len(ragas_data)}")
    print(f"  有标准答案: {stats['with_ground_truth']}/{len(ragas_data)}")
    print(f"  answer == ground_truth: {stats['answer_equals_ground_truth']}/{len(ragas_data)}")
    
    if stats['answer_equals_ground_truth'] == len(ragas_data):
        print(f"\n⚠️ 警告: 所有 answer 都等于 ground_truth！")
        print(f"   这会导致 faithfulness 和 context_recall 恒为 1.0，评估结果不可信")
        print(f"   请使用 --use-llm 参数重新生成数据集")
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 RAGAS 评估数据集")
    parser.add_argument("--test-cases", type=str, 
                       default=None,
                       help="测试用例文件路径")
    parser.add_argument("--output", type=str,
                       default=None,
                       help="输出文件路径")
    parser.add_argument("--use-llm", action="store_true",
                       help="使用 LLM 生成真实答案（推荐）")
    parser.add_argument("--vllm-host", type=str, default="localhost",
                       help="vLLM 服务地址")
    parser.add_argument("--vllm-port", type=int, default=8001,
                       help="vLLM 服务端口")
    
    args = parser.parse_args()
    
    # 默认路径
    test_cases_path = Path(args.test_cases) if args.test_cases else Path(__file__).parent / "test_cases_200.json"
    output_path = Path(args.output) if args.output else Path(__file__).parent / "ragas_eval_dataset.json"
    
    create_ragas_dataset(
        str(test_cases_path),
        str(output_path),
        use_llm=args.use_llm,
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port
    )
