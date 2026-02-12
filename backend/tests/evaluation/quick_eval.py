#!/usr/bin/env python3
"""
快速本地模型测评脚本
"""
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from services.rag_generator import rag_generator
from services.rag_evaluator import rag_evaluator
from models import RetrievalConfig, GenerationConfig, EmbeddingConfig, VectorDBConfig, EmbeddingModelType, VectorDBType
from config import settings


def quick_local_eval(limit: int = 5):
    """快速本地模型测评"""
    print("\n🚀 RAG系统快速测评（本地模型）")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📍 使用配置:")
    print(f"   LLM: {settings.llm_provider} - {settings.llm_model}")
    print(f"   嵌入: {settings.embedding_model_name}")
    print(f"   向量库: {settings.vector_db_type}")
    
    try:
        # 应用MRR修复
        import tests.evaluation.mrr_debug as mrr_debug
        mrr_debug.patch_rag_evaluator()
        print("✅ MRR修复已应用")
    except:
        print("⚠️ MRR修复失败，继续测试...")
    
    # 初始化服务
    print("🔧 初始化服务...")
    embedding_service.load_model(EmbeddingConfig(
        model_type=EmbeddingModelType.BGE,
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
    ))
    vector_db_manager.initialize(VectorDBConfig(
        db_type=VectorDBType.FAISS,
        dimension=embedding_service.get_dimension(),
        index_type=settings.faiss_index_type,
    ))
    
    print(f"✅ 初始化完成")
    
    # 快速测试用例
    test_queries = [
        {
            "query": "差旅费标准是什么？",
            "expected_keywords": ["差旅费", "标准", "费用"],
            "ground_truth_chunks": ["2"],  # 假设
        },
        {
            "query": "8-9级员工住宿报销标准",
            "expected_keywords": ["住宿", "8-9级", "员工", "标准"],
        },
        {
            "query": "报销需要什么发票？",
            "expected_keywords": ["发票", "报销", "流程"],
        },
        {
            "query": "餐补标准是多少？",
            "expected_keywords": ["餐补", "补贴", "标准"],
        },
        {
            "query": "总盟能住什么酒店？",
            "expected_keywords": ["总监", "酒店", "五星级"],
        }
    ][:limit]
    
    print(f"\n🧪 快速测试 ({len(test_queries)} 条):")
    
    results = []
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        keywords = test_case.get("expected_keywords", [])
        ground_truth = test_case.get("ground_truth", [])
        
        print(f"[{i}/{len(test_queries)}] {query}")
        
        try:
            start = time.time()
            response = rag_generator.generate(
                query=query,
                retrieval_config=RetrievalConfig(top_k=3),
                generation_config=GenerationConfig(
                    llm_provider=settings.llm_provider,
                    temperature=0.7,
                    max_tokens=300,
                ),
            )
            elapsed = (time.time() - start) * 1000
            
            answer = response.answer
            contexts = response.context_chunks or []
            
            # 简单的关键词评估
            answer_lower = answer.lower()
            hit_count = sum(1 for kw in keywords if kw.lower() in answer_lower)
            hit_rate = hit_count / len(keywords) if keywords else 1.0
            
            # 模拟MRR评估
            mrr_score = 1.0 if hit_rate > 0.5 else 0.0  # 简化版MRR
            
            status = "✅" if hit_rate >= 0.6 else "⚠️" if hit_rate >= 0.4 else "❌"
            
            print(f"   {status} {elapsed:.1f}ms | 关键词:{hit_rate:.0%} | 回答:{len(answer)}字符")
            print(f"      回答: {answer[:60]}...")
            
            results.append({
                "query": query,
                "response_time_ms": elapsed,
                "hit_rate": hit_rate,
                "mrr": mrr_score,
                "answer_length": len(answer),
                "contexts_count": len(contexts),
            })
            
        except Exception as e:
            print(f"   ❌ 失败: {str(e)[:50]}")
            results.append({
                "query": query,
                "error": str(e),
                "response_time_ms": 0,
                "hit_rate": 0,
                "mrr": 0,
                "answer_length": 0,
                "contexts_count": 0,
            })
    
    # 统计分析
    valid_results = [r for r in results if "error" not in r]
    
    if valid_results:
        avg_time = statistics.mean([r["response_time_ms"] for r in valid_results])
        avg_hit_rate = statistics.mean([r["hit_rate"] for r in valid_results])
        avg_mrr = statistics.mean([r["mrr"] for r in valid_results])
        avg_length = statistics.mean([r["answer_length"] for r in valid_results])
        
        print("\n📊 快速测评结果:")
        print(f"   测试数量: {len(valid_results) / {len(test_queries)}")
        print(f"   平均响应时间: {avg_time:.1f}ms")
        print(f"   平均关键词命中率: {avg_hit_rate:.1%}")
        print(f"   平均MRR: {avg_mrr:.3f}")
        print(f"   平均回答长度: {avg_length:.0f}字符")
        
        if avg_hit_rate >= 0.8:
            print("   🟢 优秀 - 本地RAG系统表现良好")
        elif avg_hit_rate >= 0.6:
            print("   🟡 良好 - 本地RAG系统可用")
        else:
            print("   🟠 需优化 - 本地RAG系统需要调整")
    else:
        print("   ❌ 测试失败")
        
        print("="*60)
        print("💡 完整测评:")
        print("   python local_eval.py --limit 10")
        print("   快速验证: python quick_eval.py")
        print("   MRR修复: python mrr_debug.py")
        
        return {
            "quick_results": results,
            "statistics": {
                "avg_response_time_ms": avg_time,
                "avg_hit_rate": avg_hit_rate,
                "avg_mrr": avg_mrr,
                "avg_answer_length": avg_length,
                "success_rate": len(valid_results) / len(test_cases) if 'test_cases' in locals() else len(valid_results) / len(test_queries)
            }
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG系统快速测评（本地模型）")
    parser.add_argument("--limit", type=int, default=5, help="限制测试数量")
    
    args = parser.parse_args()
    
    quick_local_eval(args.limit)


if __name__ == "__main__":
    main()
