#!/usr/bin/env python3
"""
RAG系统快速测评脚本 - 本地模型版本
"""
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings
from services.rag_generator import rag_generator
from models import RetrievalConfig, GenerationConfig

def quick_rag_eval():
    """快速RAG测评"""
    print("🚀 RAG系统快速测评（本地模型）")
    print("="*60)
    print(f"配置: {settings.llm_provider} - {settings.llm_model}")
    print(f"嵌入: {settings.embedding_model_name} ({settings.embedding_device})")
    print(f"向量库: {settings.vector_db_type}")
    
    # 测试用例
    test_cases = [
        "差旅费标准是什么？",
        "8-9级员工住宿标准", 
        "报销需要什么发票？",
        "餐补标准是多少？",
        "总盟能住什么酒店？",
        "北京和上海的住宿标准有什么区别？",
        "经理能坐飞机商务舱吗？"
    ]
    
    results = []
    
    print(f"\n🧪 运行快速测评 ({len(test_cases)} 条):")
    
    for i, query in enumerate(test_cases, 1):
        print(f"[{i:2d}/{len(test_cases)}] {query}")
        
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
            
            # 简单的自动评估
            answer = response.answer
            answer_lower = answer.lower()
            
            # 关键词命中评估
            if "差旅费" in query and "标准" in answer:
                keyword_score = 1.0
            elif "住宿" in query and any(kw in answer.lower() for kw in ["标准", "酒店", "宾馆"]):
                keyword_score = 0.8
            elif "报销" in query and any(kw in answer.lower() for kw in ["发票", "流程", "审批"]):
                keyword_score = 0.9
            elif "餐补" in query and any(kw in answer.lower() for kw in ["餐补", "补贴", "餐饮"]):
                keyword_score = 0.9
            else:
                keyword_score = 0.3
            
            # 性能评估
            if elapsed < 3000 and len(answer) > 50:
                performance_score = 1.0
            elif elapsed < 5000:
                performance_score = 0.8
            else:
                performance_score = 0.6
            
            # 综合评分
            score = (keyword_score * 0.6 + performance_score * 0.4)
            
            status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
            
            print(f"   {status} {elapsed:.0f}ms | 评分:{score:.2f} | 回答:{len(answer)}字符")
            print(f"      {answer[:50]}...")
            
            results.append({
                "query": query,
                "score": score,
                "elapsed_ms": elapsed,
                "answer_length": len(answer),
                "status": status
            })
            
        except Exception as e:
            print(f"   ❌ 失败: {str(e)[:50]}")
            results.append({
                "query": query,
                "score": 0,
                "elapsed_ms": 0,
                "answer_length": 0,
                "status": "❌",
            })
    
    # 统计分析
    valid_results = [r for r in results if r["status"] != "❌"]
    if valid_results:
        avg_score = statistics.mean([r["score"] for r in valid_results])
        success_rate = len(valid_results) / len(test_cases)
        avg_time = statistics.mean([r["elapsed_ms"] for r in valid_results])
        avg_length = statistics.mean([r["answer_length"] for r in valid_results])
        
        print(f"\n📊 测评结果:")
        print(f"   成功率: {success_rate:.1% ({len(valid_results)}/{len(test_cases)})")
        print(f"   平均评分: {avg_score:.2f}/1.0")
        print(f"   平均响应时间: {avg_time:.1f}ms")
        print(f"   平均回答长度: {avg_length:.0f}字符")
        
        if avg_score >= 0.8:
            print("   🟢 优秀 - 本地RAG系统表现卓越")
        elif avg_score >= 0.6:
            print("   🟡 �好 - 本地RAG系统可用")
        else:
            print("   🟠 需优化 - 本地RAG系统需要改进")
            
        print("="*60)
        print("💡 完整测评:")
        print("   python local_eval.py --limit 10")
        print("   或使用:")
        print(f"   python -m tests.evaluation.local_eval --limit 5")
        
        return results


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="RAG系统快速测评")
    parser.add_argument("--limit", type=int, default=5, help="限制测试数量")
    
    args = parser.parse_args()
    
    quick_rag_eval()


if __name__ == "__main__":
    main()