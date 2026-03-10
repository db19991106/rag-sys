"""
RAGAS 风格评估脚本
评估指标：精确度、召回率、忠实性、答案相关性
使用本地LLM进行评估
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("RAG 风格评估（忠实性、答案相关性、精确度、召回率）")
print("=" * 70)

# 导入项目模块
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from models import EmbeddingConfig, EmbeddingModelType, VectorDBConfig, VectorDBType
from config import settings


def load_llm():
    """加载本地LLM"""
    print("\n[1/5] 加载本地LLM...")
    
    model_path = settings.local_llm_model_path
    print(f"  模型路径: {model_path}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  设备: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            model = model.to("cpu")
        
        print(f"  ✅ LLM加载成功")
        return {"model": model, "tokenizer": tokenizer, "device": device}
    
    except Exception as e:
        print(f"  ⚠️ 主模型加载失败: {e}")
        print("  尝试使用轻量级模型...")
        
        try:
            model_path = settings.coref_llm_model_path
            print(f"  轻量级模型路径: {model_path}")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            print(f"  ✅ 轻量级LLM加载成功")
            return {"model": model, "tokenizer": tokenizer, "device": device}
        
        except Exception as e2:
            print(f"  ❌ 无法加载LLM: {e2}")
            return None


def generate_answer(llm_info, query: str, contexts: List[str]) -> str:
    """使用LLM生成答案"""
    model = llm_info["model"]
    tokenizer = llm_info["tokenizer"]
    device = llm_info["device"]
    
    context_text = "\n\n".join(contexts[:3])
    
    prompt = f"""根据以下参考信息回答问题。请直接给出简洁准确的答案。

参考信息：
{context_text[:1500]}

问题：{query}

答案："""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt in response:
            response = response[len(prompt):].strip()
        
        return response
    
    except Exception as e:
        return f"生成失败: {str(e)}"


def evaluate_faithfulness(llm_info, answer: str, contexts: List[str]) -> float:
    """评估忠实性：答案是否基于上下文"""
    if not contexts or not answer:
        return 0.5
    
    model = llm_info["model"]
    tokenizer = llm_info["tokenizer"]
    device = llm_info["device"]
    
    context_text = " ".join(contexts[:2])[:800]
    
    prompt = f"""判断以下答案的内容是否完全来自给定的上下文，没有编造信息。
只回答"是"或"否"。

上下文：{context_text}

答案：{answer[:300]}

答案是否完全基于上下文？"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        
        if "是" in response:
            return 1.0
        elif "否" in response:
            return 0.0
        return 0.5
    except:
        return 0.5


def evaluate_answer_relevancy(llm_info, question: str, answer: str) -> float:
    """评估答案相关性"""
    if not question or not answer:
        return 0.5
    
    model = llm_info["model"]
    tokenizer = llm_info["tokenizer"]
    device = llm_info["device"]
    
    prompt = f"""评估以下答案与问题的相关程度。
评分标准：
- 3分：完全相关，直接回答了问题
- 2分：部分相关，包含相关信息但不完整
- 1分：不相关或答非所问

只回答分数（1/2/3）。

问题：{question}

答案：{answer[:400]}

评分："""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "3" in response:
            return 1.0
        elif "2" in response:
            return 0.66
        elif "1" in response:
            return 0.33
        return 0.5
    except:
        return 0.5


def evaluate_context_precision(query: str, contexts: List[str], reference: str) -> float:
    """评估上下文精确度：检索结果的相关性"""
    if not contexts:
        return 0.0
    
    # 使用关键词重叠来评估
    ref_keywords = set(reference.split()[:30]) if reference else set()
    
    scores = []
    for i, ctx in enumerate(contexts[:5]):
        ctx_keywords = set(ctx.split()[:50])
        overlap = len(ref_keywords & ctx_keywords) / max(len(ref_keywords), 1)
        position_weight = 1.0 / (i + 1)
        scores.append(overlap * position_weight)
    
    return min(1.0, sum(scores))


def evaluate_context_recall(query: str, contexts: List[str], reference: str) -> float:
    """评估上下文召回率：是否检索到所有相关信息"""
    if not reference or not contexts:
        return 0.5
    
    ref_keywords = set(reference.split()[:30])
    all_ctx_keywords = set()
    for ctx in contexts:
        all_ctx_keywords.update(ctx.split()[:50])
    
    overlap = len(ref_keywords & all_ctx_keywords)
    recall = overlap / max(len(ref_keywords), 1)
    
    return min(1.0, recall)


def main():
    # 1. 加载LLM
    llm_info = load_llm()
    if llm_info is None:
        print("❌ 无法加载LLM，评估终止")
        return
    
    # 2. 加载嵌入模型和向量数据库
    print("\n[2/5] 加载嵌入模型和向量数据库...")
    if not embedding_service.is_loaded():
        embedding_service.load_model(
            EmbeddingConfig(
                model_type=EmbeddingModelType.BGE,
                model_name=settings.embedding_model_name,
                device="cpu",
            )
        )
    print(f"  ✅ 嵌入模型: {settings.embedding_model_name}")
    
    vector_db_manager.initialize(
        VectorDBConfig(
            db_type=VectorDBType.MILVUS_LITE,
            dimension=embedding_service.get_dimension(),
        )
    )
    status = vector_db_manager.get_status()
    print(f"  ✅ 向量数: {status.total_vectors}")
    
    # 3. 加载测试数据
    print("\n[3/5] 加载测试数据...")
    with open("/root/autodl-tmp/rag/retrieval_test_cases_ground_truth_v2.json", 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    ground_truth = gt_data.get('ground_truth', gt_data)
    print(f"  测试用例: {len(ground_truth)}")
    
    # 4. 运行评估
    print("\n[4/5] 运行评估...")
    
    all_metrics = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
        "latency_ms": []
    }
    
    evaluated_samples = []
    
    for i, (query_id, query_data) in enumerate(list(ground_truth.items())):
        query = query_data.get('query', '')
        reference = query_data.get('relevant_content', '')
        
        if not query:
            continue
        
        try:
            start_time = time.time()
            
            # 检索
            query_vector = embedding_service.encode([query])
            scores, metadatas = vector_db_manager.search(query_vector, top_k=5)
            contexts = [m.get('content', '') for m in metadatas[0]]
            
            # 生成答案
            answer = generate_answer(llm_info, query, contexts)
            
            # 评估指标
            faithfulness = evaluate_faithfulness(llm_info, answer, contexts)
            relevancy = evaluate_answer_relevancy(llm_info, query, answer)
            precision = evaluate_context_precision(query, contexts, reference)
            recall = evaluate_context_recall(query, contexts, reference)
            
            elapsed = (time.time() - start_time) * 1000
            
            all_metrics["faithfulness"].append(faithfulness)
            all_metrics["answer_relevancy"].append(relevancy)
            all_metrics["context_precision"].append(precision)
            all_metrics["context_recall"].append(recall)
            all_metrics["latency_ms"].append(elapsed)
            
            evaluated_samples.append({
                "query_id": query_id,
                "query": query[:100],
                "answer": answer[:200],
                "metrics": {
                    "faithfulness": faithfulness,
                    "answer_relevancy": relevancy,
                    "context_precision": precision,
                    "context_recall": recall
                }
            })
            
            if (i + 1) % 20 == 0:
                print(f"    进度: {i+1}/{len(ground_truth)}")
        
        except Exception as e:
            print(f"    ⚠️ 样本 {query_id} 评估失败: {e}")
    
    # 5. 计算结果
    print("\n[5/5] 生成评估报告...")
    
    import numpy as np
    
    metrics = {
        "faithfulness": round(np.mean(all_metrics["faithfulness"]), 4),
        "answer_relevancy": round(np.mean(all_metrics["answer_relevancy"]), 4),
        "context_precision": round(np.mean(all_metrics["context_precision"]), 4),
        "context_recall": round(np.mean(all_metrics["context_recall"]), 4),
        "avg_latency_ms": round(np.mean(all_metrics["latency_ms"]), 2)
    }
    
    overall_score = (
        metrics['faithfulness'] * 0.25 +
        metrics['answer_relevancy'] * 0.25 +
        metrics['context_precision'] * 0.25 +
        metrics['context_recall'] * 0.25
    )
    
    results = {
        "evaluation_time": datetime.now().isoformat(),
        "config": {
            "llm": settings.llm_model,
            "embedding_model": settings.embedding_model_name,
            "vector_db": "Milvus Lite",
            "test_cases": len(evaluated_samples)
        },
        "metrics": metrics,
        "overall_score": round(overall_score * 100, 2),
        "grade": "优秀" if overall_score >= 0.8 else "良好" if overall_score >= 0.6 else "一般" if overall_score >= 0.4 else "需改进"
    }
    
    # 保存结果
    output_dir = Path("/root/autodl-tmp/rag/test_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "ragas_style_evaluation_report.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存详细样本
    with open(output_dir / "ragas_evaluation_samples.json", 'w', encoding='utf-8') as f:
        json.dump(evaluated_samples, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print("\n" + "=" * 70)
    print("RAGAS 风格评估结果")
    print("=" * 70)
    print(f"\n【配置信息】")
    print(f"  LLM: {settings.llm_model}")
    print(f"  嵌入模型: {settings.embedding_model_name}")
    print(f"  向量数据库: Milvus Lite ({status.total_vectors} 向量)")
    print(f"  测试用例: {len(evaluated_samples)}")
    
    print(f"\n【RAGAS 指标】")
    print(f"  忠实性 (Faithfulness):     {metrics['faithfulness']:.4f} (权重25%)")
    print(f"  答案相关性 (Relevancy):    {metrics['answer_relevancy']:.4f} (权重25%)")
    print(f"  上下文精确度 (Precision):  {metrics['context_precision']:.4f} (权重25%)")
    print(f"  上下文召回率 (Recall):     {metrics['context_recall']:.4f} (权重25%)")
    print(f"  平均延迟: {metrics['avg_latency_ms']:.2f}ms")
    
    print(f"\n【综合评分】")
    print(f"  得分: {results['overall_score']}/100")
    print(f"  评级: {results['grade']}")
    
    print(f"\n【结果已保存】")
    print(f"  {output_dir / 'ragas_style_evaluation_report.json'}")


if __name__ == "__main__":
    main()