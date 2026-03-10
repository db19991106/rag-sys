#!/usr/bin/env python3
"""最小化评估脚本 - 使用现有向量数据库"""

import json
import sys
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 延迟导入
def run_evaluation():
    import faiss
    from services.embedding import EmbeddingService
    from models import EmbeddingConfig, EmbeddingModelType
    from config import settings
    
    print("="*60)
    print("加载嵌入模型...")
    print("="*60)
    
    embedding_service = EmbeddingService()
    config = EmbeddingConfig(
        model_name=settings.embedding_model_name,
        model_path=settings.embedding_model_name,
        model_type=EmbeddingModelType.BGE,
        device="cuda"
    )
    embedding_service.load_model(config)
    print(f"嵌入模型加载完成，维度: {embedding_service.get_dimension()}")
    
    # 加载向量索引
    print("\n" + "="*60)
    print("加载向量索引...")
    print("="*60)
    
    output_dir = PROJECT_ROOT / "tests" / "chunking_comparison" / "comparison_output"
    
    # 分层切分索引
    layered_index = faiss.read_index(str(output_dir / "vector_db_layered" / "index.faiss"))
    with open(output_dir / "vector_db_layered" / "metadata.json", 'r') as f:
        layered_metadata = json.load(f)
    print(f"分层切分索引: {layered_index.ntotal} 个向量")
    
    # 分隔符切分索引
    naive_index = faiss.read_index(str(output_dir / "vector_db_naive" / "index.faiss"))
    with open(output_dir / "vector_db_naive" / "metadata.json", 'r') as f:
        naive_metadata = json.load(f)
    print(f"分隔符切分索引: {naive_index.ntotal} 个向量")
    
    # 加载测试数据
    print("\n" + "="*60)
    print("加载测试数据...")
    print("="*60)
    
    test_data_path = PROJECT_ROOT / "tests" / "chunking_comparison" / "ragas_eval_dataset.json"
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    samples = test_data['samples'][:30]  # 只取30个样本
    print(f"加载了 {len(samples)} 个测试样本")
    
    # 评估函数
    def search(query, index, metadata, top_k=5):
        query_emb = embedding_service.encode([query])
        query_emb = np.array(query_emb).astype('float32')
        distances, indices = index.search(query_emb, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            idx_str = str(idx)
            if idx_str in metadata:
                result = metadata[idx_str].copy()
                result['score'] = float(distances[0][i])
                results.append(result)
        return results
    
    def calc_keyword_metrics(contexts, keywords):
        if not keywords:
            return {"recall": 0.0, "precision": 0.0}
        all_text = " ".join(contexts).lower()
        hits = sum(1 for k in keywords if k.lower() in all_text)
        recall = hits / len(keywords)
        precision = min(1.0, hits / 5)  # 简化计算
        return {"recall": recall, "precision": precision}
    
    def calc_similarity(query, contexts):
        query_emb = np.array(embedding_service.encode([query])).astype('float32')
        ctx_text = " ".join(contexts[:3])
        ctx_emb = np.array(embedding_service.encode([ctx_text])).astype('float32')
        sim = np.dot(query_emb[0], ctx_emb[0]) / (np.linalg.norm(query_emb[0]) * np.linalg.norm(ctx_emb[0]))
        return float(sim)
    
    # 运行评估
    print("\n" + "="*60)
    print("运行评估...")
    print("="*60)
    
    results = {
        "layered": {"samples": [], "stats": defaultdict(list)},
        "naive": {"samples": [], "stats": defaultdict(list)}
    }
    
    for i, sample in enumerate(samples):
        query = sample['question']
        keywords = sample.get('metadata', {}).get('expected_keywords', [])
        
        if (i + 1) % 5 == 0:
            print(f"进度: {i+1}/{len(samples)}")
        
        # 分层切分
        layered_ctxs = search(query, layered_index, layered_metadata)
        layered_texts = [c['content'] for c in layered_ctxs]
        layered_kw = calc_keyword_metrics(layered_texts, keywords)
        layered_sim = calc_similarity(query, layered_texts)
        
        results["layered"]["stats"]["keyword_recall"].append(layered_kw["recall"])
        results["layered"]["stats"]["keyword_precision"].append(layered_kw["precision"])
        results["layered"]["stats"]["similarity"].append(layered_sim)
        
        # 分隔符切分
        naive_ctxs = search(query, naive_index, naive_metadata)
        naive_texts = [c['content'] for c in naive_ctxs]
        naive_kw = calc_keyword_metrics(naive_texts, keywords)
        naive_sim = calc_similarity(query, naive_texts)
        
        results["naive"]["stats"]["keyword_recall"].append(naive_kw["recall"])
        results["naive"]["stats"]["keyword_precision"].append(naive_kw["precision"])
        results["naive"]["stats"]["similarity"].append(naive_sim)
    
    # 计算统计
    print("\n" + "="*60)
    print("计算统计结果...")
    print("="*60)
    
    final_stats = {}
    for method in ["layered", "naive"]:
        final_stats[method] = {}
        for metric, values in results[method]["stats"].items():
            final_stats[method][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    
    print(f"\n{'指标':<20} {'分层切分':>15} {'分隔符切分':>15} {'差异':>10}")
    print("-"*60)
    
    for metric in ["keyword_recall", "keyword_precision", "similarity"]:
        l_val = final_stats["layered"][metric]["mean"]
        n_val = final_stats["naive"][metric]["mean"]
        diff = l_val - n_val
        print(f"{metric:<20} {l_val:>15.4f} {n_val:>15.4f} {diff:>+10.4f}")
    
    # 生成报告
    report = f"""# 文档切分方法对比测试报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、测试配置

| 参数 | 值 |
|------|------|
| 测试样本数 | {len(samples)} |
| 分层切分 chunks | {layered_index.ntotal} |
| 分隔符切分 chunks | {naive_index.ntotal} |
| 检索 top_k | 5 |

## 二、总体指标对比

| 指标 | 分层切分 | 分隔符切分 | 差异 | 说明 |
|------|----------|------------|------|------|
| 关键词召回率 | {final_stats['layered']['keyword_recall']['mean']:.4f} | {final_stats['naive']['keyword_recall']['mean']:.4f} | {final_stats['layered']['keyword_recall']['mean'] - final_stats['naive']['keyword_recall']['mean']:+.4f} | 检索内容覆盖关键词的比例 |
| 关键词精确率 | {final_stats['layered']['keyword_precision']['mean']:.4f} | {final_stats['naive']['keyword_precision']['mean']:.4f} | {final_stats['layered']['keyword_precision']['mean'] - final_stats['naive']['keyword_precision']['mean']:+.4f} | 检索内容的相关性 |
| 上下文相似度 | {final_stats['layered']['similarity']['mean']:.4f} | {final_stats['naive']['similarity']['mean']:.4f} | {final_stats['layered']['similarity']['mean'] - final_stats['naive']['similarity']['mean']:+.4f} | 查询与检索内容的语义相似度 |

## 三、详细统计数据

### 3.1 分层智能切分

| 指标 | 均值 | 标准差 |
|------|------|--------|
| 关键词召回率 | {final_stats['layered']['keyword_recall']['mean']:.4f} | {final_stats['layered']['keyword_recall']['std']:.4f} |
| 关键词精确率 | {final_stats['layered']['keyword_precision']['mean']:.4f} | {final_stats['layered']['keyword_precision']['std']:.4f} |
| 上下文相似度 | {final_stats['layered']['similarity']['mean']:.4f} | {final_stats['layered']['similarity']['std']:.4f} |

### 3.2 分隔符切分

| 指标 | 均值 | 标准差 |
|------|------|--------|
| 关键词召回率 | {final_stats['naive']['keyword_recall']['mean']:.4f} | {final_stats['naive']['keyword_recall']['std']:.4f} |
| 关键词精确率 | {final_stats['naive']['keyword_precision']['mean']:.4f} | {final_stats['naive']['keyword_precision']['std']:.4f} |
| 上下文相似度 | {final_stats['naive']['similarity']['mean']:.4f} | {final_stats['naive']['similarity']['std']:.4f} |

## 四、性能差异分析

### 4.1 各指标差异对比

| 指标 | 更优方法 | 差异幅度 |
|------|----------|----------|
"""
    
    for metric, name in [("keyword_recall", "关键词召回率"), ("keyword_precision", "关键词精确率"), ("similarity", "上下文相似度")]:
        l_val = final_stats["layered"][metric]["mean"]
        n_val = final_stats["naive"][metric]["mean"]
        diff = l_val - n_val
        winner = "分层切分" if diff > 0 else ("分隔符切分" if diff < 0 else "持平")
        report += f"| {name} | {winner} | {diff:+.4f} |\n"
    
    report += f"""
### 4.2 分析说明

1. **关键词召回率**
   - 反映检索内容是否覆盖了预期的关键信息
   - 较高的召回率意味着检索结果更全面

2. **关键词精确率**
   - 反映检索内容的相关性质量
   - 较高的精确率意味着检索结果更精准

3. **上下文相似度**
   - 反映查询与检索内容的语义匹配程度
   - 使用BGE-M3模型计算向量相似度

## 五、方法选择建议

### 5.1 综合评估

"""
    
    l_total = final_stats["layered"]["keyword_recall"]["mean"] + final_stats["layered"]["keyword_precision"]["mean"] + final_stats["layered"]["similarity"]["mean"]
    n_total = final_stats["naive"]["keyword_recall"]["mean"] + final_stats["naive"]["keyword_precision"]["mean"] + final_stats["naive"]["similarity"]["mean"]
    
    if l_total > n_total:
        report += "分层智能切分在本次测试中整体表现更优，建议优先使用。\n"
    else:
        report += "分隔符切分在本次测试中整体表现更优，建议优先使用。\n"
    
    report += f"""
### 5.2 场景选择建议

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 结构化文档 | 分层切分 | 保留章节结构，上下文清晰 |
| 短文本/碎片化文档 | 分隔符切分 | 简单高效，chunk数量少 |
| 需要表格完整性 | 分层切分 | 保护表格不拆分 |
| 跨章节关联查询 | 分隔符切分 | 无章节边界限制 |

---

*报告由 RAG 系统切分对比测试工具自动生成*
"""
    
    # 保存报告
    report_path = PROJECT_ROOT.parent / "test_reports" / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存: {report_path}")
    
    # 保存详细结果
    results_path = output_dir / "detailed_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"详细结果已保存: {results_path}")
    
    return results


if __name__ == "__main__":
    run_evaluation()
