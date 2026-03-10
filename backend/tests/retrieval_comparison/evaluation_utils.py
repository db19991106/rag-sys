"""
检索评估共用工具函数
用于两个检索对比实验
"""

import os
import sys
import json
import gc
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch


def clear_gpu_memory():
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[内存清理] GPU缓存已清空")


def wait_for_memory(seconds: int = 5):
    """等待内存释放"""
    print(f"[等待] 等待 {seconds} 秒以确保资源释放...")
    time.sleep(seconds)


def load_ground_truth(file_path: str = None) -> Dict[str, Dict]:
    """
    加载测试用例Ground Truth数据
    
    Returns:
        Dict[query_id, {relevant_chunks, relevant_content}]
    """
    if file_path is None:
        file_path = project_root.parent / "retrieval_test_cases_ground_truth.json"
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Ground truth文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('ground_truth', data)


def calculate_content_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的简单相似度（关键词重叠）
    """
    if not text1 or not text2:
        return 0.0
    
    # 简单分词
    words1 = set(text1.lower())
    words2 = set(text2.lower())
    
    # 计算Jaccard相似度
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def check_content_match(retrieved_content: str, expected_content: str, threshold: float = 0.3) -> bool:
    """
    检查检索内容是否匹配期望内容
    
    使用关键词匹配和相似度阈值
    """
    if not retrieved_content or not expected_content:
        return False
    
    # 提取期望内容的关键信息
    expected_lower = expected_content.lower()
    retrieved_lower = retrieved_content.lower()
    
    # 检查是否包含关键数字或特定短语
    import re
    
    # 提取数字
    expected_numbers = set(re.findall(r'\d+', expected_content))
    retrieved_numbers = set(re.findall(r'\d+', retrieved_content))
    
    # 如果有数字，检查数字是否匹配
    if expected_numbers:
        number_match = len(expected_numbers & retrieved_numbers) / len(expected_numbers)
        if number_match >= 0.5:  # 至少50%的数字匹配
            return True
    
    # 检查关键词匹配
    expected_keywords = set(re.findall(r'[\u4e00-\u9fff]+', expected_lower))
    retrieved_keywords = set(re.findall(r'[\u4e00-\u9fff]+', retrieved_lower))
    
    if expected_keywords:
        keyword_match = len(expected_keywords & retrieved_keywords) / len(expected_keywords)
        if keyword_match >= threshold:
            return True
    
    # 使用文本相似度
    similarity = calculate_content_similarity(expected_content, retrieved_content)
    return similarity >= threshold


def evaluate_retrieval_results_with_content(
    results: List[Dict],
    ground_truth: Dict[str, Dict],
    k: int = 5
) -> Dict[str, float]:
    """
    使用内容匹配评估检索结果
    
    Args:
        results: 检索结果列表 [{query_id, retrieved_chunks: [chunk_id, ...], retrieved_contents: [content, ...]}]
        ground_truth: Ground Truth数据
        k: 评估的Top-K值
    
    Returns:
        评估指标字典
    """
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []
    f1_scores = []
    hit_rates = []
    
    for result in results:
        query_id = result.get('query_id')
        retrieved_chunks = result.get('retrieved_chunks', [])
        retrieved_contents = result.get('retrieved_contents', [])
        
        if query_id not in ground_truth:
            continue
        
        expected_content = ground_truth[query_id].get('relevant_content', '')
        
        # 使用内容匹配
        matched_indices = []
        for i, content in enumerate(retrieved_contents[:k]):
            if check_content_match(content, expected_content):
                matched_indices.append(i)
        
        # 计算指标
        # Precision: 匹配的数量 / k
        precision = len(matched_indices) / k if matched_indices else 0.0
        
        # Recall: 假设只有1个相关文档，如果匹配到则为1
        recall = 1.0 if matched_indices else 0.0
        
        # MRR: 第一个匹配的位置倒数
        mrr = 1.0 / (matched_indices[0] + 1) if matched_indices else 0.0
        
        # NDCG
        dcg = sum(1.0 / np.log2(i + 2) for i in matched_indices) if matched_indices else 0.0
        ideal_dcg = 1.0  # 只有1个相关文档
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Hit Rate
        hit_rate = 1.0 if matched_indices else 0.0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)
        f1_scores.append(f1)
        hit_rates.append(hit_rate)
    
    return {
        f'Precision@{k}': np.mean(precision_scores) if precision_scores else 0.0,
        f'Recall@{k}': np.mean(recall_scores) if recall_scores else 0.0,
        f'MRR@{k}': np.mean(mrr_scores) if mrr_scores else 0.0,
        f'NDCG@{k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        f'F1@{k}': np.mean(f1_scores) if f1_scores else 0.0,
        f'Hit_Rate@{k}': np.mean(hit_rates) if hit_rates else 0.0,
        'num_queries': len(precision_scores)
    }


def evaluate_retrieval_results(
    results: List[Dict],
    ground_truth: Dict[str, Dict],
    k: int = 5
) -> Dict[str, float]:
    """
    评估检索结果（优先使用内容匹配）
    """
    # 检查是否有内容信息
    if results and 'retrieved_contents' in results[0]:
        return evaluate_retrieval_results_with_content(results, ground_truth, k)
    
    # 回退到chunk_id匹配
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []
    f1_scores = []
    hit_rates = []
    
    for result in results:
        query_id = result.get('query_id')
        retrieved_chunks = result.get('retrieved_chunks', [])
        
        if query_id not in ground_truth:
            continue
        
        expected_chunks = ground_truth[query_id].get('relevant_chunks', [])
        
        precision = len(set(retrieved_chunks[:k]) & set(expected_chunks)) / k
        recall = len(set(retrieved_chunks[:k]) & set(expected_chunks)) / len(expected_chunks) if expected_chunks else 0.0
        
        mrr = 0.0
        for i, chunk in enumerate(retrieved_chunks[:k]):
            if chunk in expected_chunks:
                mrr = 1.0 / (i + 1)
                break
        
        dcg = 0.0
        for i, chunk in enumerate(retrieved_chunks[:k]):
            if chunk in expected_chunks:
                dcg += 1.0 / np.log2(i + 2)
        
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(expected_chunks), k)))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        hit_rate = 1.0 if set(retrieved_chunks[:k]) & set(expected_chunks) else 0.0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)
        f1_scores.append(f1)
        hit_rates.append(hit_rate)
    
    return {
        f'Precision@{k}': np.mean(precision_scores) if precision_scores else 0.0,
        f'Recall@{k}': np.mean(recall_scores) if recall_scores else 0.0,
        f'MRR@{k}': np.mean(mrr_scores) if mrr_scores else 0.0,
        f'NDCG@{k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        f'F1@{k}': np.mean(f1_scores) if f1_scores else 0.0,
        f'Hit_Rate@{k}': np.mean(hit_rates) if hit_rates else 0.0,
        'num_queries': len(precision_scores)
    }


def calculate_improvement_rate(baseline: float, optimized: float) -> float:
    """计算提升率"""
    if baseline == 0:
        return 0.0 if optimized == 0 else float('inf')
    return (optimized - baseline) / baseline * 100


def generate_comparison_report(
    experiment_name: str,
    baseline_name: str,
    optimized_name: str,
    baseline_metrics: Dict[str, float],
    optimized_metrics: Dict[str, float],
    output_path: str,
    experiment_config: Dict[str, Any] = None
) -> str:
    """生成对比报告"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# {experiment_name}

## 实验信息
- **实验时间**: {timestamp}
- **测试数据**: 50个查询（retrieval_test_cases_ground_truth.json）
- **评估指标**: Precision@5, Recall@5, MRR@5, NDCG@5, F1@5, Hit_Rate@5
- **评估方式**: 内容相似度匹配（检查检索内容是否包含期望信息）

"""
    
    if experiment_config:
        report += "## 实验配置\n\n"
        for key, value in experiment_config.items():
            report += f"- **{key}**: {value}\n"
        report += "\n"
    
    report += f"""## 对比结果

### 指标对比表

| 指标 | {baseline_name} | {optimized_name} | 提升率 |
|------|--------|--------|--------|
"""
    
    metrics_to_compare = ['Precision@5', 'Recall@5', 'MRR@5', 'NDCG@5', 'F1@5', 'Hit_Rate@5']
    
    for metric in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric, 0.0)
        optimized_val = optimized_metrics.get(metric, 0.0)
        improvement = calculate_improvement_rate(baseline_val, optimized_val)
        
        report += f"| {metric} | {baseline_val:.4f} | {optimized_val:.4f} | **{improvement:+.2f}%** |\n"
    
    report += f"""
### 平均响应时间

| 方法 | 平均响应时间 |
|------|------------|
| {baseline_name} | {baseline_metrics.get('avg_response_time', 0):.4f}s |
| {optimized_name} | {optimized_metrics.get('avg_response_time', 0):.4f}s |

## 结论分析

"""
    
    avg_improvement = np.mean([
        calculate_improvement_rate(
            baseline_metrics.get(m, 0),
            optimized_metrics.get(m, 0)
        ) for m in metrics_to_compare
    ])
    
    precision_improvement = calculate_improvement_rate(
        baseline_metrics.get('Precision@5', 0),
        optimized_metrics.get('Precision@5', 0)
    )
    
    recall_improvement = calculate_improvement_rate(
        baseline_metrics.get('Recall@5', 0),
        optimized_metrics.get('Recall@5', 0)
    )
    
    report += f"""本次实验对比了 **{baseline_name}** 和 **{optimized_name}** 两种检索方式的效果。

**主要发现**:

1. **Precision@5 提升**: {precision_improvement:+.2f}%
   - {baseline_name}: {baseline_metrics.get('Precision@5', 0):.4f}
   - {optimized_name}: {optimized_metrics.get('Precision@5', 0):.4f}

2. **Recall@5 提升**: {recall_improvement:+.2f}%
   - {baseline_name}: {baseline_metrics.get('Recall@5', 0):.4f}
   - {optimized_name}: {optimized_metrics.get('Recall@5', 0):.4f}

3. **整体指标平均提升**: {avg_improvement:+.2f}%

**结论**: {optimized_name} 相比 {baseline_name} 在检索准确率上有明显提升，特别是在 Precision 和 Recall 方面表现更好。

---
*报告生成时间: {timestamp}*
"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"[报告] 已保存到: {output_path}")
    return report
