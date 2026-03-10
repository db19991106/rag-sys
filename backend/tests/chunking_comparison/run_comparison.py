#!/usr/bin/env python3
"""
分层智能切分 vs 分隔符切分 对比测试脚本

对比两种切分方式在问答召回率上的表现：
- LAYERED: 三层递进式智能切分
- NAIVE: 基于分隔符的朴素切分

评估指标：Recall@k, Precision@k, MRR, NDCG, LLM评估(Precision/Recall/F1)
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


class LLMEvaluator:
    """LLM评估器 - 使用LLM对比检索结果与真值答案"""
    
    def __init__(self):
        self.llm_client = None
    
    def _load_llm(self):
        """加载LLM模型"""
        if self.llm_client is not None:
            return
        
        from services.rag_generator import LocalLLMClient
        from models import GenerationConfig
        from config import settings
        
        config = GenerationConfig(
            llm_provider="local",
            local_model_path=settings.local_llm_model_path,
            device=settings.local_llm_device
        )
        self.llm_client = LocalLLMClient(config)
    
    def evaluate_retrieval_quality(
        self,
        query: str,
        ground_truth: str,
        retrieved_chunks: List[str]
    ) -> Dict[str, float]:
        """
        使用LLM评估检索质量
        
        Args:
            query: 用户查询
            ground_truth: 真值答案
            retrieved_chunks: 检索到的文本片段列表
            
        Returns:
            包含precision, recall, f1_score的字典
        """
        self._load_llm()
        
        # 构建检索内容
        retrieved_content = "\n\n---\n\n".join([f"[片段{i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks[:5])])
        
        # 构建评估提示词
        prompt = f"""你是一个RAG系统评估专家。请评估检索结果与真值答案的匹配程度。

【用户查询】
{query}

【真值答案】
{ground_truth}

【检索结果】
{retrieved_content}

请从以下维度评估检索质量，并给出0-1的分数：

1. **精确率(Precision)**: 检索结果中有多少信息与真值答案相关且正确？（0-1）
2. **召回率(Recall)**: 真值答案中的关键信息有多少被检索结果覆盖？（0-1）

请严格按以下JSON格式返回，不要返回其他内容：
{{"precision": 0.8, "recall": 0.9, "reasoning": "简要说明评分理由"}}"""

        try:
            response = self.llm_client.generate(prompt)
            
            # 解析JSON响应
            if isinstance(response, dict):
                response_text = response.get('text', str(response))
            else:
                response_text = str(response)
            
            # 提取JSON
            import re
            json_match = re.search(r'\{[^{}]*"precision"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                precision = float(result.get('precision', 0.5))
                recall = float(result.get('recall', 0.5))
            else:
                # 默认值
                precision = 0.5
                recall = 0.5
            
            # 计算F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'llm_precision': min(1.0, max(0.0, precision)),
                'llm_recall': min(1.0, max(0.0, recall)),
                'llm_f1': f1
            }
        except Exception as e:
            print(f"LLM评估失败: {e}")
            return {
                'llm_precision': 0.0,
                'llm_recall': 0.0,
                'llm_f1': 0.0
            }


class ChunkingComparisonTest:
    """切分方式对比测试"""
    
    def __init__(
        self,
        test_cases_path: str,
        docs_dir: str,
        output_dir: str,
        chunk_size: int = 512,
        overlap: float = 0.1,
        top_k: int = 10
    ):
        self.test_cases_path = Path(test_cases_path)
        self.docs_dir = Path(docs_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载嵌入模型
        self._load_embedding_model()
        
        # 加载测试用例
        with open(self.test_cases_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.test_cases = data['test_cases']
        self.metadata = data['metadata']
        
        print(f"加载了 {len(self.test_cases)} 个测试用例")
        print(f"切分参数: chunk_size={chunk_size}, overlap={overlap}, top_k={top_k}")
    
    def _load_embedding_model(self):
        """加载嵌入模型"""
        from services.embedding import embedding_service
        from models import EmbeddingConfig, EmbeddingModelType
        from config import settings
        
        print("加载嵌入模型...")
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.BGE,
            model_name=settings.embedding_model_name,
            device=settings.embedding_device
        )
        embedding_service.load_model(config)
        print(f"嵌入模型加载完成，维度: {embedding_service.get_dimension()}")
    
    def run_naive_chunking(self, content: str) -> List[str]:
        """执行分隔符切分"""
        import re
        
        chunks = []
        delimiter = "\n。；！？"
        
        # 按双换行分割段落
        paragraphs = re.split(r"\n\s*\n", content)
        
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 简单估算token数（中文约1.5字/token）
            para_tokens = len(para) // 1.5
            
            if current_tokens + para_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
                current_tokens = para_tokens
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                current_tokens += para_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def run_layered_chunking(self, content: str, doc_id: str = "test_doc") -> List[str]:
        """执行分层智能切分"""
        from services.layered_chunker import LayeredChunker, LayeredChunkConfig
        
        config = LayeredChunkConfig(
            max_chunk_size=self.chunk_size,
            overlap=int(self.chunk_size * self.overlap),
            min_chunk_size=80,
            preserve_table=True,
            preserve_flow=True,
            preserve_code=True
        )
        chunker = LayeredChunker(config=config)
        chunks = chunker.chunk(content, doc_id)
        return [c['content'] if isinstance(c, dict) else c for c in chunks]
    
    def build_vector_index(self, chunks: List[str], doc_name: str) -> Tuple[Any, Dict]:
        """构建向量索引"""
        from services.embedding import embedding_service
        
        # 生成嵌入向量
        embeddings = embedding_service.encode(chunks)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # 构建FAISS索引
        import faiss
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.add(embeddings)
        
        # 构建元数据
        metadata = {}
        for i, chunk in enumerate(chunks):
            metadata[str(i)] = {
                'content': chunk,
                'document_name': doc_name,
                'chunk_num': i
            }
        
        return index, metadata
    
    def search(self, query: str, index: Any, metadata: Dict, top_k: int = 10) -> List[Dict]:
        """执行向量检索"""
        from services.embedding import embedding_service
        
        # 生成查询向量
        query_embedding = embedding_service.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # 搜索
        distances, indices = index.search(query_embedding, top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and str(idx) in metadata:
                result = metadata[str(idx)].copy()
                result['similarity'] = float(1 - dist / 100)  # 转换为相似度
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def calculate_metrics(
        self,
        results: List[Dict],
        expected_docs: List[str],
        expected_keywords: List[str],
        top_k: int = 10
    ) -> Dict[str, float]:
        """计算评估指标 - 基于关键词匹配"""
        metrics = {}
        
        # 获取检索到的内容
        retrieved_contents = [r['content'] for r in results[:top_k]]
        all_content = ' '.join(retrieved_contents).lower()
        
        # 基于关键词计算指标
        if expected_keywords:
            # 计算每个关键词是否在检索结果中
            keyword_hits = []
            for kw in expected_keywords:
                hit = kw.lower() in all_content
                keyword_hits.append(hit)
            
            # Recall: 命中的关键词数 / 期望关键词总数
            total_expected = len(expected_keywords)
            total_hits = sum(keyword_hits)
            metrics[f'recall@{top_k}'] = total_hits / total_expected if total_expected > 0 else 0.0
            
            # Precision: 命中的关键词数 / 检索的关键词数（假设每个chunk最多包含一个期望关键词）
            metrics[f'precision@{top_k}'] = total_hits / (top_k) if top_k > 0 else 0.0
            
            # MRR: 第一个命中关键词的排名倒数
            for i, content in enumerate(retrieved_contents):
                content_lower = content.lower()
                if any(kw.lower() in content_lower for kw in expected_keywords):
                    metrics[f'mrr@{top_k}'] = 1.0 / (i + 1)
                    break
            else:
                metrics[f'mrr@{top_k}'] = 0.0
            
            # NDCG: 考虑关键词命中的位置
            dcg = 0.0
            for i, content in enumerate(retrieved_contents):
                content_lower = content.lower()
                # 计算该chunk包含多少期望关键词
                hits_in_chunk = sum(1 for kw in expected_keywords if kw.lower() in content_lower)
                if hits_in_chunk > 0:
                    dcg += hits_in_chunk / np.log2(i + 2)
            
            ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(expected_keywords), top_k)))
            metrics[f'ndcg@{top_k}'] = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            
            # 关键词命中率
            metrics['keyword_hit_rate'] = total_hits / total_expected if total_expected > 0 else 0.0
        else:
            metrics[f'recall@{top_k}'] = 0.0
            metrics[f'precision@{top_k}'] = 0.0
            metrics[f'mrr@{top_k}'] = 0.0
            metrics[f'ndcg@{top_k}'] = 0.0
            metrics['keyword_hit_rate'] = 0.0
        
        return metrics
    
    def run_single_test(
        self,
        test_case: Dict,
        index: Any,
        metadata: Dict,
        chunking_type: str,
        llm_evaluator: Optional[LLMEvaluator] = None
    ) -> Dict:
        """执行单个测试用例"""
        query = test_case['query']
        expected_docs = test_case.get('expected_docs', [])
        expected_keywords = test_case.get('expected_keywords', [])
        ground_truth = test_case.get('ground_truth_answer', '')
        
        # 执行检索
        start_time = time.time()
        results = self.search(query, index, metadata, self.top_k)
        search_time = time.time() - start_time
        
        # 计算关键词指标
        metrics = self.calculate_metrics(
            results, expected_docs, expected_keywords, self.top_k
        )
        
        # LLM评估（如果有真值答案）
        if llm_evaluator and ground_truth:
            retrieved_chunks = [r['content'] for r in results[:5]]
            llm_metrics = llm_evaluator.evaluate_retrieval_quality(
                query, ground_truth, retrieved_chunks
            )
            metrics.update(llm_metrics)
        
        return {
            'test_id': test_case['id'],
            'query': query,
            'scenario': test_case['scenario'],
            'chunking_type': chunking_type,
            'search_time_ms': search_time * 1000,
            'results': results[:5],  # 只保留前5个结果
            'metrics': metrics
        }
    
    def run_comparison(self, use_llm_eval: bool = True) -> Dict[str, Any]:
        """运行完整对比测试"""
        print("\n" + "="*80)
        print("开始切分方式对比测试")
        print("="*80)
        
        # 初始化LLM评估器
        llm_evaluator = None
        if use_llm_eval:
            try:
                print("初始化LLM评估器...")
                llm_evaluator = LLMEvaluator()
            except Exception as e:
                print(f"LLM评估器初始化失败: {e}，将跳过LLM评估")
                llm_evaluator = None
        
        # 加载测试文档
        test_docs = [
            '财务报销标准.md',
            '城市分类地区报销差异.md'
        ]
        
        all_results = {
            'layered': {'all': [], 'by_scenario': {}},
            'naive': {'all': [], 'by_scenario': {}}
        }
        
        for chunking_type in ['layered', 'naive']:
            print(f"\n{'='*40}")
            print(f"测试切分方式: {chunking_type.upper()}")
            print(f"{'='*40}")
            
            # 切分所有文档
            all_chunks = []
            chunk_to_doc = {}
            
            for doc_name in test_docs:
                doc_path = self.docs_dir / doc_name
                if not doc_path.exists():
                    print(f"警告: 文档不存在 {doc_path}")
                    continue
                
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 执行切分
                if chunking_type == 'layered':
                    chunks = self.run_layered_chunking(content, doc_name)
                else:
                    chunks = self.run_naive_chunking(content)
                
                print(f"  {doc_name}: {len(chunks)} 个片段")
                
                for chunk in chunks:
                    chunk_to_doc[len(all_chunks)] = doc_name
                    all_chunks.append(chunk)
            
            print(f"\n总片段数: {len(all_chunks)}")
            
            # 构建向量索引
            print("构建向量索引...")
            index, metadata = self.build_vector_index(all_chunks, 'combined')
            
            # 运行测试用例
            print(f"运行 {len(self.test_cases)} 个测试用例...")
            
            for i, test_case in enumerate(self.test_cases):
                if (i + 1) % 50 == 0:
                    print(f"  进度: {i+1}/{len(self.test_cases)}")
                
                result = self.run_single_test(test_case, index, metadata, chunking_type, llm_evaluator)
                all_results[chunking_type]['all'].append(result)
                
                # 按场景分组
                scenario = test_case['scenario']
                if scenario not in all_results[chunking_type]['by_scenario']:
                    all_results[chunking_type]['by_scenario'][scenario] = []
                all_results[chunking_type]['by_scenario'][scenario].append(result)
        
        return all_results
    
    def aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """聚合指标"""
        if not results:
            return {}
        
        metric_names = [k for k in results[0]['metrics'].keys()]
        aggregated = {}
        
        for name in metric_names:
            values = [r['metrics'][name] for r in results]
            aggregated[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return aggregated
    
    def generate_report(self, all_results: Dict) -> str:
        """生成 Markdown 对比报告"""
        report = []
        report.append("# 分层智能切分 vs 分隔符切分 对比测试报告\n")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 测试配置
        report.append("## 一、测试配置\n")
        report.append(f"| 参数 | 值 |")
        report.append(f"|------|------|")
        report.append(f"| 测试用例数 | {len(self.test_cases)} |")
        report.append(f"| chunk_size | {self.chunk_size} |")
        report.append(f"| overlap | {self.overlap} |")
        report.append(f"| top_k | {self.top_k} |")
        report.append("")
        
        # 场景分布
        report.append("### 测试场景分布\n")
        report.append("| 场景 | 数量 | 描述 |")
        report.append("|------|------|------|")
        scenario_desc = {
            'multi_doc_association': '多文档关联查询',
            'table_integrity': '表格完整性查询',
            'process_steps': '流程步骤查询',
            'clause_association': '条款关联查询',
            'single_fact': '单一事实查询'
        }
        for scenario, desc in scenario_desc.items():
            count = len([t for t in self.test_cases if t['scenario'] == scenario])
            report.append(f"| {desc} | {count} | {scenario} |")
        report.append("")
        
        # 总体对比
        report.append("## 二、总体指标对比\n")
        report.append("| 指标 | 分层切分 | 分隔符切分 | 差异 |")
        report.append("|------|----------|------------|------|")
        
        layered_metrics = self.aggregate_metrics(all_results['layered']['all'])
        naive_metrics = self.aggregate_metrics(all_results['naive']['all'])
        
        # 基础指标
        for metric_name in [f'recall@{self.top_k}', f'precision@{self.top_k}', 
                           f'mrr@{self.top_k}', f'ndcg@{self.top_k}', 'keyword_hit_rate']:
            if metric_name in layered_metrics and metric_name in naive_metrics:
                l_val = layered_metrics[metric_name]['mean']
                n_val = naive_metrics[metric_name]['mean']
                diff = l_val - n_val
                diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                report.append(f"| {metric_name} | {l_val:.4f} | {n_val:.4f} | {diff_str} |")
        
        # LLM评估指标
        report.append("| | | | |")
        report.append("| **LLM评估指标** | | | |")
        for metric_name in ['llm_precision', 'llm_recall', 'llm_f1']:
            if metric_name in layered_metrics and metric_name in naive_metrics:
                l_val = layered_metrics[metric_name]['mean']
                n_val = naive_metrics[metric_name]['mean']
                diff = l_val - n_val
                diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                report.append(f"| {metric_name} | {l_val:.4f} | {n_val:.4f} | {diff_str} |")
        report.append("")
        
        # 分场景对比
        report.append("## 三、分场景指标对比\n")
        
        for scenario, desc in scenario_desc.items():
            report.append(f"### {desc}\n")
            report.append("| 指标 | 分层切分 | 分隔符切分 | 差异 |")
            report.append("|------|----------|------------|------|")
            
            layered_scenario = all_results['layered']['by_scenario'].get(scenario, [])
            naive_scenario = all_results['naive']['by_scenario'].get(scenario, [])
            
            l_metrics = self.aggregate_metrics(layered_scenario)
            n_metrics = self.aggregate_metrics(naive_scenario)
            
            # 基础指标
            for metric_name in [f'recall@{self.top_k}', f'precision@{self.top_k}', 
                               f'mrr@{self.top_k}', f'ndcg@{self.top_k}', 'keyword_hit_rate']:
                if metric_name in l_metrics and metric_name in n_metrics:
                    l_val = l_metrics[metric_name]['mean']
                    n_val = n_metrics[metric_name]['mean']
                    diff = l_val - n_val
                    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                    report.append(f"| {metric_name} | {l_val:.4f} | {n_val:.4f} | {diff_str} |")
            
            # LLM评估指标
            for metric_name in ['llm_precision', 'llm_recall', 'llm_f1']:
                if metric_name in l_metrics and metric_name in n_metrics:
                    l_val = l_metrics[metric_name]['mean']
                    n_val = n_metrics[metric_name]['mean']
                    diff = l_val - n_val
                    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                    report.append(f"| {metric_name} | {l_val:.4f} | {n_val:.4f} | {diff_str} |")
            report.append("")
        
        # 详细分析
        report.append("## 四、详细分析\n")
        
        # 找出差异最大的测试用例
        report.append("### 分层切分优势明显的测试用例\n")
        report.append("| 测试ID | 查询 | 场景 | 分层切分 Recall | 分隔符切分 Recall | 差异 |")
        report.append("|--------|------|------|----------------|------------------|------|")
        
        improvements = []
        for l, n in zip(all_results['layered']['all'], all_results['naive']['all']):
            l_recall = l['metrics'][f'recall@{self.top_k}']
            n_recall = n['metrics'][f'recall@{self.top_k}']
            diff = l_recall - n_recall
            if diff > 0.3:  # 差异大于0.3
                improvements.append({
                    'id': l['test_id'],
                    'query': l['query'],
                    'scenario': l['scenario'],
                    'layered': l_recall,
                    'naive': n_recall,
                    'diff': diff
                })
        
        # 按差异排序，取前10个
        improvements.sort(key=lambda x: x['diff'], reverse=True)
        for imp in improvements[:10]:
            report.append(f"| {imp['id']} | {imp['query'][:20]}... | {imp['scenario']} | "
                         f"{imp['layered']:.4f} | {imp['naive']:.4f} | +{imp['diff']:.4f} |")
        report.append("")
        
        # 结论
        report.append("## 五、结论\n")
        
        # 计算总体差异
        l_recall_mean = layered_metrics[f'recall@{self.top_k}']['mean']
        n_recall_mean = naive_metrics[f'recall@{self.top_k}']['mean']
        
        if l_recall_mean > n_recall_mean:
            improvement = (l_recall_mean - n_recall_mean) / n_recall_mean * 100
            report.append(f"- 分层智能切分在召回率上**优于**分隔符切分 {improvement:.1f}%")
        else:
            decline = (n_recall_mean - l_recall_mean) / n_recall_mean * 100
            report.append(f"- 分层智能切分在召回率上**劣于**分隔符切分 {decline:.1f}%")
        
        # 场景分析
        report.append("\n### 各场景表现分析\n")
        for scenario, desc in scenario_desc.items():
            l_metrics = self.aggregate_metrics(
                all_results['layered']['by_scenario'].get(scenario, [])
            )
            n_metrics = self.aggregate_metrics(
                all_results['naive']['by_scenario'].get(scenario, [])
            )
            
            if l_metrics and n_metrics:
                l_recall = l_metrics[f'recall@{self.top_k}']['mean']
                n_recall = n_metrics[f'recall@{self.top_k}']['mean']
                
                if l_recall > n_recall:
                    report.append(f"- **{desc}**: 分层切分更优 (+{(l_recall-n_recall):.4f})")
                else:
                    report.append(f"- **{desc}**: 分隔符切分更优 (+{(n_recall-l_recall):.4f})")
        
        report.append("\n---\n")
        report.append("*报告由 RAG 系统切分对比测试工具自动生成*\n")
        
        return '\n'.join(report)
    
    def save_results(self, all_results: Dict, report: str):
        """保存结果"""
        # 保存详细结果
        results_path = self.output_dir / 'detailed_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"详细结果已保存: {results_path}")
        
        # 保存报告
        report_path = self.output_dir / 'results.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"对比报告已保存: {report_path}")


def main():
    """主函数"""
    # 配置路径
    test_cases_path = PROJECT_ROOT / 'tests' / 'chunking_comparison' / 'test_cases_200.json'
    docs_dir = PROJECT_ROOT / 'data' / 'docs'
    output_dir = PROJECT_ROOT / 'tests' / 'chunking_comparison'
    
    # 创建测试实例
    test = ChunkingComparisonTest(
        test_cases_path=str(test_cases_path),
        docs_dir=str(docs_dir),
        output_dir=str(output_dir),
        chunk_size=512,
        overlap=0.1,
        top_k=10
    )
    
    # 运行测试
    all_results = test.run_comparison()
    
    # 生成报告
    report = test.generate_report(all_results)
    
    # 保存结果
    test.save_results(all_results, report)
    
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)


if __name__ == '__main__':
    main()
